"""Time-frequency feature extraction utilities for vibration-based RUL prediction.

This module provides research-oriented implementations of two canonical
transforms for non-stationary mechanical signals:

1. Short-Time Fourier Transform (STFT)
2. Continuous Wavelet Transform (CWT)

In bearing prognostics, degradation signatures are often localized in both time
and frequency. Pure time-domain models can miss these patterns, especially under
non-stationary operating conditions. The classes in this module convert raw
windowed vibration signals of shape ``(batch, time, sensors)`` into
spectrogram-like tensors suitable for CNN/Transformer models.

Design goals:
- Explicit and readable implementation for reproducibility.
- Rich docstrings and type hints for academic code quality.
- Compatibility with PyTorch pipelines and dataloaders.
- Robust handling of multi-sensor windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class STFTConfig:
    """Configuration for STFT feature extraction.

    Attributes:
        n_fft: FFT length used in each STFT frame.
        hop_length: Stride between adjacent STFT frames.
        win_length: Window size for each frame. If ``None``, defaults to
            ``n_fft``.
        window: Window function type. Supported: ``hann``, ``hamming``.
        center: Whether to pad input so frame is centered.
        normalized: Whether to normalize STFT output.
        log_amplitude: Whether to apply log(1 + x) compression.
        eps: Numerical stability epsilon.
    """

    n_fft: int = 64
    hop_length: int = 16
    win_length: Optional[int] = None
    window: str = "hann"
    center: bool = True
    normalized: bool = False
    log_amplitude: bool = True
    eps: float = 1e-8


@dataclass
class CWTConfig:
    """Configuration for CWT feature extraction.

    Attributes:
        num_scales: Number of scales in the wavelet bank.
        min_scale: Smallest wavelet scale.
        max_scale: Largest wavelet scale.
        wavelet_size: Support length of each discrete wavelet filter.
        wavelet_w0: Central frequency parameter of Morlet-like wavelet.
        stride: Temporal stride after convolution.
        log_amplitude: Whether to apply log(1 + x) compression.
        eps: Numerical stability epsilon.
    """

    num_scales: int = 32
    min_scale: float = 1.0
    max_scale: float = 32.0
    wavelet_size: int = 129
    wavelet_w0: float = 6.0
    stride: int = 1
    log_amplitude: bool = True
    eps: float = 1e-8


@dataclass
class TimeFrequencyOutput:
    """Container for extracted time-frequency features.

    Attributes:
        stft: STFT magnitude tensor of shape
            ``(batch, sensors, freq_bins, time_bins)``.
        cwt: CWT magnitude tensor of shape
            ``(batch, sensors, num_scales, time_bins)``.
        fused: Concatenated feature tensor of shape
            ``(batch, sensors, fused_channels, time_bins)`` where
            ``fused_channels = freq_bins + num_scales`` after temporal alignment.
    """

    stft: torch.Tensor
    cwt: torch.Tensor
    fused: torch.Tensor


class STFTFeatureExtractor(nn.Module):
    """Extract STFT spectrogram features from multi-sensor time windows.

    Input shape:
        ``x``: ``(batch, time, sensors)``

    Output shape:
        Tensor of shape ``(batch, sensors, freq_bins, time_bins)``

    Algorithm overview:
    1. Reshape sensors as independent channels in batch space.
    2. Apply ``torch.stft`` to each 1D sequence.
    3. Convert complex spectrum to magnitude.
    4. Optionally apply logarithmic amplitude compression.
    5. Restore original batch/sensor dimensions.
    """

    def __init__(self, config: Optional[STFTConfig] = None):
        super().__init__()
        self.config = config or STFTConfig()

        if self.config.window.lower() == "hann":
            base_window = torch.hann_window(self.config.win_length or self.config.n_fft)
        elif self.config.window.lower() == "hamming":
            base_window = torch.hamming_window(self.config.win_length or self.config.n_fft)
        else:
            raise ValueError(f"Unsupported window type: {self.config.window}")

        # Register as buffer so it automatically moves with .to(device).
        self.register_buffer("window_tensor", base_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude features.

        Args:
            x: Input tensor with shape ``(batch, time, sensors)``.

        Returns:
            STFT magnitude tensor with shape
            ``(batch, sensors, freq_bins, time_bins)``.
        """

        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, S), got {tuple(x.shape)}")

        batch, time_steps, sensors = x.shape

        # Move sensor axis into batch axis so each sensor stream receives
        # independent STFT processing while keeping implementation simple.
        x_flat = x.permute(0, 2, 1).reshape(batch * sensors, time_steps)

        spec = torch.stft(
            x_flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length or self.config.n_fft,
            window=self.window_tensor.to(x.device),
            center=self.config.center,
            normalized=self.config.normalized,
            onesided=True,
            return_complex=True,
        )

        mag = torch.abs(spec)
        if self.config.log_amplitude:
            mag = torch.log1p(mag + self.config.eps)

        # Reshape back: (B*S, F, T') -> (B, S, F, T').
        mag = mag.view(batch, sensors, mag.shape[-2], mag.shape[-1])
        return mag


class CWTFeatureExtractor(nn.Module):
    """Extract CWT-like scalogram features using a Morlet wavelet filter bank.

    Input shape:
        ``x``: ``(batch, time, sensors)``

    Output shape:
        Tensor of shape ``(batch, sensors, num_scales, time_bins)``

    Notes:
    - This implementation constructs real-valued Morlet-inspired kernels and
      applies grouped 1D convolution.
    - Although simplified compared to analytic CWT packages, it is stable,
      differentiable, and suitable for deep-learning pipelines.
    """

    def __init__(self, config: Optional[CWTConfig] = None):
        super().__init__()
        self.config = config or CWTConfig()
        self.register_buffer("wavelet_bank", self._build_wavelet_bank())

    def _build_wavelet_bank(self) -> torch.Tensor:
        """Build a bank of scaled Morlet-like wavelet kernels.

        Returns:
            Tensor of shape ``(num_scales, wavelet_size)``.

        Algorithm explanation:
        - Create a symmetric time support around zero.
        - Generate logarithmically spaced scales for broad spectral coverage.
        - For each scale, compute a Morlet-like envelope * cosine carrier.
        - Normalize each kernel energy to reduce scale bias.
        """

        cfg = self.config
        if cfg.wavelet_size % 2 == 0:
            raise ValueError("wavelet_size should be odd for symmetric support")

        t = torch.linspace(
            -(cfg.wavelet_size // 2),
            cfg.wavelet_size // 2,
            cfg.wavelet_size,
            dtype=torch.float32,
        )
        scales = torch.logspace(
            np.log10(cfg.min_scale),
            np.log10(cfg.max_scale),
            steps=cfg.num_scales,
            dtype=torch.float32,
        )

        kernels: List[torch.Tensor] = []
        for scale in scales:
            tau = t / scale

            # Gaussian envelope controls temporal localization.
            envelope = torch.exp(-0.5 * tau.pow(2))

            # Cosine carrier places frequency content near wavelet_w0/scale.
            carrier = torch.cos(cfg.wavelet_w0 * tau)
            kernel = envelope * carrier

            # Remove mean to enforce approximate zero-mean wavelet behavior.
            kernel = kernel - kernel.mean()

            # Normalize L2 energy so amplitudes are comparable across scales.
            kernel = kernel / torch.sqrt(torch.sum(kernel.pow(2)) + cfg.eps)
            kernels.append(kernel)

        bank = torch.stack(kernels, dim=0)
        return bank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CWT-like features by convolving wavelet bank over time.

        Args:
            x: Input tensor of shape ``(batch, time, sensors)``.

        Returns:
            CWT magnitude tensor of shape
            ``(batch, sensors, num_scales, time_bins)``.
        """

        if x.ndim != 3:
            raise ValueError(f"Expected shape (B, T, S), got {tuple(x.shape)}")

        batch, time_steps, sensors = x.shape
        cfg = self.config

        # Convert to channel-first for conv1d: (B, S, T).
        x_ch = x.permute(0, 2, 1)

        # We apply the same wavelet bank to every sensor by grouped convolution.
        # Weight shape for grouped conv1d is
        # (out_channels, in_channels/groups, kernel_size).
        bank = self.wavelet_bank.to(x.device)
        weight = bank.unsqueeze(1).repeat(sensors, 1, 1)

        # Arrange outputs as sensors * scales channels, each sensor processed independently.
        conv = F.conv1d(
            x_ch,
            weight=weight,
            bias=None,
            stride=cfg.stride,
            padding=cfg.wavelet_size // 2,
            groups=sensors,
        )

        # Reshape (B, S*num_scales, T') -> (B, S, num_scales, T').
        conv = conv.view(batch, sensors, cfg.num_scales, conv.shape[-1])
        mag = torch.abs(conv)
        if cfg.log_amplitude:
            mag = torch.log1p(mag + cfg.eps)
        return mag


class TimeFrequencyProjector(nn.Module):
    """Project and align time-frequency features to fixed embedding channels.

    Input shapes:
        stft: ``(batch, sensors, stft_bins, t_stft)``
        cwt: ``(batch, sensors, cwt_bins, t_cwt)``

    Output shape:
        ``(batch, sensors, out_channels, time_bins)`` where ``time_bins`` is
        configurable via interpolation.

    This module is useful when downstream models require a uniform feature map
    dimension regardless of STFT/CWT hyperparameters.
    """

    def __init__(self, stft_bins: int, cwt_bins: int, out_channels: int = 128):
        super().__init__()

        self.stft_proj = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cwt_proj = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.stft_bins = stft_bins
        self.cwt_bins = cwt_bins
        self.out_channels = out_channels

    def _reshape_for_cnn(self, feat: torch.Tensor) -> torch.Tensor:
        """Flatten batch and sensor dimensions for 2D conv processing.

        Args:
            feat: Tensor ``(B, S, F, T)``.

        Returns:
            Tensor ``(B*S, 1, F, T)``.
        """

        b, s, f, t = feat.shape
        return feat.view(b * s, 1, f, t)

    def forward(self, stft: torch.Tensor, cwt: torch.Tensor, target_time: Optional[int] = None) -> torch.Tensor:
        """Fuse STFT and CWT features into projected embedding maps.

        Args:
            stft: STFT tensor of shape ``(B, S, F_stft, T_stft)``.
            cwt: CWT tensor of shape ``(B, S, F_cwt, T_cwt)``.
            target_time: Optional target temporal size for interpolation.

        Returns:
            Projected fusion tensor of shape ``(B, S, out_channels, T_target)``.
        """

        if stft.ndim != 4 or cwt.ndim != 4:
            raise ValueError("stft and cwt must be 4D tensors")

        b, s, _, _ = stft.shape

        stft_2d = self._reshape_for_cnn(stft)
        cwt_2d = self._reshape_for_cnn(cwt)

        stft_feat = self.stft_proj(stft_2d)
        cwt_feat = self.cwt_proj(cwt_2d)

        # Match spatial sizes before channel-wise concatenation.
        if stft_feat.shape[-2:] != cwt_feat.shape[-2:]:
            cwt_feat = F.interpolate(cwt_feat, size=stft_feat.shape[-2:], mode="bilinear", align_corners=False)

        fused_2d = torch.cat([stft_feat, cwt_feat], dim=1)
        fused_2d = self.fusion_proj(fused_2d)

        if target_time is not None and fused_2d.shape[-1] != target_time:
            fused_2d = F.interpolate(
                fused_2d,
                size=(fused_2d.shape[-2], target_time),
                mode="bilinear",
                align_corners=False,
            )

        # Average over frequency axis to get channel-time representation.
        fused_2d = fused_2d.mean(dim=-2)

        # Restore (B, S, C, T).
        out = fused_2d.view(b, s, self.out_channels, fused_2d.shape[-1])
        return out


class TimeFrequencyFeatureExtractor(nn.Module):
    """Combined STFT + CWT extractor returning raw and fused representations.

    Input shape:
        ``(batch, time, sensors)``

    Output:
        ``TimeFrequencyOutput`` object containing STFT, CWT, and concatenated
        spectrogram features.
    """

    def __init__(
        self,
        stft_config: Optional[STFTConfig] = None,
        cwt_config: Optional[CWTConfig] = None,
    ):
        super().__init__()
        self.stft_extractor = STFTFeatureExtractor(stft_config)
        self.cwt_extractor = CWTFeatureExtractor(cwt_config)

    def forward(self, x: torch.Tensor) -> TimeFrequencyOutput:
        """Extract both STFT and CWT feature maps and concatenate them.

        Args:
            x: Input tensor of shape ``(B, T, S)``.

        Returns:
            ``TimeFrequencyOutput`` containing STFT/CWT/fused tensors.
        """

        stft = self.stft_extractor(x)
        cwt = self.cwt_extractor(x)

        # Align temporal bins so channel concatenation is well-defined.
        if stft.shape[-1] != cwt.shape[-1]:
            cwt = F.interpolate(cwt, size=(cwt.shape[-2], stft.shape[-1]), mode="bilinear", align_corners=False)

        fused = torch.cat([stft, cwt], dim=2)
        return TimeFrequencyOutput(stft=stft, cwt=cwt, fused=fused)


def flatten_time_frequency_features(feat: torch.Tensor) -> torch.Tensor:
    """Flatten spectrogram features for transformer-compatible sequences.

    Args:
        feat: Tensor with shape ``(B, S, F, T)``.

    Returns:
        Tensor with shape ``(B, T, S*F)``.

    Explanation:
        Many sequence models expect features as ``(batch, time, channels)``.
        This utility merges sensor and frequency dimensions per time bin.
    """

    if feat.ndim != 4:
        raise ValueError(f"Expected (B,S,F,T), got {tuple(feat.shape)}")

    b, s, f, t = feat.shape
    return feat.permute(0, 3, 1, 2).reshape(b, t, s * f)


def normalize_spectrogram(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize spectrogram over frequency and time axes per sample/sensor.

    Args:
        feat: Tensor with shape ``(B, S, F, T)``.
        eps: Stability term.

    Returns:
        Normalized tensor with same shape.
    """

    if feat.ndim != 4:
        raise ValueError("normalize_spectrogram expects a 4D tensor")

    mean = feat.mean(dim=(-2, -1), keepdim=True)
    std = feat.std(dim=(-2, -1), keepdim=True)
    return (feat - mean) / (std + eps)


def compute_spectral_statistics(feat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute compact spectral descriptors from time-frequency maps.

    Args:
        feat: Tensor with shape ``(B, S, F, T)``.

    Returns:
        Dictionary with descriptors:
        - spectral_energy: ``(B, S)``
        - spectral_centroid: ``(B, S)``
        - temporal_modulation: ``(B, S)``
    """

    if feat.ndim != 4:
        raise ValueError("compute_spectral_statistics expects 4D features")

    b, s, f, t = feat.shape
    energy = feat.mean(dim=(-2, -1))

    freq_axis = torch.linspace(0, 1, steps=f, device=feat.device, dtype=feat.dtype).view(1, 1, f, 1)
    weighted = (feat * freq_axis).sum(dim=(-2, -1))
    denom = feat.sum(dim=(-2, -1)) + 1e-6
    centroid = weighted / denom

    temporal_std = feat.mean(dim=2).std(dim=-1)

    return {
        "spectral_energy": energy,
        "spectral_centroid": centroid,
        "temporal_modulation": temporal_std,
    }


def demo_time_frequency_pipeline(
    batch: int = 2,
    time_steps: int = 40,
    sensors: int = 6,
    seed: int = 42,
) -> Dict[str, Tuple[int, ...]]:
    """Run a synthetic demo and return output shapes.

    This helper is useful for quick sanity checks and unit-test-like behavior.

    Args:
        batch: Number of synthetic samples.
        time_steps: Sequence length.
        sensors: Number of sensor channels.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping tensor names to shape tuples.
    """

    torch.manual_seed(seed)

    # Synthetic vibration-like waveform: sinusoids + random perturbation.
    t = torch.linspace(0, 1, time_steps)
    base = torch.stack([torch.sin(2 * np.pi * (i + 1) * t) for i in range(sensors)], dim=-1)
    x = base.unsqueeze(0).repeat(batch, 1, 1)
    x = x + 0.05 * torch.randn_like(x)

    extractor = TimeFrequencyFeatureExtractor()
    out = extractor(x)

    flat = flatten_time_frequency_features(out.fused)
    stats = compute_spectral_statistics(out.fused)

    return {
        "input": tuple(x.shape),
        "stft": tuple(out.stft.shape),
        "cwt": tuple(out.cwt.shape),
        "fused": tuple(out.fused.shape),
        "flattened": tuple(flat.shape),
        "energy": tuple(stats["spectral_energy"].shape),
    }
