"""Data augmentation methods specialized for degradation sequence modeling.

Small run-to-failure datasets (such as PHM2012) are vulnerable to overfitting.
This module provides augmentation operators that preserve broad degradation
trends while injecting realistic variability:

- Gaussian noise
- Time warping
- Window slicing / random crop-resize
- Mixup between degradation windows

The code is designed for both offline and on-the-fly usage in dataloaders.
All augmentors operate on tensors shaped ``(batch, time, sensors)`` unless
explicitly documented otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GaussianNoiseConfig:
    """Hyperparameters for Gaussian noise augmentation.

    Attributes:
        std: Base standard deviation of additive noise.
        per_sensor_scale: If True, noise std is modulated per sensor based on
            sensor signal standard deviation.
        clamp_min: Optional lower clamp after augmentation.
        clamp_max: Optional upper clamp after augmentation.
    """

    std: float = 0.02
    per_sensor_scale: bool = True
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None


@dataclass
class TimeWarpConfig:
    """Hyperparameters for temporal warping augmentation.

    Attributes:
        max_warp: Maximum relative warping magnitude.
        num_control_points: Number of spline-like control points.
        interpolation_mode: Interpolation mode for resampling.
    """

    max_warp: float = 0.2
    num_control_points: int = 4
    interpolation_mode: str = "linear"


@dataclass
class WindowSlicingConfig:
    """Hyperparameters for random window slicing.

    Attributes:
        min_slice_ratio: Minimum retained ratio before resizing back.
        max_slice_ratio: Maximum retained ratio before resizing back.
    """

    min_slice_ratio: float = 0.6
    max_slice_ratio: float = 0.95


@dataclass
class MixupConfig:
    """Hyperparameters for mixup augmentation.

    Attributes:
        alpha: Beta distribution parameter.
        probability: Probability to apply mixup on a batch.
    """

    alpha: float = 0.3
    probability: float = 0.5


class GaussianNoiseAugmentor:
    """Apply additive Gaussian noise to multi-sensor degradation windows.

    Input shape:
        ``x``: ``(batch, time, sensors)``

    Output shape:
        same as input
    """

    def __init__(self, config: Optional[GaussianNoiseConfig] = None):
        self.config = config or GaussianNoiseConfig()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentation.

        Args:
            x: Input tensor ``(B, T, S)``.

        Returns:
            Augmented tensor with same shape.
        """

        if x.ndim != 3:
            raise ValueError("GaussianNoiseAugmentor expects (B,T,S)")

        cfg = self.config
        if cfg.per_sensor_scale:
            sensor_std = x.std(dim=1, keepdim=True)
            noise_std = cfg.std * (sensor_std + 1e-6)
        else:
            noise_std = torch.full_like(x, cfg.std)

        noise = torch.randn_like(x) * noise_std
        x_aug = x + noise

        if cfg.clamp_min is not None or cfg.clamp_max is not None:
            x_aug = torch.clamp(
                x_aug,
                min=cfg.clamp_min if cfg.clamp_min is not None else float("-inf"),
                max=cfg.clamp_max if cfg.clamp_max is not None else float("inf"),
            )
        return x_aug


class TimeWarpAugmentor:
    """Randomly warp temporal axis using smooth control point perturbations.

    Input shape:
        ``x``: ``(batch, time, sensors)``

    Output shape:
        same as input

    Algorithm explanation:
    1. Sample random offsets for a small set of control points.
    2. Interpolate offsets to full temporal resolution.
    3. Create warped sampling grid.
    4. Resample each sensor via linear interpolation.
    """

    def __init__(self, config: Optional[TimeWarpConfig] = None):
        self.config = config or TimeWarpConfig()

    def _build_warp_grid(self, time_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Construct a monotonic warped grid in [0, time_steps - 1]."""

        cfg = self.config
        cp = cfg.num_control_points

        # Control point positions are uniformly spaced across timeline.
        cp_pos = torch.linspace(0, time_steps - 1, steps=cp, device=device, dtype=dtype)

        # Random offsets perturb local temporal speed while preserving order.
        max_shift = cfg.max_warp * time_steps
        cp_shift = (torch.rand(cp, device=device, dtype=dtype) * 2 - 1) * max_shift

        # Enforce boundary conditions to keep first/last point stable.
        cp_shift[0] = 0.0
        cp_shift[-1] = 0.0

        warped_cp = cp_pos + cp_shift

        # Sort to maintain monotonicity and avoid fold-over artifacts.
        warped_cp = torch.sort(warped_cp).values

        full_pos = torch.linspace(0, time_steps - 1, steps=time_steps, device=device, dtype=dtype)

        # Piecewise linear interpolation for dense warp map.
        # `torch.interp` is unavailable in some versions, so we use numpy and
        # convert back while preserving dtype/device.
        grid_np = np.interp(
            full_pos.detach().cpu().numpy(),
            cp_pos.detach().cpu().numpy(),
            warped_cp.detach().cpu().numpy(),
        )
        grid = torch.from_numpy(grid_np).to(device=device, dtype=dtype)
        return grid

    def _resample_sequence(self, seq: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Resample sequence according to warped grid.

        Args:
            seq: Tensor ``(T, S)``.
            grid: Tensor ``(T,)`` with source indices.

        Returns:
            Resampled tensor ``(T, S)``.
        """

        t, s = seq.shape

        left = torch.floor(grid).long().clamp(0, t - 1)
        right = (left + 1).clamp(0, t - 1)
        alpha = (grid - left.float()).unsqueeze(-1)

        seq_left = seq[left]
        seq_right = seq[right]

        return (1 - alpha) * seq_left + alpha * seq_right

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal warping to each sequence in batch."""

        if x.ndim != 3:
            raise ValueError("TimeWarpAugmentor expects (B,T,S)")

        b, t, s = x.shape
        warped: List[torch.Tensor] = []

        for i in range(b):
            grid = self._build_warp_grid(t, device=x.device, dtype=x.dtype)
            warped_i = self._resample_sequence(x[i], grid)
            warped.append(warped_i)

        return torch.stack(warped, dim=0)


class WindowSlicingAugmentor:
    """Randomly slice subsections and resize back to original window length.

    Input shape:
        ``x``: ``(batch, time, sensors)``

    Output shape:
        same as input

    This operation simulates partial observation and mild sampling-rate drift.
    """

    def __init__(self, config: Optional[WindowSlicingConfig] = None):
        self.config = config or WindowSlicingConfig()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply window slicing and interpolation.

        Args:
            x: Input tensor ``(B, T, S)``.

        Returns:
            Augmented tensor ``(B, T, S)``.
        """

        if x.ndim != 3:
            raise ValueError("WindowSlicingAugmentor expects (B,T,S)")

        b, t, s = x.shape
        cfg = self.config
        out: List[torch.Tensor] = []

        for i in range(b):
            ratio = np.random.uniform(cfg.min_slice_ratio, cfg.max_slice_ratio)
            slice_len = max(2, int(t * ratio))
            start = np.random.randint(0, t - slice_len + 1)
            seg = x[i, start : start + slice_len]  # (slice_len, S)

            # Interpolate back to original temporal length.
            seg_ch = seg.transpose(0, 1).unsqueeze(0)  # (1, S, slice_len)
            seg_resized = F.interpolate(seg_ch, size=t, mode="linear", align_corners=False)
            seg_resized = seg_resized.squeeze(0).transpose(0, 1)
            out.append(seg_resized)

        return torch.stack(out, dim=0)


class MixupAugmentor:
    """Mixup augmentation for regression tasks (features + labels).

    Input shapes:
        x: ``(batch, time, sensors)``
        y: ``(batch, 1)`` or ``(batch,)``

    Output shapes:
        x_mix: same as x
        y_mix: same as y (after reshape)

    Algorithm:
        lambda ~ Beta(alpha, alpha)
        x' = lambda * x_i + (1-lambda) * x_j
        y' = lambda * y_i + (1-lambda) * y_j
    """

    def __init__(self, config: Optional[MixupConfig] = None):
        self.config = config or MixupConfig()

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup on batch.

        Args:
            x: Feature tensor ``(B,T,S)``.
            y: Target tensor ``(B,1)`` or ``(B,)``.

        Returns:
            Tuple ``(x_mix, y_mix)``.
        """

        if x.ndim != 3:
            raise ValueError("MixupAugmentor expects x with shape (B,T,S)")

        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim != 2:
            raise ValueError("MixupAugmentor expects y with shape (B,) or (B,1)")

        cfg = self.config
        if np.random.rand() > cfg.probability:
            return x, y

        b = x.size(0)
        idx = torch.randperm(b, device=x.device)

        # Sample lambda and enforce symmetric stronger mixing around 0.5.
        lam = np.random.beta(cfg.alpha, cfg.alpha)
        lam = max(lam, 1.0 - lam)

        x_mix = lam * x + (1.0 - lam) * x[idx]
        y_mix = lam * y + (1.0 - lam) * y[idx]
        return x_mix, y_mix


@dataclass
class AugmentationPipelineConfig:
    """Configuration controlling combined augmentation pipeline behavior."""

    noise_prob: float = 0.6
    warp_prob: float = 0.4
    slice_prob: float = 0.4
    mixup_prob: float = 0.5


class DegradationAugmentationPipeline:
    """Composable augmentation pipeline for RUL model training.

    Input shapes:
        x: ``(batch, time, sensors)``
        y: ``(batch, 1)``

    Output shapes:
        x_aug: ``(batch, time, sensors)``
        y_aug: ``(batch, 1)``

    Pipeline order:
        1) Gaussian noise
        2) Time warp
        3) Window slicing
        4) Mixup
    """

    def __init__(
        self,
        pipeline_cfg: Optional[AugmentationPipelineConfig] = None,
        noise_cfg: Optional[GaussianNoiseConfig] = None,
        warp_cfg: Optional[TimeWarpConfig] = None,
        slice_cfg: Optional[WindowSlicingConfig] = None,
        mixup_cfg: Optional[MixupConfig] = None,
    ):
        self.pipeline_cfg = pipeline_cfg or AugmentationPipelineConfig()
        self.noise = GaussianNoiseAugmentor(noise_cfg)
        self.warp = TimeWarpAugmentor(warp_cfg)
        self.slice = WindowSlicingAugmentor(slice_cfg)

        mix_cfg = mixup_cfg or MixupConfig()
        mix_cfg.probability = self.pipeline_cfg.mixup_prob
        self.mixup = MixupAugmentor(mix_cfg)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply pipeline with stochastic gates.

        Args:
            x: Feature tensor ``(B,T,S)``.
            y: Label tensor ``(B,1)``.

        Returns:
            Augmented feature-label pair.
        """

        cfg = self.pipeline_cfg
        x_aug = x

        if np.random.rand() < cfg.noise_prob:
            x_aug = self.noise(x_aug)

        if np.random.rand() < cfg.warp_prob:
            x_aug = self.warp(x_aug)

        if np.random.rand() < cfg.slice_prob:
            x_aug = self.slice(x_aug)

        x_aug, y_aug = self.mixup(x_aug, y)
        return x_aug, y_aug


class AugmentedBatchCollator:
    """Collate function wrapper that injects augmentation at dataloader level.

    This class can be passed as ``collate_fn`` to a PyTorch DataLoader for
    on-the-fly augmentation.
    """

    def __init__(self, pipeline: Optional[DegradationAugmentationPipeline] = None):
        self.pipeline = pipeline or DegradationAugmentationPipeline()

    def __call__(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[str]]:
        """Collate and augment a list of dataset samples.

        Args:
            batch: Sequence of dict samples with keys ``x``, ``y``, ``id``.

        Returns:
            Dict with batched/augmented tensors and id list.
        """

        xs = torch.stack([item["x"] for item in batch], dim=0)
        ys = torch.stack([item["y"] for item in batch], dim=0)
        ids = [str(item["id"]) for item in batch]

        xs_aug, ys_aug = self.pipeline(xs, ys)

        return {
            "x": xs_aug,
            "y": ys_aug,
            "id": ids,
        }


def visualize_augmentation_effect(
    x: torch.Tensor,
    augmentor: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Generate before/after tensors for visualization scripts.

    Args:
        x: Input tensor ``(B,T,S)``.
        augmentor: Callable that maps ``x -> x_aug``.

    Returns:
        Dictionary with ``original`` and ``augmented`` tensors.
    """

    x_aug = augmentor(x)
    return {"original": x.detach().cpu(), "augmented": x_aug.detach().cpu()}


def compute_augmentation_statistics(x: torch.Tensor, x_aug: torch.Tensor) -> Dict[str, float]:
    """Compute quantitative shift metrics between original and augmented data.

    Args:
        x: Original tensor ``(B,T,S)``.
        x_aug: Augmented tensor with same shape.

    Returns:
        Dictionary with MAE, RMSE, relative energy change.
    """

    if x.shape != x_aug.shape:
        raise ValueError("x and x_aug must have identical shape")

    diff = x_aug - x
    mae = float(torch.mean(torch.abs(diff)).item())
    rmse = float(torch.sqrt(torch.mean(diff.pow(2))).item())

    energy_orig = torch.mean(x.pow(2)).item() + 1e-8
    energy_aug = torch.mean(x_aug.pow(2)).item()
    rel_energy = float((energy_aug - energy_orig) / energy_orig)

    return {
        "mae": mae,
        "rmse": rmse,
        "relative_energy_change": rel_energy,
    }


def demo_augmentation_pipeline(
    batch: int = 8,
    time_steps: int = 40,
    sensors: int = 6,
    seed: int = 42,
) -> Dict[str, float]:
    """Run synthetic demo of all augmentation operators.

    Args:
        batch: Number of synthetic windows.
        time_steps: Window length.
        sensors: Number of channels.
        seed: Reproducibility seed.

    Returns:
        Dictionary with shift statistics.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Construct synthetic degradation trajectories with monotonic trend + noise.
    t = torch.linspace(0, 1, time_steps)
    trend = t.unsqueeze(-1) * torch.linspace(0.5, 1.5, sensors).unsqueeze(0)
    base = trend + 0.1 * torch.sin(2 * np.pi * 3 * t).unsqueeze(-1)
    x = base.unsqueeze(0).repeat(batch, 1, 1)
    x = x + 0.02 * torch.randn_like(x)

    y = torch.linspace(120, 10, batch).unsqueeze(-1)

    pipeline = DegradationAugmentationPipeline()
    x_aug, y_aug = pipeline(x, y)

    stats = compute_augmentation_statistics(x, x_aug)
    stats["label_shift_mae"] = float(torch.mean(torch.abs(y_aug - y)).item())
    return stats


# -----------------------------
# Compatibility helper section.
# -----------------------------

def maybe_apply_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    enabled: bool,
    pipeline: Optional[DegradationAugmentationPipeline] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conditionally apply augmentation.

    Args:
        x: Input features ``(B,T,S)``.
        y: Labels ``(B,1)``.
        enabled: If False, returns inputs untouched.
        pipeline: Optional custom pipeline.

    Returns:
        Tuple of processed tensors.
    """

    if not enabled:
        return x, y
    pipe = pipeline or DegradationAugmentationPipeline()
    return pipe(x, y)


def build_default_augmentation_pipeline() -> DegradationAugmentationPipeline:
    """Create a default pipeline with recommended hyperparameters.

    Returns:
        Instantiated ``DegradationAugmentationPipeline``.
    """

    return DegradationAugmentationPipeline(
        pipeline_cfg=AugmentationPipelineConfig(
            noise_prob=0.7,
            warp_prob=0.5,
            slice_prob=0.5,
            mixup_prob=0.5,
        ),
        noise_cfg=GaussianNoiseConfig(std=0.02, per_sensor_scale=True),
        warp_cfg=TimeWarpConfig(max_warp=0.15, num_control_points=4),
        slice_cfg=WindowSlicingConfig(min_slice_ratio=0.65, max_slice_ratio=0.95),
        mixup_cfg=MixupConfig(alpha=0.2, probability=0.5),
    )
