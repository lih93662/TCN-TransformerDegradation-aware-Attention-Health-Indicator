"""Health Indicator (HI) module.

HI can be computed from:
- handcrafted statistical descriptors (`stats`)
- learned temporal features from the TCN pathway (`tcn`)
- hybrid concatenation of both (`hybrid`)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class StatisticalFeatureExtractor(nn.Module):
    """Compute RMS, variance, kurtosis, and skewness over temporal dimension.

    Input shape:
        (batch, time, sensors)

    Output shape:
        (batch, sensors * 4)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,T,S), got {tuple(x.shape)}")

        mean = x.mean(dim=1)
        centered = x - mean.unsqueeze(1)
        var = centered.pow(2).mean(dim=1)
        std = torch.sqrt(var + self.eps)

        rms = torch.sqrt((x.pow(2).mean(dim=1)) + self.eps)
        skewness = centered.pow(3).mean(dim=1) / (std.pow(3) + self.eps)
        kurtosis = centered.pow(4).mean(dim=1) / (std.pow(4) + self.eps)

        # Stack features per sensor, then flatten.
        feat = torch.stack([rms, var, kurtosis, skewness], dim=-1)
        feat = feat.reshape(x.size(0), -1)
        return feat


class HealthIndicatorNet(nn.Module):
    """Compute scalar health indicator in [0,1].

    Architecture:
        input_dim -> 32 -> 16 -> 1 -> sigmoid
    """

    def __init__(self, sensor_dim: int, tcn_channels: int, hi_input_source: str = "tcn"):
        super().__init__()
        self.hi_input_source = str(hi_input_source).lower()
        if self.hi_input_source not in {"stats", "tcn", "hybrid"}:
            raise ValueError(f"Unsupported hi_input_source: {hi_input_source}")

        self.stats_dim = sensor_dim * 4
        self.temporal_dim = tcn_channels * 3
        if self.hi_input_source == "stats":
            input_dim = self.stats_dim
        elif self.hi_input_source == "tcn":
            input_dim = self.temporal_dim
        else:
            input_dim = self.stats_dim + self.temporal_dim

        self.feature_dim = input_dim
        self.extractor = StatisticalFeatureExtractor()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def _extract_tcn_temporal_feat(self, tcn_seq: torch.Tensor | None, fallback_x: torch.Tensor) -> torch.Tensor:
        if tcn_seq is None:
            b = fallback_x.size(0)
            return torch.zeros((b, self.temporal_dim), device=fallback_x.device, dtype=fallback_x.dtype)
        if tcn_seq.ndim != 3:
            raise ValueError(f"Expected 3D tcn_seq tensor (B,T,C), got {tuple(tcn_seq.shape)}")

        mean_feat = tcn_seq.mean(dim=1)
        std_feat = tcn_seq.std(dim=1, unbiased=False)
        last_feat = tcn_seq[:, -1, :]
        return torch.cat([mean_feat, std_feat, last_feat], dim=-1)

    def forward(self, x: torch.Tensor, tcn_seq: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return HI scalar and raw HI input features."""

        stat_feat = self.extractor(x)
        temporal_feat = self._extract_tcn_temporal_feat(tcn_seq, fallback_x=x)

        if self.hi_input_source == "stats":
            hi_feat = stat_feat
        elif self.hi_input_source == "tcn":
            hi_feat = temporal_feat
        else:
            hi_feat = torch.cat([temporal_feat, stat_feat], dim=-1)

        hi_feat_norm = self.norm(hi_feat)
        hi_logit = self.mlp(hi_feat_norm)
        hi_score = torch.sigmoid(hi_logit)

        # Keep deterministic [0,1] output per window (no batch-wise normalization).
        return hi_score, hi_feat
