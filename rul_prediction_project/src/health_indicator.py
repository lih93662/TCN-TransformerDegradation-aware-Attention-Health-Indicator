"""Health Indicator (HI) module.

The HI is derived from statistical descriptors over each sliding window and then
mapped into [0, 1] by an MLP.
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

    def __init__(self, sensor_dim: int):
        super().__init__()
        input_dim = sensor_dim * 4
        self.extractor = StatisticalFeatureExtractor()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return HI scalar and raw statistical feature embedding."""

        stat_feat = self.extractor(x)
        hi_logit = self.mlp(stat_feat)
        hi_score = self.out_act(hi_logit)
        return hi_score, stat_feat
