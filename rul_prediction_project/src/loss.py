"""Losses and metrics for RUL prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """Root-mean-square error loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


class MAELoss(nn.Module):
    """Mean absolute error loss."""

    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mae(pred, target)


@dataclass
class LossOutput:
    """Loss container for logging."""

    total: torch.Tensor
    rmse: torch.Tensor
    mae: torch.Tensor


class RawRegressionLoss(nn.Module):
    """Raw regression supervision using either MSE or MAE only."""

    def __init__(self, mode: str = "mse"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"mse", "mae"}:
            raise ValueError(f"Unsupported regression loss mode: {mode}")
        self.mode = mode
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.rmse_metric = RMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        total = self.mse(pred, target) if self.mode == "mse" else self.mae(pred, target)
        rmse = self.rmse_metric(pred, target)
        mae = self.mae(pred, target)
        return LossOutput(total=total, rmse=rmse, mae=mae)


def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute scalar RMSE and MAE from tensors."""

    err = pred - target
    rmse = torch.sqrt(torch.mean(err.pow(2))).item()
    mae = torch.mean(torch.abs(err)).item()
    return {"rmse": rmse, "mae": mae}


def phm2012_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PHM challenge score.

    For each sample i with error d_i = pred_i - true_i:
    - if d_i < 0: exp(-d_i / 13) - 1
    - else:       exp(d_i / 10) - 1

    Lower is better.
    """

    d = (pred.view(-1) - target.view(-1)).detach()
    neg = torch.exp(-d / 13.0) - 1.0
    pos = torch.exp(d / 10.0) - 1.0
    score = torch.where(d < 0, neg, pos)
    return float(score.sum().item())
