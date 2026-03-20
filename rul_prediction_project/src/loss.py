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
    primary: torch.Tensor
    rmse: torch.Tensor
    mae: torch.Tensor
    mean_error: torch.Tensor
    bias_penalty: torch.Tensor


class RegressionLoss(nn.Module):
    """Configurable regression objective with optional calibration regularization.

    Supported modes:
    - ``mse``
    - ``mae``
    - ``huber``
    - ``mse_mae`` (weighted combination)

    The optional ``bias_weight`` adds a lightweight calibration penalty on the
    squared batch-wise mean error. This is intentionally simple and paper-friendly:
    it reduces systematic global prediction offset without changing model outputs at
    inference time.
    """

    def __init__(
        self,
        mode: str = "mse",
        mse_weight: float = 1.0,
        mae_weight: float = 0.0,
        huber_delta: float = 1.0,
        bias_weight: float = 0.0,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.bias_weight = float(bias_weight)

        supported = {"mse", "mae", "huber", "mse_mae"}
        if self.mode not in supported:
            raise ValueError(f"Unsupported regression loss mode: {mode}")

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=float(huber_delta))
        self.rmse_metric = RMSELoss()

    def _primary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "mse":
            return self.mse(pred, target)
        if self.mode == "mae":
            return self.mae(pred, target)
        if self.mode == "huber":
            return self.huber(pred, target)
        return (self.mse_weight * self.mse(pred, target)) + (self.mae_weight * self.mae(pred, target))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        primary = self._primary_loss(pred, target)
        mean_error = pred.mean() - target.mean()
        bias_penalty = mean_error.pow(2)
        total = primary + (self.bias_weight * bias_penalty)
        rmse = self.rmse_metric(pred, target)
        mae = self.mae(pred, target)
        return LossOutput(
            total=total,
            primary=primary,
            rmse=rmse,
            mae=mae,
            mean_error=mean_error,
            bias_penalty=bias_penalty,
        )


class RawRegressionLoss(RegressionLoss):
    """Backward-compatible alias for the existing trainer code path."""

    def __init__(self, mode: str = "mse"):
        super().__init__(mode=mode)


class CompositeRULLoss(RegressionLoss):
    """Backward-compatible weighted MSE+MAE loss used by benchmark scripts."""

    def __init__(self, rmse_weight: float = 1.0, mae_weight: float = 0.3):
        super().__init__(mode="mse_mae", mse_weight=rmse_weight, mae_weight=mae_weight)


def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute scalar RMSE, MAE, R2, and summary statistics from tensors."""

    pred = pred.view(-1).float()
    target = target.view(-1).float()
    err = pred - target
    rmse = torch.sqrt(torch.mean(err.pow(2))).item()
    mae = torch.mean(torch.abs(err)).item()
    target_mean = torch.mean(target)
    ss_res = torch.sum(err.pow(2))
    ss_tot = torch.sum((target - target_mean).pow(2))
    r2 = 1.0 - float((ss_res / torch.clamp(ss_tot, min=1e-8)).item())
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pred_mean": float(torch.mean(pred).item()),
        "pred_std": float(torch.std(pred, unbiased=False).item()),
        "pred_min": float(torch.min(pred).item()),
        "pred_max": float(torch.max(pred).item()),
        "true_mean": float(torch.mean(target).item()),
        "true_std": float(torch.std(target, unbiased=False).item()),
        "true_min": float(torch.min(target).item()),
        "true_max": float(torch.max(target).item()),
        "mean_error": float(torch.mean(err).item()),
    }


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
