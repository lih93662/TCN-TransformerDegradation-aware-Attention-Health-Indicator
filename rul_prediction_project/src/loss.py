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
    mse: torch.Tensor
    mean_bias: torch.Tensor
    bias_penalty: torch.Tensor


class RawRegressionLoss(nn.Module):
    """Configurable regression supervision for ablation-friendly RUL training."""

    def __init__(
        self,
        mode: str = "mse",
        huber_delta: float = 0.1,
        mse_weight: float = 1.0,
        mae_weight: float = 0.0,
        bias_regularization_weight: float = 0.0,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"mse", "mae", "huber", "mse_mae"}:
            raise ValueError(f"Unsupported regression loss mode: {mode}")
        self.mode = mode
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.bias_regularization_weight = float(bias_regularization_weight)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=float(huber_delta))
        self.rmse_metric = RMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        mse = self.mse(pred, target)
        mae = self.mae(pred, target)
        rmse = self.rmse_metric(pred, target)
        mean_bias = torch.mean(pred - target)
        bias_penalty = mean_bias.pow(2)

        if self.mode == "mse":
            total = mse
        elif self.mode == "mae":
            total = mae
        elif self.mode == "huber":
            total = self.huber(pred, target)
        else:
            total = self.mse_weight * mse + self.mae_weight * mae

        if self.bias_regularization_weight > 0:
            total = total + self.bias_regularization_weight * bias_penalty

        return LossOutput(
            total=total,
            rmse=rmse,
            mae=mae,
            mse=mse,
            mean_bias=mean_bias,
            bias_penalty=bias_penalty,
        )


def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute scalar regression metrics from tensors."""

    pred = pred.view(-1).float()
    target = target.view(-1).float()
    err = pred - target

    mse = torch.mean(err.pow(2)).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(err)).item()
    mean_bias = torch.mean(err).item()

    target_mean = torch.mean(target)
    ss_res = torch.sum(err.pow(2))
    ss_tot = torch.sum((target - target_mean).pow(2))
    r2 = 1.0 - float(ss_res / torch.clamp(ss_tot, min=1e-8))

    pred_std = float(torch.std(pred, unbiased=False).item())
    true_std = float(torch.std(target, unbiased=False).item())

    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "mean_bias": mean_bias,
        "pred_mean": float(torch.mean(pred).item()),
        "pred_std": pred_std,
        "pred_min": float(torch.min(pred).item()),
        "pred_max": float(torch.max(pred).item()),
        "true_mean": float(torch.mean(target).item()),
        "true_std": true_std,
        "true_min": float(torch.min(target).item()),
        "true_max": float(torch.max(target).item()),
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
