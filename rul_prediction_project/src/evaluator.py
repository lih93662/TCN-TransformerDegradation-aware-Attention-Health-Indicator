"""Evaluation utilities for trained RUL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .loss import compute_regression_metrics, phm2012_score


@dataclass
class EvalResult:
    rmse: float
    mae: float
    r2: float
    phm_score: float
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float
    true_mean: float
    true_std: float
    true_min: float
    true_max: float
    mean_error: float
    y_true: np.ndarray
    y_pred: np.ndarray
    hi: np.ndarray
    ids: List[str]


@dataclass
class GroupMetric:
    group: str
    num_samples: int
    rmse: float
    mae: float
    r2: float
    pred_mean: float
    true_mean: float
    mean_error: float


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """Run model inference and compute regression metrics."""

    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    his: List[np.ndarray] = []
    ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            p = out["pred"]
            h = out["hi"]

            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
            his.append(h.cpu().numpy())
            ids.extend(batch["id"])

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    hi = np.concatenate(his, axis=0)

    y_pred_t = torch.from_numpy(y_pred)
    y_true_t = torch.from_numpy(y_true)
    metrics = compute_regression_metrics(y_pred_t, y_true_t)
    score = phm2012_score(y_pred_t, y_true_t)

    return EvalResult(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        phm_score=score,
        pred_mean=metrics["pred_mean"],
        pred_std=metrics["pred_std"],
        pred_min=metrics["pred_min"],
        pred_max=metrics["pred_max"],
        true_mean=metrics["true_mean"],
        true_std=metrics["true_std"],
        true_min=metrics["true_min"],
        true_max=metrics["true_max"],
        mean_error=metrics["mean_error"],
        y_true=y_true.squeeze(-1),
        y_pred=y_pred.squeeze(-1),
        hi=hi.squeeze(-1),
        ids=ids,
    )


def group_predictions_by_id(
    ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hi: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Group arrays by bearing ID for per-bearing plots."""

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for i, bid in enumerate(ids):
        if bid not in grouped:
            grouped[bid] = {"y_true": [], "y_pred": [], "hi": []}
        grouped[bid]["y_true"].append(float(y_true[i]))
        grouped[bid]["y_pred"].append(float(y_pred[i]))
        grouped[bid]["hi"].append(float(hi[i]))

    finalized: Dict[str, Dict[str, np.ndarray]] = {}
    for bid, data in grouped.items():
        finalized[bid] = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
    return finalized


def summarize_group_metrics(grouped: Dict[str, Dict[str, np.ndarray]]) -> List[GroupMetric]:
    """Compute per-group regression summaries for bearings or life stages."""

    rows: List[GroupMetric] = []
    for group, data in grouped.items():
        pred = torch.from_numpy(np.asarray(data["y_pred"], dtype=np.float32))
        true = torch.from_numpy(np.asarray(data["y_true"], dtype=np.float32))
        metrics = compute_regression_metrics(pred, true)
        rows.append(
            GroupMetric(
                group=group,
                num_samples=int(len(pred)),
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                r2=metrics["r2"],
                pred_mean=metrics["pred_mean"],
                true_mean=metrics["true_mean"],
                mean_error=metrics["mean_error"],
            )
        )
    return rows


def group_predictions_by_life_stage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hi: np.ndarray,
    num_bins: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Group samples by target-life stage using quantile bins."""

    if len(y_true) == 0:
        return {}

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(y_true, quantiles)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    names = ["late_life", "mid_life", "early_life"] if num_bins == 3 else [f"stage_{i}" for i in range(num_bins)]

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for i in range(num_bins):
        mask = (y_true >= edges[i]) & (y_true < edges[i + 1])
        if not np.any(mask):
            continue
        name = names[i] if i < len(names) else f"stage_{i}"
        grouped[name] = {
            "y_true": list(y_true[mask]),
            "y_pred": list(y_pred[mask]),
            "hi": list(hi[mask]),
        }

    return {k: {kk: np.asarray(vv, dtype=np.float32) for kk, vv in data.items()} for k, data in grouped.items()}
