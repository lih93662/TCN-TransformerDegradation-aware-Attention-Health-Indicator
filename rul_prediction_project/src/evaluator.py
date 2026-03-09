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
    phm_score: float
    y_true: np.ndarray
    y_pred: np.ndarray
    hi: np.ndarray
    ids: List[str]


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """Run model inference and compute RMSE/MAE/PHM score."""

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
        phm_score=score,
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
