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
    r2: float
    mean_bias: float
    y_true: np.ndarray
    y_pred: np.ndarray
    hi: np.ndarray
    ids: List[str]
    attention_maps: List[np.ndarray]
    temporal_attention: List[np.ndarray]


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """Run model inference and compute regression/diagnostic metrics."""

    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    his: List[np.ndarray] = []
    ids: List[str] = []
    attention_maps: List[np.ndarray] = []
    temporal_attention: List[np.ndarray] = []

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

            if out.get("attn_map") is not None:
                attention_maps.extend(out["attn_map"].detach().cpu().numpy())
            if out.get("temporal_attn") is not None:
                temporal_attention.extend(out["temporal_attn"].detach().cpu().numpy())

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
        r2=metrics["r2"],
        mean_bias=metrics["mean_bias"],
        y_true=y_true.squeeze(-1),
        y_pred=y_pred.squeeze(-1),
        hi=hi.squeeze(-1),
        ids=ids,
        attention_maps=attention_maps,
        temporal_attention=temporal_attention,
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


def grouped_regression_metrics(ids: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, float | str]]:
    """Compute per-group metrics, using bearing/run id as the group key."""

    grouped_rows: List[Dict[str, float | str]] = []
    grouped_indices: Dict[str, List[int]] = {}
    for idx, bid in enumerate(ids):
        grouped_indices.setdefault(bid, []).append(idx)

    for bid, idxs in sorted(grouped_indices.items()):
        yt = torch.from_numpy(np.asarray(y_true[idxs], dtype=np.float32))
        yp = torch.from_numpy(np.asarray(y_pred[idxs], dtype=np.float32))
        metrics = compute_regression_metrics(yp, yt)
        grouped_rows.append(
            {
                "group": bid,
                "num_samples": len(idxs),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "mean_bias": metrics["mean_bias"],
                "pred_mean": metrics["pred_mean"],
                "pred_std": metrics["pred_std"],
                "true_mean": metrics["true_mean"],
                "true_std": metrics["true_std"],
            }
        )
    return grouped_rows
