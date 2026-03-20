"""Visualization utilities for PHM2012 RUL project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_numpy_1d(x: np.ndarray | List[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


def plot_rul_curve(true_rul: np.ndarray | List[float], pred_rul: np.ndarray | List[float], save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)

    y_true = _to_numpy_1d(true_rul)
    y_pred = _to_numpy_1d(pred_rul)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    x = np.arange(n)

    plt.figure(figsize=(11, 5))
    plt.plot(x, y_true, label="True RUL", linewidth=2.2, color="#2563eb")
    plt.plot(x, y_pred, label="Predicted RUL", linewidth=2.0, linestyle="--", color="#dc2626")
    plt.xlabel("Window Index")
    plt.ylabel("RUL")
    plt.title("RUL Prediction Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_hi_curve(hi_values: np.ndarray | List[float], save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)

    hi = _to_numpy_1d(hi_values)
    x = np.arange(len(hi))

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, hi, linewidth=2.2, color="#7c3aed")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Window Index")
    plt.ylabel("Health Indicator")
    plt.title("Health Indicator Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_attention(attention_matrix: np.ndarray, save_path: str | Path, title: str = "Attention Heatmap") -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)

    attn = np.asarray(attention_matrix, dtype=np.float32)
    while attn.ndim > 2:
        attn = attn.mean(axis=0)
    if attn.ndim == 1:
        attn = attn[None, :]
    if attn.ndim != 2:
        raise ValueError(f"Expected 2D attention matrix, got shape {attn.shape}")

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(attn, cmap="mako", cbar=True)
    plt.title(title)
    plt.xlabel("Key Index")
    plt.ylabel("Query Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    low = float(min(y_true.min(initial=0.0), y_pred.min(initial=0.0)))
    high = float(max(y_true.max(initial=1.0), y_pred.max(initial=1.0)))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.6, color="#2563eb")
    plt.plot([low, high], [low, high], linestyle="--", color="#111827", linewidth=1.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    residual = _to_numpy_1d(y_pred) - _to_numpy_1d(y_true)

    plt.figure(figsize=(7, 5))
    sns.histplot(residual, bins=30, kde=True, color="#dc2626")
    plt.axvline(0.0, linestyle="--", color="#111827", linewidth=1.5)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_residual_vs_target(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    y_true = _to_numpy_1d(y_true)
    residual = _to_numpy_1d(y_pred) - y_true

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residual, s=18, alpha=0.6, color="#7c3aed")
    plt.axhline(0.0, linestyle="--", color="#111827", linewidth=1.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residual vs Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)

    plt.figure(figsize=(7, 5))
    sns.kdeplot(y_true, label="Ground Truth", linewidth=2.0, color="#2563eb")
    sns.kdeplot(y_pred, label="Prediction", linewidth=2.0, color="#dc2626")
    plt.xlabel("Normalized RUL")
    plt.ylabel("Density")
    plt.title("Prediction vs Ground-Truth Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_grouped_metrics(rows: List[Dict[str, float]], save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    if not rows:
        return

    labels = [str(r["group"]) for r in rows]
    rmse = np.asarray([float(r["rmse"]) for r in rows], dtype=np.float32)
    mae = np.asarray([float(r["mae"]) for r in rows], dtype=np.float32)
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, rmse, width=width, label="RMSE", color="#2563eb")
    plt.bar(x + width / 2, mae, width=width, label="MAE", color="#dc2626")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_loss(history: List[Dict[str, float]], out_path: Path) -> None:
    _ensure_dir(out_path)
    epochs = np.arange(1, len(history) + 1)
    train_loss = np.asarray([h["train_total"] for h in history], dtype=np.float32)
    val_loss = np.asarray([h["valid_total"] for h in history], dtype=np.float32)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_rul_prediction_curve(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plot_rul_curve(y_true, y_pred, out_path)


def plot_health_indicator_curve(hi: np.ndarray, out_path: Path) -> None:
    plot_hi_curve(hi, out_path)


def plot_single_bearing_prediction(
    bearing_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    _ensure_dir(out_path)
    x = np.arange(len(y_true))

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true, label="True RUL", linewidth=2.1)
    plt.plot(x, y_pred, label="Predicted RUL", linewidth=2.0, linestyle="--")
    plt.title(f"Example Bearing Prediction: {bearing_id}")
    plt.xlabel("Window Index")
    plt.ylabel("RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
