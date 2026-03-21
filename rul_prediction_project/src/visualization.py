"""Visualization utilities for PHM2012 RUL project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional plotting dependency
    sns = None


if sns is not None:
    sns.set_theme(style="whitegrid", context="talk")
else:
    plt.style.use("seaborn-v0_8-whitegrid")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_numpy_1d(x: np.ndarray | Sequence[float]) -> np.ndarray:
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
    if attn.ndim != 2:
        raise ValueError(f"Expected 2D attention matrix, got shape {attn.shape}")

    plt.figure(figsize=(6.5, 5.5))
    if sns is not None:
        sns.heatmap(attn, cmap="mako", cbar=True)
    else:
        plt.imshow(attn, aspect="auto", cmap="viridis")
        plt.colorbar()
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
    lo = float(min(y_true.min(initial=0.0), y_pred.min(initial=0.0)))
    hi = float(max(y_true.max(initial=1.0), y_pred.max(initial=1.0)))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=20, alpha=0.6, color="#2563eb", edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="#dc2626", linewidth=1.8, label="Ideal")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title("Prediction vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_residual_histogram(residuals: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    residuals = _to_numpy_1d(residuals)

    plt.figure(figsize=(7, 5))
    if sns is not None:
        sns.histplot(residuals, bins=30, kde=True, color="#7c3aed")
    else:
        plt.hist(residuals, bins=30, color="#7c3aed", alpha=0.8)
    plt.axvline(0.0, color="#111827", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Prediction - Ground Truth)")
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_residual_vs_target(y_true: np.ndarray, residuals: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    y_true = _to_numpy_1d(y_true)
    residuals = _to_numpy_1d(residuals)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residuals, s=20, alpha=0.6, color="#dc2626", edgecolors="none")
    plt.axhline(0.0, color="#111827", linestyle="--", linewidth=1.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Residual")
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
    if sns is not None:
        sns.kdeplot(y_true, label="Ground Truth", linewidth=2.2, color="#2563eb")
        sns.kdeplot(y_pred, label="Prediction", linewidth=2.2, color="#dc2626", linestyle="--")
    else:
        plt.hist(y_true, bins=30, density=True, alpha=0.45, color="#2563eb", label="Ground Truth")
        plt.hist(y_pred, bins=30, density=True, alpha=0.45, color="#dc2626", label="Prediction")
    plt.xlabel("Normalized RUL")
    plt.title("Prediction Distribution vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_attention_weights(weights: np.ndarray, save_path: str | Path, title: str = "Temporal Attention Weights") -> None:
    save_path = Path(save_path)
    _ensure_dir(save_path)
    weights = _to_numpy_1d(weights)

    plt.figure(figsize=(8, 3.5))
    plt.plot(np.arange(len(weights)), weights, linewidth=2.0, color="#0f766e")
    plt.xlabel("Time Index")
    plt.ylabel("Weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_metrics(history: List[Dict[str, float]], out_path: Path) -> None:
    _ensure_dir(out_path)
    epochs = np.arange(1, len(history) + 1)
    train_rmse = np.asarray([h["train_rmse"] for h in history], dtype=np.float32)
    valid_rmse = np.asarray([h["valid_rmse"] for h in history], dtype=np.float32)
    train_mae = np.asarray([h["train_mae"] for h in history], dtype=np.float32)
    valid_mae = np.asarray([h["valid_mae"] for h in history], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_rmse, label="Train RMSE", linewidth=2)
    axes[0].plot(epochs, valid_rmse, label="Valid RMSE", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE by Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_mae, label="Train MAE", linewidth=2)
    axes[1].plot(epochs, valid_mae, label="Valid MAE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE by Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --------------------------
# Backward-compatible helpers
# --------------------------

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
