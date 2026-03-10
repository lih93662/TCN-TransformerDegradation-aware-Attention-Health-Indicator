"""Visualization utilities for PHM2012 RUL project.

This module centralizes plotting logic for training and evaluation artifacts.
Requested core functions:
- plot_rul_curve(true_rul, pred_rul, save_path)
- plot_hi_curve(hi_values, save_path)
- plot_attention(attention_matrix, save_path)

Implementation details:
- Uses matplotlib + seaborn for publication-friendly style.
- Accepts numpy-like arrays and converts safely.
- Creates output directories automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Use seaborn theme for clearer research plots.
sns.set_theme(style="whitegrid", context="talk")


def _ensure_dir(path: Path) -> None:
    """Ensure parent directory exists before saving figures.

    Args:
        path: Output image path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def _to_numpy_1d(x: np.ndarray | List[float]) -> np.ndarray:
    """Convert array-like values to flattened numpy vector.

    Args:
        x: Input values.

    Returns:
        Float numpy vector of shape ``(N,)``.
    """

    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


def plot_rul_curve(true_rul: np.ndarray | List[float], pred_rul: np.ndarray | List[float], save_path: str | Path) -> None:
    """Plot true RUL versus predicted RUL.

    Args:
        true_rul: Ground-truth RUL sequence.
        pred_rul: Predicted RUL sequence.
        save_path: Path to save image.

    Returns:
        None. Figure is written to disk.
    """

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
    """Plot health indicator trajectory over time.

    Args:
        hi_values: Health-indicator values in chronological order.
        save_path: Path to save image.

    Returns:
        None. Figure is written to disk.
    """

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


def plot_attention(attention_matrix: np.ndarray, save_path: str | Path) -> None:
    """Plot attention heatmap.

    Args:
        attention_matrix: 2D attention matrix ``(T, T)`` or modality matrix.
        save_path: Path to save image.

    Returns:
        None. Figure is written to disk.
    """

    save_path = Path(save_path)
    _ensure_dir(save_path)

    attn = np.asarray(attention_matrix, dtype=np.float32)
    if attn.ndim != 2:
        raise ValueError(f"Expected 2D attention matrix, got shape {attn.shape}")

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(attn, cmap="mako", cbar=True)
    plt.title("Attention Heatmap")
    plt.xlabel("Key Index")
    plt.ylabel("Query Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# --------------------------
# Backward-compatible helpers
# --------------------------

def plot_training_loss(history: List[Dict[str, float]], out_path: Path) -> None:
    """Plot training and validation losses vs epoch.

    This helper remains for compatibility with existing training scripts.
    """

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
    """Compatibility alias for old API name.

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.
        out_path: Output image path.
    """

    plot_rul_curve(y_true, y_pred, out_path)


def plot_health_indicator_curve(hi: np.ndarray, out_path: Path) -> None:
    """Compatibility alias for old API name.

    Args:
        hi: Health indicator values.
        out_path: Output image path.
    """

    plot_hi_curve(hi, out_path)


def plot_single_bearing_prediction(
    bearing_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Plot one bearing's RUL trajectory.

    Args:
        bearing_id: Bearing identifier string.
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        out_path: Output image path.
    """

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
