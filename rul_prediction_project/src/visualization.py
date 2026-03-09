"""Visualization functions for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_loss(history: List[Dict[str, float]], out_path: Path) -> None:
    """Plot training and validation losses vs epoch."""

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


def plot_rul_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Plot true vs predicted RUL over sample index."""

    _ensure_dir(out_path)
    x = np.arange(len(y_true))

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true, label="True RUL", linewidth=2)
    plt.plot(x, y_pred, label="Predicted RUL", linewidth=1.8, alpha=0.85)
    plt.xlabel("Time Cycle / Window Index")
    plt.ylabel("RUL")
    plt.title("Predicted vs True RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_health_indicator_curve(hi: np.ndarray, out_path: Path) -> None:
    """Plot HI progression over time."""

    _ensure_dir(out_path)
    x = np.arange(len(hi))

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, hi, color="purple", linewidth=2)
    plt.xlabel("Time Cycle / Window Index")
    plt.ylabel("Health Indicator")
    plt.ylim(0.0, 1.0)
    plt.title("Health Indicator Degradation Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_single_bearing_prediction(
    bearing_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Plot one example bearing prediction."""

    _ensure_dir(out_path)
    x = np.arange(len(y_true))
    plt.figure(figsize=(9, 5))
    plt.plot(x, y_true, label="True RUL", linewidth=2)
    plt.plot(x, y_pred, label="Predicted RUL", linewidth=2, linestyle="--")
    plt.xlabel("Window Index")
    plt.ylabel("RUL")
    plt.title(f"Example Prediction: {bearing_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
