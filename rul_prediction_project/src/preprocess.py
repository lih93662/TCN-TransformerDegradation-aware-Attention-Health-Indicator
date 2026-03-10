"""Preprocessing pipeline for PHM2012 RUL learning.

Responsibilities:
- Build linear/clipped RUL targets per cycle.
- Fit and apply standardization on sensor channels.
- Construct sliding windows for supervised training.
- Provide structured splits for trainer/evaluator modules.

Design for inconsistent PHM2012 sensor headers:
- We intentionally do NOT align channels by column name intersection.
- Instead, each run uses the first ``sensor_dim`` numeric sensor columns in its
  native column order.
- Default ``sensor_dim=2`` targets common two-channel vibration data.
- This strategy avoids empty-column intersections and scaler broadcasting errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dataset import (
    DEFAULT_DATASET_PATH,
    BearingRun,
    PHM2012RULDataset,
    build_sliding_windows,
    load_all_bearings,
    split_train_valid,
)


@dataclass
class StandardScaler:
    """Simple per-feature standardization helper."""

    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply feature-wise standardization.

        Args:
            x: Array of shape ``(n_samples, sensor_dim)``.

        Returns:
            Standardized array with same shape.
        """

        return (x - self.mean_) / np.clip(self.std_, 1e-8, None)


@dataclass
class PreparedData:
    """Container returned by ``prepare_datasets``."""

    train_dataset: PHM2012RULDataset
    valid_dataset: PHM2012RULDataset
    test_dataset: PHM2012RULDataset
    scaler: StandardScaler
    feature_dim: int
    detected_sensor_columns: List[str]


def compute_linear_rul(cycles: np.ndarray, max_rul: int = 125) -> np.ndarray:
    """Compute clipped linear RUL labels.

    Formula:
        RUL = max_cycle - current_cycle
        RUL = min(RUL, max_rul)
    """

    max_cycle = int(np.max(cycles))
    rul = max_cycle - cycles
    rul = np.clip(rul, 0, max_rul)
    return rul.astype(np.float32)


def _select_sensor_columns(run: BearingRun, sensor_dim: int) -> List[str]:
    """Select the first ``sensor_dim`` numeric sensor columns from one run.

    Args:
        run: Bearing run object.
        sensor_dim: Required number of channels.

    Returns:
        Selected column names in deterministic source order.

    Raises:
        ValueError: If run does not have enough numeric sensor columns.
    """

    candidates = run.sensor_columns
    if len(candidates) < sensor_dim:
        raise ValueError(
            f"Run {run.bearing_id} has only {len(candidates)} sensor columns, "
            f"but sensor_dim={sensor_dim} is required."
        )
    return candidates[:sensor_dim]


def fit_scaler(train_runs: Sequence[BearingRun], sensor_dim: int = 2) -> Tuple[StandardScaler, List[str]]:
    """Fit a global scaler using training runs only.

    Args:
        train_runs: Training bearing runs.
        sensor_dim: Number of channels to extract per run.

    Returns:
        Tuple ``(scaler, reference_sensor_columns)``.

    Algorithm explanation:
    - For each run, select first ``sensor_dim`` numeric sensor columns.
    - Stack all selected channels from all training runs.
    - Compute global mean/std for stable train/test normalization.
    """

    mats: List[np.ndarray] = []
    ref_cols: List[str] = []

    for i, run in enumerate(train_runs):
        cols = _select_sensor_columns(run, sensor_dim=sensor_dim)
        if i == 0:
            ref_cols = cols
        mats.append(run.frame[cols].to_numpy(dtype=np.float32))

    if not mats:
        raise RuntimeError("No runs available to fit scaler.")

    stacked = np.concatenate(mats, axis=0)
    mean_ = np.mean(stacked, axis=0)
    std_ = np.std(stacked, axis=0)
    return StandardScaler(mean_=mean_.astype(np.float32), std_=std_.astype(np.float32)), ref_cols


def _runs_to_samples(
    runs: Sequence[BearingRun],
    scaler: StandardScaler,
    window_size: int,
    stride: int,
    max_rul: int,
    sensor_dim: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert run list into ``(X, y, ids)`` arrays with consistent sensor dimension."""

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_ids: List[str] = []

    for run in runs:
        cols = _select_sensor_columns(run, sensor_dim=sensor_dim)
        sensors = run.frame[cols].to_numpy(dtype=np.float32)
        sensors = scaler.transform(sensors)

        if sensors.shape[1] != sensor_dim:
            raise RuntimeError(
                f"Sensor dimension mismatch for run {run.bearing_id}: "
                f"got {sensors.shape[1]} expected {sensor_dim}."
            )

        cycles = run.frame["cycle"].to_numpy(dtype=np.int32)
        rul = compute_linear_rul(cycles, max_rul=max_rul)

        x, y = build_sliding_windows(
            series=sensors,
            rul=rul,
            window_size=window_size,
            stride=stride,
        )
        if x.shape[0] == 0:
            continue
        all_x.append(x)
        all_y.append(y)
        all_ids.extend([run.bearing_id] * x.shape[0])

    if not all_x:
        return (
            np.empty((0, window_size, sensor_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    x_cat = np.concatenate(all_x, axis=0)
    y_cat = np.concatenate(all_y, axis=0)
    return x_cat, y_cat, all_ids


def prepare_datasets(
    dataset_root: str | Path = DEFAULT_DATASET_PATH,
    window_size: int = 40,
    stride: int = 1,
    max_rul: int = 125,
    valid_ratio: float = 0.2,
    seed: int = 42,
    sensor_dim: int = 2,
) -> PreparedData:
    """Prepare train/valid/test datasets from raw PHM2012 files.

    Args:
        dataset_root: Root path to PHM2012 files.
        window_size: Sliding window length.
        stride: Sliding window stride.
        max_rul: RUL clipping value.
        valid_ratio: Validation split ratio by run.
        seed: Random seed for deterministic splitting.
        sensor_dim: Number of sensor channels per run (default=2).

    Returns:
        ``PreparedData`` with consistent train/valid/test feature dimensions.
    """

    root = Path(dataset_root)
    runs = load_all_bearings(root)

    learning_runs = [r for r in runs if r.split == "Learning_set"]
    test_runs = [r for r in runs if r.split == "Test_set"]
    if not learning_runs:
        learning_runs = runs
        test_runs = []

    train_runs, valid_runs = split_train_valid(learning_runs, valid_ratio=valid_ratio, seed=seed)

    scaler, detected_cols = fit_scaler(train_runs, sensor_dim=sensor_dim)

    x_train, y_train, id_train = _runs_to_samples(
        train_runs, scaler, window_size, stride, max_rul, sensor_dim=sensor_dim
    )
    x_valid, y_valid, id_valid = _runs_to_samples(
        valid_runs, scaler, window_size, stride, max_rul, sensor_dim=sensor_dim
    )
    x_test, y_test, id_test = _runs_to_samples(
        test_runs, scaler, window_size, stride, max_rul, sensor_dim=sensor_dim
    )

    if x_train.shape[0] == 0:
        raise RuntimeError("No training windows were generated. Check dataset structure.")

    # Defensive check to ensure consistent train/test channel dimensions.
    if x_valid.size and x_valid.shape[-1] != x_train.shape[-1]:
        raise RuntimeError("Validation sensor dimension mismatch with training set.")
    if x_test.size and x_test.shape[-1] != x_train.shape[-1]:
        raise RuntimeError("Test sensor dimension mismatch with training set.")

    train_ds = PHM2012RULDataset(x_train, y_train, id_train)
    valid_ds = PHM2012RULDataset(x_valid, y_valid, id_valid) if len(x_valid) else train_ds
    test_ds = PHM2012RULDataset(x_test, y_test, id_test) if len(x_test) else valid_ds

    return PreparedData(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        test_dataset=test_ds,
        scaler=scaler,
        feature_dim=x_train.shape[-1],
        detected_sensor_columns=detected_cols,
    )


def summarize_dataset(prepared: PreparedData) -> Dict[str, object]:
    """Return split statistics and detected channels for logging.

    The ``detected_sensor_columns`` field is included so caller logs can clearly
    print which sensor columns were selected during preprocessing.
    """

    return {
        "train_samples": len(prepared.train_dataset),
        "valid_samples": len(prepared.valid_dataset),
        "test_samples": len(prepared.test_dataset),
        "num_sensors": prepared.feature_dim,
        "detected_sensor_columns": prepared.detected_sensor_columns,
    }
