"""Preprocessing pipeline for PHM2012 RUL learning.

Responsibilities:
- Build linear/clipped RUL targets per cycle.
- Fit and apply standardization on sensor channels.
- Construct sliding windows for supervised training.
- Provide structured splits for trainer/evaluator modules.
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
        return (x - self.mean_) / np.clip(self.std_, 1e-8, None)


@dataclass
class PreparedData:
    """Container returned by `prepare_datasets`."""

    train_dataset: PHM2012RULDataset
    valid_dataset: PHM2012RULDataset
    test_dataset: PHM2012RULDataset
    scaler: StandardScaler
    feature_dim: int


def compute_linear_rul(cycles: np.ndarray, max_rul: int = 125) -> np.ndarray:
    """Compute clipped linear RUL labels.

    Formula:
        RUL = max_cycle - current_cycle
        RUL = min(RUL, max_rul)

    Args:
        cycles: Observed cycle index values.
        max_rul: Upper clipping threshold.
    """

    max_cycle = int(np.max(cycles))
    rul = max_cycle - cycles
    rul = np.clip(rul, 0, max_rul)
    return rul.astype(np.float32)


def fit_scaler(train_runs: Sequence[BearingRun]) -> StandardScaler:
    """Fit a global scaler using training runs only."""

    mats = []
    for run in train_runs:
        sensor_cols = run.sensor_columns
        mats.append(run.frame[sensor_cols].to_numpy(dtype=np.float32))
    stacked = np.concatenate(mats, axis=0)
    mean_ = np.mean(stacked, axis=0)
    std_ = np.std(stacked, axis=0)
    return StandardScaler(mean_=mean_.astype(np.float32), std_=std_.astype(np.float32))


def _runs_to_samples(
    runs: Sequence[BearingRun],
    scaler: StandardScaler,
    window_size: int,
    stride: int,
    max_rul: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert run list into (X, y, ids) tensors."""

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_ids: List[str] = []

    for run in runs:
        sensors = run.frame[run.sensor_columns].to_numpy(dtype=np.float32)
        sensors = scaler.transform(sensors)
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
            np.empty((0, window_size, 1), dtype=np.float32),
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
) -> PreparedData:
    """Prepare train/valid/test datasets from raw PHM2012 files."""

    root = Path(dataset_root)
    runs = load_all_bearings(root)

    learning_runs = [r for r in runs if r.split == "Learning_set"]
    test_runs = [r for r in runs if r.split == "Test_set"]
    if not learning_runs:
        # Fallback when split detection failed on unconventional archives.
        learning_runs = runs
        test_runs = []

    train_runs, valid_runs = split_train_valid(learning_runs, valid_ratio=valid_ratio, seed=seed)
    scaler = fit_scaler(train_runs)

    x_train, y_train, id_train = _runs_to_samples(
        train_runs, scaler, window_size, stride, max_rul
    )
    x_valid, y_valid, id_valid = _runs_to_samples(
        valid_runs, scaler, window_size, stride, max_rul
    )
    x_test, y_test, id_test = _runs_to_samples(test_runs, scaler, window_size, stride, max_rul)

    if x_train.shape[0] == 0:
        raise RuntimeError("No training windows were generated. Check dataset structure.")

    train_ds = PHM2012RULDataset(x_train, y_train, id_train)
    valid_ds = PHM2012RULDataset(x_valid, y_valid, id_valid) if len(x_valid) else train_ds
    test_ds = PHM2012RULDataset(x_test, y_test, id_test) if len(x_test) else valid_ds

    return PreparedData(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        test_dataset=test_ds,
        scaler=scaler,
        feature_dim=x_train.shape[-1],
    )


def summarize_dataset(prepared: PreparedData) -> Dict[str, int]:
    """Return split statistics for logs and README snippets."""

    return {
        "train_samples": len(prepared.train_dataset),
        "valid_samples": len(prepared.valid_dataset),
        "test_samples": len(prepared.test_dataset),
        "num_sensors": prepared.feature_dim,
    }
