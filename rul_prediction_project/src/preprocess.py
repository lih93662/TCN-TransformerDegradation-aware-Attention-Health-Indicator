"""Preprocessing pipeline for PHM2012 RUL learning.

This module handles the complete conversion from run-level bearing signals to
supervised sliding-window datasets.

Key design goals implemented here:
1. Stable/reproducible preprocessing.
2. Robust handling of inconsistent PHM2012 sensor column names.
3. Controlled dataset size for faster experiments.

Dataset-size controls:
- window_size defaults to 40.
- stride defaults to 10.
- max_rul defaults to 125.
- max_windows_per_bearing defaults to 20000.
- if windows exceed the cap, reproducible random subsampling is applied.
"""

from __future__ import annotations

import hashlib
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
)


@dataclass
class StandardScaler:
    """Simple per-feature standardization helper.

    Attributes:
        mean_: Channel-wise mean with shape ``(sensor_dim,)``.
        std_: Channel-wise std with shape ``(sensor_dim,)``.
    """

    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply feature-wise standardization.

        Args:
            x: Input array with shape ``(num_samples, sensor_dim)``.

        Returns:
            Standardized array with the same shape.
        """

        return (x - self.mean_) / np.clip(self.std_, 1e-8, None)


@dataclass
class PreparedData:
    """Container returned by ``prepare_datasets``.

    Attributes:
        train_dataset: Train split dataset.
        valid_dataset: Validation split dataset.
        test_dataset: Test split dataset.
        scaler: Fitted global scaler from training runs.
        feature_dim: Number of selected sensor channels.
        detected_sensor_columns: Sensor column names selected from first
            training run (for logging/traceability).
    """

    train_dataset: PHM2012RULDataset
    valid_dataset: PHM2012RULDataset
    test_dataset: PHM2012RULDataset
    scaler: StandardScaler
    feature_dim: int
    detected_sensor_columns: List[str]




def split_official_learning_test(
    runs: Sequence[BearingRun],
    valid_ratio: float = 0.33,
    seed: int = 42,
) -> Tuple[List[BearingRun], List[BearingRun], List[BearingRun]]:
    """Split PHM2012 runs by official set first, then random train/valid on Learning_set.

    - Learning_set -> train + validation
    - Test_set -> test only

    Sliding windows from one run remain within a single split because splitting is
    performed at run level before windowing.
    """

    learning_runs = [r for r in runs if r.split == "Learning_set"]
    test_runs = [r for r in runs if r.split == "Test_set"]

    if not learning_runs:
        # Fallback for non-standard mirrors without split labels.
        learning_runs = list(runs)
        test_runs = []

    shuffled = list(learning_runs)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    valid_size = max(1, int(len(shuffled) * valid_ratio))
    valid_runs = shuffled[:valid_size]
    train_runs = shuffled[valid_size:]

    if not train_runs:
        train_runs, valid_runs = valid_runs, train_runs

    return train_runs, valid_runs, test_runs


def compute_linear_rul(cycles: np.ndarray, max_rul: int = 125) -> np.ndarray:
    """Compute clipped linear RUL labels.

    Formula:
        RUL = max_cycle - current_cycle
        RUL = clip(RUL, 0, max_rul)

    Args:
        cycles: Cycle index array with shape ``(num_cycles,)``.
        max_rul: Maximum RUL clipping threshold.

    Returns:
        RUL array of shape ``(num_cycles,)``.
    """

    max_cycle = int(np.max(cycles))
    rul = max_cycle - cycles
    return np.clip(rul, 0, max_rul).astype(np.float32)


def _stable_int_hash(text: str) -> int:
    """Compute a stable positive integer hash for reproducible run sampling.

    Args:
        text: Input string.

    Returns:
        Deterministic integer hash.
    """

    h = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:8], 16)


def _select_sensor_columns(run: BearingRun, sensor_dim: int) -> List[str]:
    """Select the first ``sensor_dim`` numeric sensor columns from one run.

    We use position-based channel selection to avoid dependence on inconsistent
    sensor names across PHM2012 mirrors.

    Args:
        run: Bearing run container.
        sensor_dim: Number of channels to retain.

    Returns:
        List of selected column names.

    Raises:
        ValueError: If the run has fewer numeric columns than ``sensor_dim``.
    """

    candidates = run.sensor_columns
    if len(candidates) < sensor_dim:
        raise ValueError(
            f"Run {run.bearing_id} has {len(candidates)} sensor columns, "
            f"but sensor_dim={sensor_dim} is required."
        )
    return candidates[:sensor_dim]


def fit_scaler(train_runs: Sequence[BearingRun], sensor_dim: int = 2) -> Tuple[StandardScaler, List[str]]:
    """Fit a global channel-wise scaler from training runs only.

    Args:
        train_runs: Training run list.
        sensor_dim: Number of selected channels per run.

    Returns:
        Tuple ``(scaler, detected_sensor_columns)``.

    Algorithm:
        1. Select first ``sensor_dim`` numeric columns per run.
        2. Concatenate all train samples.
        3. Compute mean/std per channel.
    """

    mats: List[np.ndarray] = []
    detected_cols: List[str] = []

    for i, run in enumerate(train_runs):
        cols = _select_sensor_columns(run, sensor_dim)
        if i == 0:
            detected_cols = cols
        mats.append(run.frame[cols].to_numpy(dtype=np.float32))

    if not mats:
        raise RuntimeError("No training runs available to fit scaler.")

    stacked = np.concatenate(mats, axis=0)
    mean_ = np.mean(stacked, axis=0).astype(np.float32)
    std_ = np.std(stacked, axis=0).astype(np.float32)
    return StandardScaler(mean_=mean_, std_=std_), detected_cols


def _cap_windows_per_bearing(
    x: np.ndarray,
    y: np.ndarray,
    bearing_id: str,
    max_windows_per_bearing: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap window count for one bearing via reproducible random sampling.

    Args:
        x: Window tensor ``(num_windows, window_size, sensor_dim)``.
        y: Label array ``(num_windows,)``.
        bearing_id: Bearing identifier for deterministic per-run RNG.
        max_windows_per_bearing: Maximum windows retained for this run.
        seed: Global seed.

    Returns:
        Possibly subsampled ``(x, y)``.

    Notes:
        - If ``num_windows <= cap``, arrays are returned unchanged.
        - Sampling is without replacement.
        - Sorting sampled indices preserves chronological order after sampling,
          improving temporal consistency for qualitative plotting.
    """

    num = x.shape[0]
    if num <= max_windows_per_bearing:
        return x, y

    run_seed = seed + _stable_int_hash(bearing_id)
    rng = np.random.default_rng(run_seed)
    idx = rng.choice(num, size=max_windows_per_bearing, replace=False)
    idx = np.sort(idx)
    return x[idx], y[idx]


def _runs_to_samples(
    runs: Sequence[BearingRun],
    scaler: StandardScaler,
    window_size: int,
    stride: int,
    max_rul: int,
    sensor_dim: int,
    max_windows_per_bearing: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert run list into supervised arrays with sample-size control.

    Args:
        runs: Bearing runs.
        scaler: Global fitted scaler.
        window_size: Sliding window length.
        stride: Sliding stride.
        max_rul: RUL clipping value.
        sensor_dim: Number of selected channels.
        max_windows_per_bearing: Per-run sample cap.
        seed: Global reproducibility seed.

    Returns:
        Tuple ``(X, y, ids)`` where
        - ``X`` has shape ``(N, window_size, sensor_dim)``
        - ``y`` has shape ``(N,)``
        - ``ids`` has length ``N``
    """

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_ids: List[str] = []

    for run in runs:
        cols = _select_sensor_columns(run, sensor_dim=sensor_dim)
        sensors = run.frame[cols].to_numpy(dtype=np.float32)
        sensors = scaler.transform(sensors)

        # Per-run per-sensor z-score normalization before windowing.
        mean = np.mean(sensors, axis=0, keepdims=True)
        std = np.std(sensors, axis=0, keepdims=True) + 1e-6
        sensors = (sensors - mean) / std

        if sensors.shape[1] != sensor_dim:
            raise RuntimeError(
                f"Sensor dimension mismatch for run {run.bearing_id}: "
                f"got {sensors.shape[1]} expected {sensor_dim}."
            )

        channel_std = np.std(sensors, axis=0)
        if np.any(channel_std < 1e-8):
            raise RuntimeError(
                f"Degenerate normalized sensor channel detected for run {run.bearing_id}: {channel_std}."
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

        x, y = _cap_windows_per_bearing(
            x=x,
            y=y,
            bearing_id=run.bearing_id,
            max_windows_per_bearing=max_windows_per_bearing,
            seed=seed,
        )

        all_x.append(x)
        all_y.append(y)
        all_ids.extend([run.bearing_id] * x.shape[0])

    if not all_x:
        return (
            np.empty((0, window_size, sensor_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0), all_ids


def prepare_datasets(
    dataset_root: str | Path = DEFAULT_DATASET_PATH,
    window_size: int = 40,
    stride: int = 10,
    max_rul: int = 125,
    valid_ratio: float = 0.2,
    seed: int = 42,
    sensor_dim: int = 2,
    max_windows_per_bearing: int = 20000,
) -> PreparedData:
    """Prepare train/valid/test datasets from PHM2012 raw files.

    Args:
        dataset_root: PHM2012 root path.
        window_size: Sliding window length (default 40).
        stride: Sliding stride (default 10 for faster training).
        max_rul: RUL clipping value (default 125).
        valid_ratio: Validation split ratio by run.
        seed: Random seed.
        sensor_dim: Number of channels (default 2).
        max_windows_per_bearing: Window cap per run (default 20000).

    Returns:
        ``PreparedData`` object.
    """

    root = Path(dataset_root)
    runs = load_all_bearings(root)

    train_runs, valid_runs, test_runs = split_official_learning_test(
        runs=runs,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    print("Train runs:", [r.bearing_id for r in train_runs])
    print("Valid runs:", [r.bearing_id for r in valid_runs])
    print("Test runs:", [r.bearing_id for r in test_runs])

    scaler, detected_cols = fit_scaler(train_runs, sensor_dim=sensor_dim)

    x_train, y_train, id_train = _runs_to_samples(
        runs=train_runs,
        scaler=scaler,
        window_size=window_size,
        stride=stride,
        max_rul=max_rul,
        sensor_dim=sensor_dim,
        max_windows_per_bearing=max_windows_per_bearing,
        seed=seed,
    )
    x_valid, y_valid, id_valid = _runs_to_samples(
        runs=valid_runs,
        scaler=scaler,
        window_size=window_size,
        stride=stride,
        max_rul=max_rul,
        sensor_dim=sensor_dim,
        max_windows_per_bearing=max_windows_per_bearing,
        seed=seed,
    )
    x_test, y_test, id_test = _runs_to_samples(
        runs=test_runs,
        scaler=scaler,
        window_size=window_size,
        stride=stride,
        max_rul=max_rul,
        sensor_dim=sensor_dim,
        max_windows_per_bearing=max_windows_per_bearing,
        seed=seed,
    )

    if x_train.shape[0] == 0:
        raise RuntimeError("No training windows were generated. Check dataset structure.")

    # Guard against silent shape drift between splits.
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
    """Build compact summary dictionary for logs.

    Args:
        prepared: Prepared data object.

    Returns:
        Dictionary with split counts and detected sensor metadata.
    """

    return {
        "train_samples": len(prepared.train_dataset),
        "valid_samples": len(prepared.valid_dataset),
        "test_samples": len(prepared.test_dataset),
        "num_sensors": prepared.feature_dim,
        "detected_sensor_columns": prepared.detected_sensor_columns,
    }
