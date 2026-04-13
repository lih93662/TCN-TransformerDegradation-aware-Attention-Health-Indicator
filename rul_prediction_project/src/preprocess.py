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
    target_scale: float
    split_diagnostics: Dict[str, object]


DEFAULT_RUL_BIN_EDGES: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VALID_SPLIT_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "official_default": ("Bearing2_2",),
    "alt_bearing1_1": ("Bearing1_1",),
    "alt_bearing1_2": ("Bearing1_2",),
    "alt_bearing2_1": ("Bearing2_1",),
    "alt_bearing3_1": ("Bearing3_1",),
    "alt_bearing3_2": ("Bearing3_2",),
}




def split_official_learning_test(
    runs: Sequence[BearingRun],
    valid_ratio: float = 0.33,
    seed: int = 42,
    valid_bearing_ids: Sequence[str] | None = None,
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

    if valid_bearing_ids:
        valid_set = {str(v) for v in valid_bearing_ids}
        valid_runs = [r for r in learning_runs if r.bearing_id in valid_set]
        train_runs = [r for r in learning_runs if r.bearing_id not in valid_set]
    else:
        shuffled = list(learning_runs)
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled)

        valid_size = max(1, int(len(shuffled) * valid_ratio))
        valid_runs = shuffled[:valid_size]
        train_runs = shuffled[valid_size:]

    if not train_runs:
        train_runs, valid_runs = valid_runs, train_runs

    return train_runs, valid_runs, test_runs


def _bearing_suffix_id(bearing_id: str) -> str:
    tokens = str(bearing_id).split("_")
    if len(tokens) >= 2:
        return "_".join(tokens[-2:])
    return str(bearing_id)


def _resolve_valid_bearing_ids(
    learning_runs: Sequence[BearingRun],
    valid_bearing_ids: Sequence[str] | None,
) -> List[str]:
    if not valid_bearing_ids:
        return []
    by_full = {str(r.bearing_id): str(r.bearing_id) for r in learning_runs}
    by_suffix = {_bearing_suffix_id(r.bearing_id): str(r.bearing_id) for r in learning_runs}
    resolved: List[str] = []
    for raw_id in valid_bearing_ids:
        candidate = str(raw_id)
        if candidate in by_full:
            resolved.append(by_full[candidate])
            continue
        suffix = _bearing_suffix_id(candidate)
        if suffix in by_suffix:
            resolved.append(by_suffix[suffix])
    return sorted(set(resolved))


def _auto_select_valid_bearings(
    learning_runs: Sequence[BearingRun],
    seed: int,
    count: int,
) -> List[str]:
    if not learning_runs:
        return []
    k = max(1, min(int(count), 2, len(learning_runs)))
    shuffled = list(learning_runs)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    return [str(r.bearing_id) for r in shuffled[:k]]


def compute_linear_rul(cycles: np.ndarray, max_rul: int = 125) -> np.ndarray:
    """Compute linear RUL labels, optionally with clipping.

    Formula:
        RUL = max_cycle - current_cycle
        If ``max_rul > 0``: RUL = clip(RUL, 0, max_rul)

    Args:
        cycles: Cycle index array with shape ``(num_cycles,)``.
        max_rul: Maximum RUL clipping threshold.

    Returns:
        RUL array of shape ``(num_cycles,)``.
    """

    max_cycle = int(np.max(cycles))
    rul = (max_cycle - cycles).astype(np.float32)
    if max_rul is not None and int(max_rul) > 0:
        rul = np.clip(rul, 0, max_rul)
    return rul.astype(np.float32)


def compute_target_scale(train_runs: Sequence[BearingRun], max_rul: int) -> float:
    """Compute a fixed target normalization scale from training runs only.

    If ``max_rul > 0``, the configured cap is used.
    Otherwise, the maximum unclipped training RUL is used so labels preserve the
    full degradation slope without per-run normalization.
    """

    if max_rul is not None and int(max_rul) > 0:
        return float(max_rul)

    maxima: List[float] = []
    for run in train_runs:
        cycles = run.frame["cycle"].to_numpy(dtype=np.int32)
        maxima.append(float(np.max(compute_linear_rul(cycles, max_rul=0))))
    if not maxima:
        raise RuntimeError("No training runs available to compute target scale.")
    return float(max(maxima))


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


def fit_scaler(
    train_runs: Sequence[BearingRun],
    sensor_dim: int = 2,
    mode: str = "standard",
) -> Tuple[StandardScaler, List[str]]:
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
    mode = str(mode).lower()
    if mode == "robust":
        median_ = np.median(stacked, axis=0).astype(np.float32)
        q1 = np.percentile(stacked, 25, axis=0).astype(np.float32)
        q3 = np.percentile(stacked, 75, axis=0).astype(np.float32)
        scale_ = np.clip(q3 - q1, 1e-6, None).astype(np.float32)
        mean_ = median_
        std_ = scale_
    else:
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
    label_scale: float,
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
            label_scale=float(label_scale),
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


def _normalize_bin_edges(bin_edges: Sequence[float] | None) -> np.ndarray:
    edges = np.asarray(bin_edges if bin_edges else DEFAULT_RUL_BIN_EDGES, dtype=np.float32)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("rul_bin_edges must contain at least two values.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"rul_bin_edges must be strictly increasing, got: {edges.tolist()}")
    if edges[0] > 0.0:
        edges = np.concatenate([np.array([0.0], dtype=np.float32), edges], axis=0)
    if np.isinf(edges[-1]):
        return edges.astype(np.float32)
    if edges[-1] >= 1.0:
        edges = edges.copy()
        edges[-1] = np.inf
    else:
        edges = np.concatenate([edges, np.array([np.inf], dtype=np.float32)], axis=0)
    return edges.astype(np.float32)


def _rul_bin_labels(bin_edges: np.ndarray) -> List[str]:
    labels: List[str] = []
    for idx in range(len(bin_edges) - 1):
        left = float(bin_edges[idx])
        right = float(bin_edges[idx + 1])
        if idx == 0:
            labels.append(f"[{left:.1f},{right:.1f}]")
        elif np.isinf(right):
            labels.append(f"({left:.1f},1.0+]")
        else:
            labels.append(f"({left:.1f},{right:.1f}]")
    return labels


def _rul_bin_histogram(y: np.ndarray, bin_edges: np.ndarray) -> Dict[str, int]:
    labels = _rul_bin_labels(bin_edges)
    hist, _ = np.histogram(y, bins=bin_edges) if y.size else (np.zeros(len(labels), dtype=np.int64), bin_edges)
    return {label: int(hist[idx]) for idx, label in enumerate(labels)}


def _resolve_balance_target_count(
    counts: np.ndarray,
    mode: str,
    target: str,
    custom_target: int | None,
) -> int:
    active = counts[counts > 0]
    if active.size == 0:
        return 0
    target_key = target.lower()
    if target_key == "custom":
        if custom_target is None or int(custom_target) <= 0:
            raise ValueError("train_balance_target='custom' requires positive train_balance_custom_count.")
        return int(custom_target)
    if target_key == "min":
        return int(np.min(active))
    if target_key == "median":
        return int(np.median(active))
    if target_key == "max":
        return int(np.max(active))
    if target_key == "auto":
        if mode == "downsample":
            return int(np.min(active))
        if mode == "oversample":
            return int(np.max(active))
        return int(np.median(active))
    raise ValueError(f"Unsupported train_balance_target: {target}")


def _balance_train_samples_by_rul_bins(
    x: np.ndarray,
    y: np.ndarray,
    ids: List[str],
    bin_edges: np.ndarray,
    mode: str,
    target: str,
    custom_target: int | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if x.shape[0] == 0:
        return x, y, ids

    mode = mode.lower()
    if mode not in {"oversample", "downsample", "hybrid"}:
        raise ValueError(f"Unsupported train_balance_mode: {mode}")

    bin_idx = np.digitize(y, bins=bin_edges[1:-1], right=True)
    num_bins = len(bin_edges) - 1
    counts = np.bincount(bin_idx, minlength=num_bins)
    target_count = _resolve_balance_target_count(counts, mode=mode, target=target, custom_target=custom_target)
    if target_count <= 0:
        return x, y, ids

    rng = np.random.default_rng(seed)
    sampled_idx: List[np.ndarray] = []
    for b in range(num_bins):
        idx = np.where(bin_idx == b)[0]
        if idx.size == 0:
            continue
        if mode == "oversample":
            take = max(target_count, idx.size)
            chosen = rng.choice(idx, size=take, replace=take > idx.size)
        elif mode == "downsample":
            take = min(target_count, idx.size)
            chosen = rng.choice(idx, size=take, replace=False)
        else:  # hybrid
            take = target_count
            chosen = rng.choice(idx, size=take, replace=take > idx.size)
        sampled_idx.append(chosen)

    if not sampled_idx:
        return x, y, ids

    merged_idx = np.concatenate(sampled_idx, axis=0)
    rng.shuffle(merged_idx)
    return x[merged_idx], y[merged_idx], [ids[i] for i in merged_idx.tolist()]


def prepare_datasets(
    dataset_root: str | Path = DEFAULT_DATASET_PATH,
    window_size: int = 40,
    stride: int = 10,
    max_rul: int = 125,
    valid_ratio: float = 0.2,
    seed: int = 42,
    sensor_dim: int = 2,
    max_windows_per_bearing: int = 20000,
    scaler_mode: str = "standard",
    valid_bearing_ids: Sequence[str] | None = None,
    valid_split_variant: str | None = None,
    late_stage_threshold: float = 0.2,
    late_stage_oversample_factor: float = 1.0,
    balance_train_rul_bins: bool = False,
    rul_bin_edges: Sequence[float] | None = None,
    train_balance_mode: str = "hybrid",
    train_balance_target: str = "median",
    train_balance_custom_count: int | None = None,
    train_balance_seed: int | None = None,
    auto_valid_bearings_count: int = 1,
    max_auto_valid_tries: int = 16,
    min_valid_active_bins: int = 3,
    valid_skew_ratio_warn: float = 25.0,
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

    bin_edges = _normalize_bin_edges(rul_bin_edges)
    learning_runs = [r for r in runs if r.split == "Learning_set"] or list(runs)
    explicit_valid_ids = list(valid_bearing_ids) if valid_bearing_ids else []
    if valid_split_variant:
        variant_key = str(valid_split_variant).strip()
        if variant_key not in VALID_SPLIT_VARIANTS:
            raise ValueError(
                f"Unknown valid_split_variant='{variant_key}'. Available: {sorted(VALID_SPLIT_VARIANTS.keys())}"
            )
        explicit_valid_ids = list(VALID_SPLIT_VARIANTS[variant_key])
    resolved_explicit_valid = _resolve_valid_bearing_ids(learning_runs, explicit_valid_ids)

    attempts = max(1, int(max_auto_valid_tries))
    best_payload: Dict[str, object] | None = None
    for attempt in range(attempts):
        if resolved_explicit_valid:
            selected_valid_ids = resolved_explicit_valid
        else:
            selected_valid_ids = _auto_select_valid_bearings(
                learning_runs=learning_runs,
                seed=int(seed + attempt),
                count=auto_valid_bearings_count,
            )
        train_runs, valid_runs, test_runs = split_official_learning_test(
            runs=runs,
            valid_ratio=valid_ratio,
            seed=seed + attempt,
            valid_bearing_ids=selected_valid_ids,
        )
        train_ids = {r.bearing_id for r in train_runs}
        valid_ids = {r.bearing_id for r in valid_runs}
        if not valid_ids:
            print("WARNING: validation run set is empty; reselection attempt", attempt + 1)
            continue
        if train_ids.intersection(valid_ids):
            raise RuntimeError(f"Data leakage detected: train/valid overlap {sorted(train_ids.intersection(valid_ids))}")

        scaler, detected_cols = fit_scaler(train_runs, sensor_dim=sensor_dim, mode=scaler_mode)
        target_scale = compute_target_scale(train_runs, max_rul=max_rul)
        x_train, y_train, id_train = _runs_to_samples(
            runs=train_runs,
            scaler=scaler,
            window_size=window_size,
            stride=stride,
            max_rul=max_rul,
            label_scale=target_scale,
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
            label_scale=target_scale,
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
            label_scale=target_scale,
            sensor_dim=sensor_dim,
            max_windows_per_bearing=max_windows_per_bearing,
            seed=seed,
        )
        valid_bins = _rul_bin_histogram(y_valid, bin_edges)
        active_valid_bins = sum(1 for c in valid_bins.values() if c > 0)
        non_zero_counts = [c for c in valid_bins.values() if c > 0]
        skew_ratio = (max(non_zero_counts) / max(min(non_zero_counts), 1)) if non_zero_counts else float("inf")
        best_payload = {
            "train_runs": train_runs,
            "valid_runs": valid_runs,
            "test_runs": test_runs,
            "scaler": scaler,
            "detected_cols": detected_cols,
            "target_scale": target_scale,
            "x_train": x_train,
            "y_train": y_train,
            "id_train": id_train,
            "x_valid": x_valid,
            "y_valid": y_valid,
            "id_valid": id_valid,
            "x_test": x_test,
            "y_test": y_test,
            "id_test": id_test,
            "valid_bins": valid_bins,
            "active_valid_bins": active_valid_bins,
            "valid_skew_ratio": skew_ratio,
        }
        if y_valid.size == 0:
            print("WARNING: validation windows are empty; reselection attempt", attempt + 1)
            if resolved_explicit_valid:
                break
            continue
        if active_valid_bins < int(min_valid_active_bins):
            print(
                f"WARNING: validation RUL coverage is narrow ({active_valid_bins} active bins); reselection attempt {attempt + 1}"
            )
            if resolved_explicit_valid:
                break
            continue
        break

    if best_payload is None:
        raise RuntimeError("Failed to produce non-empty validation split. Check valid bearing settings.")

    train_runs = best_payload["train_runs"]
    valid_runs = best_payload["valid_runs"]
    test_runs = best_payload["test_runs"]
    scaler = best_payload["scaler"]
    detected_cols = best_payload["detected_cols"]
    target_scale = best_payload["target_scale"]
    x_train = best_payload["x_train"]
    y_train = best_payload["y_train"]
    id_train = best_payload["id_train"]
    x_valid = best_payload["x_valid"]
    y_valid = best_payload["y_valid"]
    id_valid = best_payload["id_valid"]
    x_test = best_payload["x_test"]
    y_test = best_payload["y_test"]
    id_test = best_payload["id_test"]
    valid_bins = best_payload["valid_bins"]
    valid_skew_ratio = float(best_payload["valid_skew_ratio"])

    print("Train runs:", [r.bearing_id for r in train_runs])
    print("Valid runs:", [r.bearing_id for r in valid_runs])
    print("Test runs:", [r.bearing_id for r in test_runs])
    if valid_skew_ratio > float(valid_skew_ratio_warn):
        print(f"WARNING: validation RUL-bin skew ratio is high ({valid_skew_ratio:.2f}).")

    pre_balance_train_bins = _rul_bin_histogram(y_train, bin_edges)
    test_bins = _rul_bin_histogram(y_test, bin_edges)

    if x_train.shape[0] == 0:
        raise RuntimeError("No training windows were generated. Check dataset structure.")

    if late_stage_oversample_factor > 1.0:
        late_mask = y_train <= float(late_stage_threshold)
        late_idx = np.where(late_mask)[0]
        if late_idx.size > 0:
            rep = int(round(float(late_stage_oversample_factor) - 1.0))
            rep = max(rep, 0)
            if rep > 0:
                add_idx = np.tile(late_idx, rep)
                x_train = np.concatenate([x_train, x_train[add_idx]], axis=0)
                y_train = np.concatenate([y_train, y_train[add_idx]], axis=0)
                id_train = id_train + [id_train[i] for i in add_idx.tolist()]

    if bool(balance_train_rul_bins):
        x_train, y_train, id_train = _balance_train_samples_by_rul_bins(
            x=x_train,
            y=y_train,
            ids=id_train,
            bin_edges=bin_edges,
            mode=train_balance_mode,
            target=train_balance_target,
            custom_target=train_balance_custom_count,
            seed=int(seed if train_balance_seed is None else train_balance_seed),
        )

    # Guard against silent shape drift between splits.
    if x_valid.size and x_valid.shape[-1] != x_train.shape[-1]:
        raise RuntimeError("Validation sensor dimension mismatch with training set.")
    if x_test.size and x_test.shape[-1] != x_train.shape[-1]:
        raise RuntimeError("Test sensor dimension mismatch with training set.")

    if len(x_valid) == 0:
        raise RuntimeError("Validation dataset is empty after split selection. Aborting to prevent leakage.")

    train_ds = PHM2012RULDataset(x_train, y_train, id_train)
    valid_ds = PHM2012RULDataset(x_valid, y_valid, id_valid)
    test_ds = PHM2012RULDataset(x_test, y_test, id_test) if len(x_test) else valid_ds

    def _split_diag(ids: List[str], y: np.ndarray) -> Dict[str, object]:
        unique, counts = np.unique(np.asarray(ids, dtype=object), return_counts=True) if ids else (np.array([]), np.array([]))
        return {
            "windows_per_bearing": {str(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())},
            "rul_bin_counts": _rul_bin_histogram(y, bin_edges),
        }

    split_diagnostics = {
        "window_size": int(window_size),
        "stride": int(stride),
        "scaler_mode": str(scaler_mode),
        "valid_split_variant": str(valid_split_variant or ""),
        "valid_bearing_ids": [r.bearing_id for r in valid_runs],
        "late_stage_threshold": float(late_stage_threshold),
        "late_stage_oversample_factor": float(late_stage_oversample_factor),
        "balance_train_rul_bins": bool(balance_train_rul_bins),
        "rul_bin_edges": [float(v) for v in bin_edges.tolist()],
        "train_balance_mode": str(train_balance_mode),
        "train_balance_target": str(train_balance_target),
        "train_balance_custom_count": None if train_balance_custom_count is None else int(train_balance_custom_count),
        "train_balance_seed": int(seed if train_balance_seed is None else train_balance_seed),
        "rul_bin_counts_before_balance": {
            "train": pre_balance_train_bins,
            "valid": valid_bins,
            "test": test_bins,
        },
        "train": _split_diag(id_train, y_train),
        "valid": _split_diag(id_valid, y_valid),
        "test": _split_diag(id_test, y_test),
    }
    print("Split diagnostics:", split_diagnostics)

    return PreparedData(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        test_dataset=test_ds,
        scaler=scaler,
        feature_dim=x_train.shape[-1],
        detected_sensor_columns=detected_cols,
        target_scale=target_scale,
        split_diagnostics=split_diagnostics,
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
        "target_scale": prepared.target_scale,
        "split_diagnostics": prepared.split_diagnostics,
    }
