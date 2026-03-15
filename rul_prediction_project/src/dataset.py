"""Dataset definitions for PHM 2012 bearing RUL prediction.

PHM2012 storage layout note (critical):
- A *bearing run* is represented by a directory containing many CSV/TXT
  vibration segments.
- Each segment file is one slice of the same run, not an independent run.
- Therefore, files inside one bearing directory must be sorted and concatenated
  into a single continuous time-series.

This module implements that behavior to avoid exploding sample counts from
incorrect file-level run interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_DATASET_PATH = (
    r"C:\peng\RULdata\ieee-phm-2012-data-challenge-dataset-master"
    r"\phm-ieee-2012-data-challenge-dataset-master"
)


@dataclass
class BearingRun:
    """Container for one run-to-failure bearing sequence.

    Attributes:
        bearing_id: Canonical identifier for the run.
        split: Dataset split name (``Learning_set`` / ``Test_set`` / ``Unknown``).
        frame: Numeric dataframe sorted by continuous cycle.
    """

    bearing_id: str
    split: str
    frame: pd.DataFrame

    @property
    def num_cycles(self) -> int:
        """Return number of observed cycles."""
        return int(len(self.frame))

    @property
    def sensor_columns(self) -> List[str]:
        """Return candidate numeric sensor columns excluding cycle/time."""

        cols: List[str] = []
        for c in self.frame.columns:
            if c.lower() in {"cycle", "time"}:
                continue
            if np.issubdtype(self.frame[c].dtype, np.number):
                cols.append(c)
        return cols


def _detect_split_from_path(path: Path) -> str:
    """Infer split label from path parts."""

    parts = [p.lower() for p in path.parts]
    if "learning_set" in parts:
        return "Learning_set"
    if "test_set" in parts:
        return "Test_set"
    return "Unknown"


def discover_bearing_run_dirs(root: Path) -> Dict[str, List[Path]]:
    """Discover bearing run directories (not individual files).

    Returns:
        Mapping from split name to sorted list of run directories.

    Algorithm:
    - Scan all CSV/TXT files.
    - Use each file's parent directory as a run directory.
    - Group unique directories by split.
    """

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    files = list(root.rglob("*.csv")) + list(root.rglob("*.txt"))
    split_to_dirs: Dict[str, set[Path]] = {}

    for f in files:
        run_dir = f.parent
        split = _detect_split_from_path(f)
        split_to_dirs.setdefault(split, set()).add(run_dir)

    out: Dict[str, List[Path]] = {}
    for split, dirs in split_to_dirs.items():
        out[split] = sorted(dirs)
    return out


def _read_segment_file(path: Path) -> pd.DataFrame:
    """Read one vibration segment file into numeric dataframe.

    The returned frame may still contain varying column names between files;
    alignment is handled during run-level concatenation by position.
    """

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, sep=None, engine="python")

    # Keep only numeric columns to avoid accidental text/meta fields.
    num_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    frame = frame[num_cols].copy()

    # Drop legacy cycle/time columns if present in segment files. The run-level
    # continuous cycle index is rebuilt after concatenation.
    drop_cols = [c for c in frame.columns if str(c).lower() in {"cycle", "time"}]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)

    frame = frame.reset_index(drop=True)
    return frame


def _load_run_from_directory(run_dir: Path) -> pd.DataFrame:
    """Load all segment files under one run directory and concatenate.

    Args:
        run_dir: Path to one bearing run directory.

    Returns:
        Concatenated dataframe with standardized sensor names and continuous
        ``cycle`` index starting at 1.
    """

    seg_files = sorted([*run_dir.glob("*.csv"), *run_dir.glob("*.txt")], key=lambda p: p.name)
    if not seg_files:
        raise RuntimeError(f"No segment files in run directory: {run_dir}")

    seg_frames: List[pd.DataFrame] = []
    for f in seg_files:
        seg = _read_segment_file(f)
        if len(seg) == 0 or seg.shape[1] == 0:
            continue
        seg_frames.append(seg)

    if not seg_frames:
        raise RuntimeError(f"No usable numeric segment data in: {run_dir}")

    # PHM mirrors can have slight segment schema drift; align by position using
    # the minimum shared channel count across segment files.
    shared_dim = min(seg.shape[1] for seg in seg_frames)
    if shared_dim < 1:
        raise RuntimeError(f"No shared sensor channels found in: {run_dir}")

    normalized_segments: List[pd.DataFrame] = []
    sensor_names = [f"sensor_{i}" for i in range(shared_dim)]
    for seg in seg_frames:
        part = seg.iloc[:, :shared_dim].copy()
        part.columns = sensor_names
        normalized_segments.append(part)

    run_frame = pd.concat(normalized_segments, axis=0, ignore_index=True)
    run_frame.insert(0, "cycle", np.arange(1, len(run_frame) + 1, dtype=np.int32))
    return run_frame


def _bearing_id_from_dir(run_dir: Path) -> str:
    """Build deterministic run id from directory path."""

    parent = run_dir.parent.name
    return f"{parent}_{run_dir.name}"


def load_all_bearings(dataset_root: Path) -> List[BearingRun]:
    """Load PHM2012 runs by bearing directory.

    Important:
        This function creates exactly one ``BearingRun`` per run directory,
        preventing the massive sample inflation caused by file-level runs.
    """

    split_dirs = discover_bearing_run_dirs(dataset_root)
    runs: List[BearingRun] = []

    for split, dirs in split_dirs.items():
        for run_dir in dirs:
            try:
                frame = _load_run_from_directory(run_dir)
                if len(frame) < 2:
                    continue
                runs.append(
                    BearingRun(
                        bearing_id=_bearing_id_from_dir(run_dir),
                        split=split,
                        frame=frame,
                    )
                )
            except Exception:
                continue

    if not runs:
        raise RuntimeError(f"No usable bearing runs found under: {dataset_root}")
    return runs


def build_sliding_windows(
    series: np.ndarray,
    rul: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one bearing sequence into overlapping windows."""

    n_cycles = int(series.shape[0])
    if n_cycles < window_size:
        return np.empty((0, window_size, series.shape[1]), dtype=np.float32), np.empty(
            (0,), dtype=np.float32
        )

    starts = np.arange(0, n_cycles - window_size + 1, stride, dtype=np.int32)
    windows = np.stack([series[s : s + window_size] for s in starts], axis=0).astype(np.float32)

    rul_norm = rul.astype(np.float32)
    max_rul = float(np.max(rul_norm)) if rul_norm.size else 1.0
    if max_rul > 0:
        rul_norm = rul_norm / max_rul

    labels = rul_norm[starts + window_size - 1].astype(np.float32)
    return windows, labels


class PHM2012RULDataset(Dataset):
    """PyTorch dataset for PHM2012 bearing RUL prediction."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, ids: Sequence[str]):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float().unsqueeze(-1)
        self.ids = list(ids)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        return {
            "x": self.features[index],
            "y": self.labels[index],
            "id": self.ids[index],
        }


def split_train_valid(
    bearing_runs: Sequence[BearingRun], valid_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[BearingRun], List[BearingRun]]:
    """Split by run identifier to avoid sequence leakage."""

    rng = np.random.default_rng(seed)
    indices = np.arange(len(bearing_runs))
    rng.shuffle(indices)
    cut = max(1, int(len(indices) * (1.0 - valid_ratio)))
    train_idx, valid_idx = indices[:cut], indices[cut:]
    train = [bearing_runs[i] for i in train_idx]
    valid = [bearing_runs[i] for i in valid_idx]
    return train, valid
