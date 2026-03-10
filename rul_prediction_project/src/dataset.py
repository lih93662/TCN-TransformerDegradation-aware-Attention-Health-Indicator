"""Dataset definitions for PHM 2012 bearing RUL prediction.

This module contains:
1. Filesystem discovery utilities for PHM2012 challenge files.
2. In-memory indexing of bearing run-to-failure sequences.
3. Sliding window conversion for sequence-to-one regression.
4. PyTorch ``Dataset`` classes for train/test subsets.

Important preprocessing design note:
PHM2012 mirrors can expose different sensor *column names* between files.
To avoid schema coupling, downstream preprocessing should select channels by
position (first N numeric sensor columns) rather than relying on name overlap.
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
        split: Dataset split name (``Learning_set`` / ``Test_set``).
        frame: Numeric dataframe sorted by cycle.
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
        """Return candidate numeric sensor columns excluding cycle/time.

        The order of columns is preserved from the source file to support
        position-based channel selection in preprocessing.
        """

        cols: List[str] = []
        for c in self.frame.columns:
            if c.lower() in {"cycle", "time"}:
                continue
            if np.issubdtype(self.frame[c].dtype, np.number):
                cols.append(c)
        return cols


def discover_bearing_files(root: Path) -> Dict[str, List[Path]]:
    """Discover CSV/TXT files inside PHM2012 dataset folder."""

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    candidates = list(root.rglob("*.csv")) + list(root.rglob("*.txt"))
    split_to_files: Dict[str, List[Path]] = {}
    for file in candidates:
        parent_parts = [p.lower() for p in file.parts]
        if "learning_set" in parent_parts:
            split = "Learning_set"
        elif "test_set" in parent_parts:
            split = "Test_set"
        else:
            split = "Unknown"
        split_to_files.setdefault(split, []).append(file)

    for key in split_to_files:
        split_to_files[key] = sorted(split_to_files[key])

    return split_to_files


def _read_table(path: Path) -> pd.DataFrame:
    """Read one bearing file with robust delimiter handling."""

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, sep=None, engine="python")

    lowered = {c.lower(): c for c in frame.columns}
    if "cycle" not in lowered:
        frame.insert(0, "cycle", np.arange(1, len(frame) + 1, dtype=np.int32))
    elif lowered.get("cycle") != "cycle":
        frame = frame.rename(columns={lowered["cycle"]: "cycle"})

    numeric_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if "cycle" not in numeric_cols:
        numeric_cols = ["cycle"] + numeric_cols
    frame = frame[numeric_cols].copy()
    frame = frame.sort_values("cycle").reset_index(drop=True)
    return frame


def _bearing_id_from_path(path: Path) -> str:
    stem = path.stem
    for prefix in ["Bearing", "bearing", "acc", "vibration"]:
        stem = stem.replace(prefix, "")
    return path.parent.name + "_" + stem


def load_all_bearings(dataset_root: Path) -> List[BearingRun]:
    """Load every detected bearing run as ``BearingRun``."""

    files = discover_bearing_files(dataset_root)
    runs: List[BearingRun] = []
    for split, split_files in files.items():
        for file in split_files:
            try:
                frame = _read_table(file)
                if len(frame) < 2:
                    continue
                runs.append(
                    BearingRun(
                        bearing_id=_bearing_id_from_path(file),
                        split=split,
                        frame=frame,
                    )
                )
            except Exception:
                continue

    if not runs:
        raise RuntimeError(f"No usable bearing files found under: {dataset_root}")
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
    labels = rul[starts + window_size - 1].astype(np.float32)
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
