"""General utility helpers for the RUL project."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a JSON file with pretty formatting."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def configure_logging(log_file: str | Path) -> logging.Logger:
    """Configure console + file logger."""

    logger = logging.getLogger("rul_project")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    lf = Path(log_file)
    lf.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(lf, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_device() -> torch.device:
    """Return CUDA device when available."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_project_paths(root: str | Path) -> Dict[str, Path]:
    """Ensure outputs folders exist and return path mapping."""

    root = Path(root)
    outputs = root / "outputs"
    checkpoints = outputs / "checkpoints"
    figures = outputs / "figures"
    logs = outputs / "logs"

    checkpoints.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "outputs": outputs,
        "checkpoints": checkpoints,
        "figures": figures,
        "logs": logs,
    }
