"""General utility helpers for the RUL project."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def save_csv_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Save list-of-dicts rows to CSV when rows are available."""

    import csv

    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def slugify(text: str) -> str:
    """Convert free-form experiment names into filesystem-friendly slugs."""

    safe = [c.lower() if c.isalnum() else "_" for c in str(text)]
    slug = "".join(safe)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "experiment"


def resolve_seeds(experiment_cfg: Dict[str, Any]) -> List[int]:
    """Resolve experiment seed list from config.

    Supports either a single ``seed`` or a list under ``seeds``.
    """

    if "seeds" in experiment_cfg and experiment_cfg["seeds"] is not None:
        values = experiment_cfg["seeds"]
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            seeds = [int(v) for v in values]
            if seeds:
                return seeds
    return [int(experiment_cfg.get("seed", 42))]


def experiment_run_name(base_name: str, seed: int, multi_seed: bool) -> str:
    """Build stable run name."""

    slug = slugify(base_name)
    return f"{slug}_seed_{seed}" if multi_seed else slug


def ensure_project_paths(root: str | Path, run_name: str | None = None) -> Dict[str, Path]:
    """Ensure outputs folders exist and return path mapping.

    If ``run_name`` is provided, artifacts are written under
    ``outputs/experiments/<run_name>/`` while preserving the same subfolder layout.
    """

    root = Path(root)
    outputs_root = root / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    outputs = outputs_root if run_name is None else outputs_root / "experiments" / run_name
    checkpoints = outputs / "checkpoints"
    figures = outputs / "figures"
    logs = outputs / "logs"
    results = outputs / "results"

    checkpoints.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "outputs_root": outputs_root,
        "outputs": outputs,
        "checkpoints": checkpoints,
        "figures": figures,
        "logs": logs,
        "results": results,
    }
