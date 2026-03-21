"""General utility helpers for the RUL project."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""

    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file with optional lightweight inheritance.

    If the YAML contains ``base_config``, it is resolved relative to the current
    config file and recursively merged, with the current file taking precedence.
    """

    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_cfg = data.pop("base_config", None)
    if not base_cfg:
        return data

    base_path = Path(base_cfg)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()

    merged = _deep_merge_dict(load_yaml(base_path), data)
    return merged


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a JSON file with pretty formatting."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv_rows(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Save a sequence of dictionaries as a CSV file."""

    import csv

    rows = list(rows)
    if not rows:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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


def slugify_name(text: str) -> str:
    """Convert experiment name into a filesystem-friendly slug."""

    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in str(text)]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "experiment"


def build_run_name(base_name: str, seed: int, suffix: str | None = None) -> str:
    """Build deterministic experiment run name."""

    parts = [slugify_name(base_name), f"seed{int(seed)}"]
    if suffix:
        parts.append(slugify_name(suffix))
    return "__".join(parts)


def timestamp_string() -> str:
    """Return a compact UTC timestamp string."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_project_paths(root: str | Path, run_name: str | None = None) -> Dict[str, Path]:
    """Ensure outputs folders exist and return path mapping.

    When ``run_name`` is provided, run-specific directories are created under
    ``outputs/runs/<run_name>/`` while preserving the project-level outputs
    folders for backward compatibility.
    """

    root = Path(root)
    outputs = root / "outputs"
    checkpoints = outputs / "checkpoints"
    figures = outputs / "figures"
    logs = outputs / "logs"
    results = outputs / "results"
    runs = outputs / "runs"

    for path in (checkpoints, figures, logs, results, runs):
        path.mkdir(parents=True, exist_ok=True)

    path_map: Dict[str, Path] = {
        "root": root,
        "outputs": outputs,
        "checkpoints": checkpoints,
        "figures": figures,
        "logs": logs,
        "results": results,
        "runs": runs,
    }

    if run_name:
        run_dir = runs / run_name
        run_logs = run_dir / "logs"
        run_figures = run_dir / "figures"
        run_checkpoints = run_dir / "checkpoints"
        run_results = run_dir / "results"
        for path in (run_dir, run_logs, run_figures, run_checkpoints, run_results):
            path.mkdir(parents=True, exist_ok=True)
        path_map.update(
            {
                "run_dir": run_dir,
                "run_logs": run_logs,
                "run_figures": run_figures,
                "run_checkpoints": run_checkpoints,
                "run_results": run_results,
            }
        )

    return path_map
