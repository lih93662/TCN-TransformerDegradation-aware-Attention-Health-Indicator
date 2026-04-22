"""Run the minimal 4-setting ablation suite and export paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import build_run_name, ensure_project_paths, load_yaml, save_csv_rows


ABLATONS = [
    ("A", "backbone only", ROOT / "configs" / "minimal_compare" / "backbone_baseline.yaml"),
    ("B", "backbone + attention", ROOT / "configs" / "minimal_compare" / "plus_attention.yaml"),
    ("C", "backbone + HI auxiliary", ROOT / "configs" / "minimal_compare" / "plus_hi.yaml"),
    ("D", "full model", ROOT / "configs" / "minimal_compare" / "full_model.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal ablation A/B/C/D")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoints and only run evaluation.")
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _metrics_path(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    exp = cfg.get("experiment", {})
    run_name = build_run_name(
        base_name=str(exp.get("name", "rul_experiment")),
        seed=int(exp.get("seed", 42)),
        suffix=str(exp.get("tag", "")) or None,
    )
    return ROOT / "outputs" / "runs" / run_name / "results" / "evaluation_test" / "tables" / "metrics.json"


def _checkpoint_path(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    exp = cfg.get("experiment", {})
    run_name = build_run_name(
        base_name=str(exp.get("name", "rul_experiment")),
        seed=int(exp.get("seed", 42)),
        suffix=str(exp.get("tag", "")) or None,
    )
    return ROOT / "outputs" / "runs" / run_name / "checkpoints" / "best_model.pth"


def _read_metrics_json(path: Path) -> Dict[str, float]:
    import json

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload["raw"]
    return {
        "RMSE": float(raw["rmse"]),
        "MAE": float(raw["mae"]),
        "R2": float(raw["r2"]),
        "PHM": float(raw["phm_score"]),
    }


def main() -> None:
    args = parse_args()
    rows = []

    for label, name, cfg_path in ABLATONS:
        if not args.skip_train:
            _run([sys.executable, "scripts/train.py", "--config", str(cfg_path)])
        ckpt = _checkpoint_path(cfg_path)
        _run([sys.executable, "scripts/evaluate.py", "--config", str(cfg_path), "--checkpoint", str(ckpt), "--split", "test"])
        metrics = _read_metrics_json(_metrics_path(cfg_path))
        rows.append({"ablation": label, "model": name, **metrics})

    project_paths = ensure_project_paths(ROOT)
    csv_path = project_paths["results"] / "ablation_results.csv"
    md_path = project_paths["results"] / "ablation_results.md"
    save_csv_rows(csv_path, rows)

    lines = [
        "| ablation | model | RMSE | MAE | R2 | PHM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        [
            f"| {r['ablation']} | {r['model']} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['R2']:.6f} | {r['PHM']:.6f} |"
            for r in rows
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
