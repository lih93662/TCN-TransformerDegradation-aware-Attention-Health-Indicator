"""Run the four core ablation experiments sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable or "python"

EXPERIMENTS: List[Dict[str, str]] = [
    {
        "name": "A_backbone_only",
        "config": "configs/config_backbone.yaml",
        "output_dir": "outputs/experiments/ablation_backbone_only",
        "checkpoint": "outputs/experiments/ablation_backbone_only/checkpoints/best_model.pth",
    },
    {
        "name": "B_backbone_attention",
        "config": "configs/config_attention.yaml",
        "output_dir": "outputs/experiments/ablation_backbone_attention",
        "checkpoint": "outputs/experiments/ablation_backbone_attention/checkpoints/best_model.pth",
    },
    {
        "name": "C_backbone_improved_loss",
        "config": "configs/config_loss.yaml",
        "output_dir": "outputs/experiments/ablation_backbone_improved_loss",
        "checkpoint": "outputs/experiments/ablation_backbone_improved_loss/checkpoints/best_model.pth",
    },
    {
        "name": "D_full",
        "config": "configs/config_full.yaml",
        "output_dir": "outputs/experiments/ablation_full",
        "checkpoint": "outputs/experiments/ablation_full/checkpoints/best_model.pth",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four ablation experiments sequentially")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    summary_rows: List[Dict[str, object]] = []
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        cmd = [PYTHON, "scripts/train.py", "--config", exp["config"]]
        printable = " ".join(cmd)
        print(f"\n=== Running {exp['name']} ===")
        print(printable)

        row: Dict[str, object] = {
            "experiment": exp["name"],
            "config": exp["config"],
            "command": printable,
            "output_dir": exp["output_dir"],
            "checkpoint": exp["checkpoint"],
        }

        if args.dry_run:
            row["status"] = "dry_run"
            summary_rows.append(row)
            continue

        proc = subprocess.run(cmd, cwd=ROOT)
        row["returncode"] = int(proc.returncode)
        if proc.returncode != 0:
            row["status"] = "failed"
            summary_rows.append(row)
            with open(results_dir / "ablation_run_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_rows, f, indent=2)
            print(f"Experiment failed: {exp['name']}")
            return proc.returncode

        row["status"] = "completed"
        metrics = load_json_if_exists(ROOT / exp["output_dir"] / "logs" / "test_metrics.json")
        history = load_json_if_exists(ROOT / exp["output_dir"] / "logs" / "history.json")
        if metrics:
            row.update(metrics)
        if history.get("history"):
            best_epoch = min(
                history["history"],
                key=lambda item: float(item.get("valid_rmse", float("inf"))),
            ).get("epoch")
            row["best_epoch"] = int(best_epoch)
        summary_rows.append(row)

    with open(results_dir / "ablation_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print("\nAll ablation experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
