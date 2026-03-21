"""Minimal sequential launcher for the 4 key ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))





def _load_project_utils():
    from src.utils import build_run_name, ensure_project_paths, load_yaml, save_csv_rows

    return build_run_name, ensure_project_paths, load_yaml, save_csv_rows


EXPERIMENTS = [
    ("A_backbone", Path("configs/config_backbone.yaml")),
    ("B_attention", Path("configs/config_attention.yaml")),
    ("C_loss", Path("configs/config_loss.yaml")),
    ("D_full", Path("configs/config_full.yaml")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4 ablation experiments sequentially")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands")
    return parser.parse_args()


def run_dir_from_config(config_path: Path) -> Path:
    build_run_name, ensure_project_paths, load_yaml, _ = _load_project_utils()
    cfg = load_yaml(config_path)
    run_name = build_run_name(
        base_name=str(cfg["experiment"].get("name", "rul_experiment")),
        seed=int(cfg["experiment"].get("seed", 42)),
        suffix=str(cfg["experiment"].get("tag", "")) or None,
    )
    return ensure_project_paths(ROOT, run_name=run_name)["run_dir"]


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_last_valid_attention(attn_csv: Path) -> Dict[str, float]:
    if not attn_csv.exists():
        return {}
    with open(attn_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid_rows = [r for r in rows if r.get("phase") == "valid"]
    if not valid_rows:
        return {}
    row = valid_rows[-1]
    return {
        "attn_min": float(row.get("attn_min", "nan")),
        "attn_max": float(row.get("attn_max", "nan")),
        "attn_mean": float(row.get("attn_mean", "nan")),
        "attn_std": float(row.get("attn_std", "nan")),
    }


def _read_best_valid_metrics(epoch_csv: Path) -> Dict[str, float]:
    if not epoch_csv.exists():
        return {}
    with open(epoch_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = min(rows, key=lambda r: float(r["valid_rmse"]))
    return {
        "best_epoch": int(float(best["epoch"])),
        "best_valid_rmse": float(best["valid_rmse"]),
        "best_valid_mae": float(best["valid_mae"]),
        "best_valid_r2": float(best.get("valid_pred_r2", "nan")),
        "best_valid_mean_error": float(best.get("valid_pred_mean_bias", "nan")),
        "pred_mean": float(best.get("valid_pred_pred_mean", "nan")),
        "true_mean": float(best.get("valid_pred_true_mean", "nan")),
        "pred_std": float(best.get("valid_pred_pred_std", "nan")),
        "true_std": float(best.get("valid_pred_true_std", "nan")),
    }


def collect_summary(label: str, config_path: Path) -> Dict[str, object]:
    _, _, load_yaml, _ = _load_project_utils()
    cfg = load_yaml(config_path)
    run_dir = run_dir_from_config(config_path)
    metrics_path = run_dir / "results" / "test_metrics.json"
    epoch_csv = run_dir / "logs" / "epoch_metrics.csv"
    attn_csv = run_dir / "logs" / "attention_epoch_stats.csv"
    checkpoint = run_dir / "checkpoints" / "best_model.pth"

    row: Dict[str, object] = {
        "experiment": label,
        "config": str(config_path),
        "attention": bool(cfg.get("model", {}).get("use_attention", True)),
        "loss_name": str(cfg.get("train", {}).get("loss_name", "mse")),
        "bias_regularization_weight": float(cfg.get("train", {}).get("bias_regularization_weight", 0.0)),
        "output_dir": str(run_dir),
        "best_checkpoint": str(checkpoint),
    }
    row.update(_read_best_valid_metrics(epoch_csv))
    row.update(_read_last_valid_attention(attn_csv))
    if metrics_path.exists():
        row.update(_read_json(metrics_path))
    return row


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []

    for label, rel_cfg in EXPERIMENTS:
        cfg_path = ROOT / rel_cfg
        cmd = [args.python, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path)]
        print(f"\n[{label}] {' '.join(cmd)}")
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"Experiment {label} failed with exit code {completed.returncode}")
        rows.append(collect_summary(label, cfg_path))

    if rows:
        _, ensure_project_paths, _, save_csv_rows = _load_project_utils()
        shared_paths = ensure_project_paths(ROOT)
        save_csv_rows(shared_paths["results"] / "ablation_summary.csv", rows)
        md_path = shared_paths["results"] / "ablation_summary.md"
        headers = list(rows[0].keys())
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
        print(f"\nSaved summary: {shared_paths['results'] / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
