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
    ("A", "backbone", Path("configs/config_backbone.yaml")),
    ("B", "attention", Path("configs/config_attention.yaml")),
    ("C", "main_loss", Path("configs/config_loss.yaml")),
    ("D", "full", Path("configs/config_full.yaml")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4 ablation experiments sequentially")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing A/B/C/D test metrics into the ablation table")
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


def collect_summary(variant: str, variant_name: str, config_path: Path) -> Dict[str, object]:
    _, _, load_yaml, _ = _load_project_utils()
    cfg = load_yaml(config_path)
    run_dir = run_dir_from_config(config_path)
    metrics_path = run_dir / "results" / "test_metrics.json"
    epoch_csv = run_dir / "logs" / "epoch_metrics.csv"
    attn_csv = run_dir / "logs" / "attention_epoch_stats.csv"
    checkpoint = run_dir / "checkpoints" / "best_model.pth"

    row: Dict[str, object] = {
        "variant": variant,
        "variant_name": variant_name,
        "config": str(config_path),
        "attention_on": bool(cfg.get("model", {}).get("use_attention", True)),
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


def _formalize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    formal_rows: List[Dict[str, object]] = []
    for row in rows:
        formal_rows.append(
            {
                "variant": row.get("variant"),
                "variant_name": row.get("variant_name"),
                "attention_on": row.get("attention_on"),
                "loss_name": row.get("loss_name"),
                "bias_regularization_weight": row.get("bias_regularization_weight"),
                "rmse": row.get("rmse"),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "mean_bias": row.get("mean_bias"),
                "pred_std": row.get("pred_std"),
                "true_std": row.get("true_std"),
                "pred_true_std_ratio": row.get("pred_true_std_ratio"),
                "best_epoch": row.get("best_epoch"),
                "best_checkpoint": row.get("best_checkpoint"),
                "output_dir": row.get("output_dir"),
            }
        )
    return formal_rows


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []

    for variant, variant_name, rel_cfg in EXPERIMENTS:
        cfg_path = ROOT / rel_cfg
        cmd = [args.python, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path)]
        print(f"\n[{variant}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        if args.collect_only:
            rows.append(collect_summary(variant, variant_name, cfg_path))
            continue

        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"Experiment {variant} failed with exit code {completed.returncode}")
        rows.append(collect_summary(variant, variant_name, cfg_path))

    if rows:
        _, ensure_project_paths, _, save_csv_rows = _load_project_utils()
        shared_paths = ensure_project_paths(ROOT)
        save_csv_rows(shared_paths["results"] / "ablation_summary.csv", rows)
        formal_rows = _formalize_rows(rows)
        save_csv_rows(shared_paths["results"] / "ablation_table.csv", formal_rows)
        md_path = shared_paths["results"] / "ablation_table.md"
        headers = list(formal_rows[0].keys())
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in formal_rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
        print(f"\nSaved ablation table: {shared_paths['results'] / 'ablation_table.csv'}")


if __name__ == "__main__":
    main()
