"""Scalable launcher for publication-ready ablation suites."""

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

from scripts.ablation_registry import AblationSpec, get_specs, list_suites, to_rows


def _load_project_utils():
    from src.utils import build_run_name, ensure_project_paths, load_yaml, save_csv_rows

    return build_run_name, ensure_project_paths, load_yaml, save_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments sequentially")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect existing ablation metrics into the summary table",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        help=f"Suite name: all, {', '.join(list_suites())}",
    )
    parser.add_argument("--code", type=str, default=None, help="Run one experiment code (e.g., A, OPT1, HI2)")
    parser.add_argument("--list", action="store_true", help="List registered ablations and exit")
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


def collect_summary(spec: AblationSpec) -> Dict[str, object]:
    _, _, load_yaml, _ = _load_project_utils()
    cfg_path = ROOT / spec.config
    cfg = load_yaml(cfg_path)
    run_dir = run_dir_from_config(cfg_path)
    metrics_path = run_dir / "results" / "test_metrics.json"
    epoch_csv = run_dir / "logs" / "epoch_metrics.csv"
    checkpoint = run_dir / "checkpoints" / "best_model.pth"

    row: Dict[str, object] = {
        "code": spec.code,
        "name": spec.name,
        "suite": spec.suite,
        "question": spec.question,
        "config": str(spec.config),
        "attention_on": bool(cfg.get("model", {}).get("use_attention", True)),
        "use_hi": bool(cfg.get("model", {}).get("use_hi", True)),
        "backbone_variant": str(cfg.get("model", {}).get("backbone_variant", "tcn_transformer")),
        "loss_name": str(cfg.get("train", {}).get("loss_name", "mse")),
        "use_mae_term": bool(cfg.get("train", {}).get("use_mae_term", False)),
        "use_bias_regularization": bool(cfg.get("train", {}).get("use_bias_regularization", False)),
        "bias_regularization_weight": float(cfg.get("train", {}).get("bias_regularization_weight", 0.0)),
        "output_dir": str(run_dir),
        "best_checkpoint": str(checkpoint),
    }
    row.update(_read_best_valid_metrics(epoch_csv))
    if metrics_path.exists():
        row.update(_read_json(metrics_path))
    return row


def _formalize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = [
        "code",
        "name",
        "suite",
        "attention_on",
        "use_hi",
        "backbone_variant",
        "loss_name",
        "use_mae_term",
        "use_bias_regularization",
        "bias_regularization_weight",
        "rmse",
        "mae",
        "r2",
        "mean_bias",
        "pred_std",
        "true_std",
        "pred_true_std_ratio",
        "best_epoch",
        "best_checkpoint",
        "output_dir",
    ]
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append({k: row.get(k) for k in keys})
    return out


def main() -> None:
    args = parse_args()
    specs = get_specs(suite=args.suite, code=args.code)

    if args.list:
        print(json.dumps(to_rows(specs or get_specs()), indent=2, ensure_ascii=False))
        return

    if not specs:
        suites = ", ".join(list_suites())
        raise SystemExit(f"No ablation specs selected. Use --list. Available suites: {suites}")

    rows: List[Dict[str, object]] = []

    for spec in specs:
        cfg_path = ROOT / spec.config
        cmd = [args.python, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path)]
        print(f"\n[{spec.code} | {spec.suite}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        if not args.collect_only:
            completed = subprocess.run(cmd, cwd=ROOT)
            if completed.returncode != 0:
                raise SystemExit(f"Experiment {spec.code} failed with exit code {completed.returncode}")
        rows.append(collect_summary(spec))

    if rows:
        _, ensure_project_paths, _, save_csv_rows = _load_project_utils()
        shared_paths = ensure_project_paths(ROOT)
        save_csv_rows(shared_paths["results"] / "ablation_summary.csv", rows)
        formal_rows = _formalize_rows(rows)
        save_csv_rows(shared_paths["results"] / "ablation_table.csv", formal_rows)
        with open(shared_paths["results"] / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
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
