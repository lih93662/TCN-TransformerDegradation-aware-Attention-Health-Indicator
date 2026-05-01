"""Generate paper artifact figures from evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import build_run_name, ensure_project_paths, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SCI paper artifact figures")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    p.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    return p.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)

    run_name = build_run_name(
        base_name=str(cfg["experiment"].get("name", "rul_experiment")),
        seed=int(cfg["experiment"].get("seed", 42)),
        suffix=str(cfg["experiment"].get("tag", "")) or None,
    )
    run_paths = ensure_project_paths(ROOT, run_name=run_name)
    eval_tables = run_paths["run_results"] / f"evaluation_{args.split}" / "tables"
    eval_figures = run_paths["run_results"] / f"evaluation_{args.split}" / "figures"

    out_dir = run_paths["run_dir"] / "paper_artifacts" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    scatter_rows = _read_csv(eval_tables / "prediction_scatter_raw.csv")
    grouped_rows = _read_csv(eval_tables / "grouped_metrics.csv")

    if not scatter_rows:
        raise FileNotFoundError(f"Missing evaluation scatter table: {eval_tables / 'prediction_scatter_raw.csv'}")

    y_true = np.asarray([float(r["true_rul_raw"]) for r in scatter_rows], dtype=np.float32)
    y_pred = np.asarray([float(r["pred_rul_raw"]) for r in scatter_rows], dtype=np.float32)
    hi = np.asarray([float(r["health_indicator"]) for r in scatter_rows], dtype=np.float32)
    groups = [r["group"] for r in scatter_rows]

    # 1) predicted vs true curve per bearing
    unique_groups = []
    for g in groups:
        if g not in unique_groups:
            unique_groups.append(g)
    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        plt.figure(figsize=(9, 4))
        plt.plot(y_true[idx], label="true", linewidth=2)
        plt.plot(y_pred[idx], label="pred", linestyle="--", linewidth=1.8)
        plt.title(f"Pred vs True RUL | {g}")
        plt.xlabel("window index")
        plt.ylabel("RUL (raw)")
        plt.legend()
        _savefig(out_dir / f"rul_curve_{g}.png")

    # 2) pred vs true scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
    plt.xlabel("true RUL (raw)")
    plt.ylabel("pred RUL (raw)")
    plt.title("Pred vs True Scatter")
    _savefig(out_dir / "pred_vs_true_scatter.png")

    # 3) HI curve per bearing
    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        plt.figure(figsize=(9, 3.5))
        plt.plot(hi[idx], color="#7c3aed")
        plt.xlabel("window index")
        plt.ylabel("HI")
        plt.title(f"HI curve | {g}")
        _savefig(out_dir / f"hi_curve_{g}.png")

    # 4) attention mean heatmap (copied from evaluation output if present)
    attn_fig = eval_figures / "attention_heatmap_mean.png"
    if attn_fig.exists():
        import shutil

        shutil.copy2(attn_fig, out_dir / "attention_mean_heatmap.png")

    # 5) training/validation loss curve (copied)
    train_curve = run_paths["run_figures"] / "training_loss_curve.png"
    if train_curve.exists():
        import shutil

        shutil.copy2(train_curve, out_dir / "training_validation_loss_curve.png")

    # 6) per-bearing RMSE bar chart
    if grouped_rows:
        labels = [r["group"] for r in grouped_rows]
        rmse = [float(r["rmse_raw"]) if r.get("rmse_raw") else float(r["rmse"]) for r in grouped_rows]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, rmse, color="#2563eb")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("RMSE (raw)")
        plt.title("Per-bearing RMSE")
        _savefig(out_dir / "per_bearing_rmse_bar.png")

    print(f"Saved paper figures to {out_dir}")


if __name__ == "__main__":
    main()
