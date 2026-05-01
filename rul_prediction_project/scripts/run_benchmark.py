"""Run paper baseline benchmark (4 baselines + full model) and export main tables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import build_run_name, ensure_project_paths, load_yaml, save_csv_rows


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config: Path


BASELINES: List[BenchmarkSpec] = [
    BenchmarkSpec("TCN only", Path("configs/ablations/backbone_tcn_only.yaml")),
    BenchmarkSpec("Transformer only", Path("configs/ablations/backbone_transformer_only.yaml")),
    BenchmarkSpec("TCN + Transformer", Path("configs/ablations/backbone_tcn_transformer.yaml")),
    BenchmarkSpec("Full model", Path("configs/config.yaml")),
]


def _run_dir_from_config(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    run_name = build_run_name(
        base_name=str(cfg["experiment"].get("name", "rul_experiment")),
        seed=int(cfg["experiment"].get("seed", 42)),
        suffix=str(cfg["experiment"].get("tag", "")) or None,
    )
    return ensure_project_paths(ROOT, run_name=run_name)["run_dir"]


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline benchmark experiments")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []

    for spec in BASELINES:
        cfg_path = ROOT / spec.config
        cmd = [args.python, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path)]
        print(f"[{spec.name}] {' '.join(cmd)}")
        if not args.dry_run and not args.collect_only:
            completed = subprocess.run(cmd, cwd=ROOT)
            if completed.returncode != 0:
                raise SystemExit(f"Failed benchmark run for {spec.name}: exit={completed.returncode}")

        run_dir = _run_dir_from_config(cfg_path)
        metrics_path = run_dir / "results" / "test_metrics.json"
        if not metrics_path.exists():
            print(f"WARNING: metrics not found for {spec.name}: {metrics_path}")
            continue
        metrics = _read_json(metrics_path)
        rows.append(
            {
                "model": spec.name,
                "config": str(spec.config),
                "run_dir": str(run_dir),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "r2": metrics.get("r2"),
                "bias": metrics.get("mean_bias", metrics.get("bias")),
                "pearson_corr": metrics.get("pearson_corr"),
                "pred_std": metrics.get("pred_std"),
                "true_std": metrics.get("true_std"),
                "pred_true_std_ratio": metrics.get("pred_true_std_ratio"),
                "phm_score": metrics.get("phm_score"),
            }
        )

    if not rows:
        print("No benchmark rows collected.")
        return

    paths = ensure_project_paths(ROOT)
    csv_path = paths["results"] / "main_results.csv"
    md_path = paths["results"] / "main_results.md"
    save_csv_rows(csv_path, rows)

    headers = list(rows[0].keys())
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")

    print(f"Saved benchmark tables: {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
