"""Evaluation entrypoint for saved hybrid PHM2012 RUL model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import (
    evaluate_model,
    group_predictions_by_id,
    group_predictions_by_life_stage,
    summarize_group_metrics,
)
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.loss import compute_regression_metrics, phm2012_score
from src.preprocess import prepare_datasets
from src.utils import (
    configure_logging,
    ensure_project_paths,
    experiment_run_name,
    load_yaml,
    resolve_seeds,
    save_csv_rows,
    save_json,
    set_seed,
)
from src.visualization import (
    plot_attention,
    plot_grouped_metrics,
    plot_hi_curve,
    plot_prediction_distribution,
    plot_prediction_scatter,
    plot_residual_histogram,
    plot_residual_vs_target,
    plot_rul_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid PHM2012 RUL model")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
        help="Checkpoint file. For multi-seed runs this can be overridden per run.",
    )
    parser.add_argument("--attention_samples", type=int, default=3, help="Number of sample attention maps to export.")
    return parser.parse_args()


def build_model_config(cfg: Dict, sensor_dim: int) -> ModelConfig:
    m = cfg["model"]
    return ModelConfig(
        sensor_dim=sensor_dim,
        tcn_channels=int(m["tcn_channels"]),
        tcn_kernel_size=int(m["tcn_kernel_size"]),
        tcn_dilations=tuple(m["tcn_dilations"]),
        transformer_embed_dim=int(m["transformer_embed_dim"]),
        transformer_heads=int(m["transformer_heads"]),
        transformer_layers=int(m["transformer_layers"]),
        transformer_ffn_dim=int(m["transformer_ffn_dim"]),
        dropout=float(m["dropout"]),
        attention_mode=str(m.get("attention_mode", "degradation")),
        attention_temperature=float(m.get("attention_temperature", 1.0)),
        attention_use_qk_norm=bool(m.get("attention_use_qk_norm", False)),
        attention_bias_scale=float(m.get("attention_bias_scale", 1.0)),
        prediction_activation=str(m.get("prediction_activation", "identity")),
    )


def _aggregate_attention_map(attn: torch.Tensor) -> np.ndarray:
    attn_avg = attn.detach().cpu()
    while attn_avg.ndim > 2:
        attn_avg = attn_avg.mean(dim=0)
    arr = attn_avg.numpy().astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def export_attention_samples(
    model: HybridRULModel,
    dataset,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            out = model(x)
            attn = _aggregate_attention_map(out["attn_map"])
            save_path = output_dir / f"sample_{idx:03d}_{sample['id']}.png"
            plot_attention(attn, save_path, title=f"Attention Sample {idx:03d}")
            rows.append({
                "sample_index": idx,
                "bearing_id": sample["id"],
                "prediction": float(out["pred"].item()),
                "hi": float(out["hi"].item()),
                "file": str(save_path),
            })
    return rows


def run_single_evaluation(cfg: Dict, seed: int, run_name: str, checkpoint_override: str | None, multi_seed: bool, attention_samples: int) -> Dict[str, float]:
    paths = ensure_project_paths(ROOT, run_name if multi_seed else None)
    logger = configure_logging(paths["logs"] / "evaluate.log")
    set_seed(seed)

    data_cfg = cfg["data"]
    prepared = prepare_datasets(
        dataset_root=data_cfg["dataset_root"],
        window_size=int(data_cfg.get("window_size", 40)),
        stride=int(data_cfg.get("stride", 10)),
        max_rul=int(data_cfg.get("max_rul", 125)),
        valid_ratio=float(data_cfg.get("valid_ratio", 0.2)),
        seed=seed,
        sensor_dim=int(data_cfg.get("sensor_dim", 2)),
        max_windows_per_bearing=int(data_cfg.get("max_windows_per_bearing", 20000)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRULModel(build_model_config(cfg, sensor_dim=prepared.feature_dim)).to(device)

    ckpt = Path(checkpoint_override) if checkpoint_override else (paths["checkpoints"] / "best_model.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state"])
    logger.info("Loaded checkpoint from epoch %s", payload.get("epoch", "unknown"))

    loader = DataLoader(prepared.test_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    result = evaluate_model(model, loader, device)

    y_true_t = torch.from_numpy(result.y_true)
    y_pred_t = torch.from_numpy(result.y_pred)
    metrics = compute_regression_metrics(y_pred_t, y_true_t)
    metrics["phm_score"] = phm2012_score(y_pred_t, y_true_t)
    metrics["num_samples"] = int(len(result.y_true))
    metrics["seed"] = seed
    metrics["run_name"] = run_name

    save_json(paths["logs"] / "evaluation_metrics.json", metrics)
    logger.info("Evaluation summary: %s", metrics)

    plot_rul_curve(result.y_true, result.y_pred, paths["figures"] / "rul_curve.png")
    plot_hi_curve(result.hi, paths["figures"] / "hi_curve.png")
    plot_prediction_scatter(result.y_true, result.y_pred, paths["figures"] / "pred_vs_true_scatter.png")
    plot_residual_histogram(result.y_true, result.y_pred, paths["figures"] / "residual_histogram.png")
    plot_residual_vs_target(result.y_true, result.y_pred, paths["figures"] / "residual_vs_target.png")
    plot_prediction_distribution(result.y_true, result.y_pred, paths["figures"] / "prediction_distribution.png")

    grouped_ids = group_predictions_by_id(result.ids, result.y_true, result.y_pred, result.hi)
    grouped_life = group_predictions_by_life_stage(result.y_true, result.y_pred, result.hi)

    id_rows = [row.__dict__ for row in summarize_group_metrics(grouped_ids)]
    life_rows = [row.__dict__ for row in summarize_group_metrics(grouped_life)]
    save_csv_rows(paths["results"] / "group_metrics_by_id.csv", id_rows)
    save_csv_rows(paths["results"] / "group_metrics_by_life_stage.csv", life_rows)
    if id_rows:
        plot_grouped_metrics(id_rows[: min(10, len(id_rows))], paths["figures"] / "group_metrics_by_id.png", "Per-Bearing Metrics")
    if life_rows:
        plot_grouped_metrics(life_rows, paths["figures"] / "group_metrics_by_life_stage.png", "Life-Stage Metrics")

    attention_rows = export_attention_samples(
        model=model,
        dataset=prepared.test_dataset,
        device=device,
        output_dir=paths["figures"] / "attention_samples",
        num_samples=int(attention_samples),
    )
    save_json(paths["results"] / "attention_samples.json", {"samples": attention_rows})
    logger.info("Saved paper-ready evaluation figures under %s", paths["figures"])
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    exp_cfg = cfg.get("experiment", {})
    seeds = resolve_seeds(exp_cfg)
    base_name = str(exp_cfg.get("name", "tcn_transformer_degradation_attention_hi"))
    multi_seed = len(seeds) > 1

    summaries: List[Dict[str, float]] = []
    for seed in seeds:
        run_name = experiment_run_name(base_name, seed, multi_seed=multi_seed)
        checkpoint = None if multi_seed else args.checkpoint
        summaries.append(run_single_evaluation(cfg, seed, run_name, checkpoint, multi_seed, args.attention_samples))

    if len(summaries) > 1:
        agg: Dict[str, Dict[str, float]] = {}
        for key in ("rmse", "mae", "r2", "phm_score", "pred_mean", "pred_std", "true_mean", "true_std", "mean_error"):
            values = [float(row[key]) for row in summaries]
            agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        agg_paths = ensure_project_paths(ROOT, base_name)
        save_json(agg_paths["results"] / "evaluation_multi_seed_summary.json", {"runs": summaries, "aggregate": agg})


if __name__ == "__main__":
    main()
