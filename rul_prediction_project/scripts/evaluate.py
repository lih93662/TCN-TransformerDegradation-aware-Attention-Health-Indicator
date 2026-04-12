"""Evaluation entrypoint for saved hybrid PHM2012 RUL model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_model, group_predictions_by_id, grouped_regression_metrics
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.loss import compute_regression_metrics, phm2012_score
from src.preprocess import prepare_datasets
from src.utils import (
    build_run_name,
    configure_logging,
    ensure_project_paths,
    load_yaml,
    save_csv_rows,
    save_json,
    set_seed,
)
VARIANT_LABELS = {
    "ablation_backbone": "A",
    "ablation_attention": "B",
    "ablation_loss": "C",
    "ablation_full": "D",
}

from src.visualization import (
    plot_attention,
    plot_attention_weights,
    plot_hi_curve,
    plot_prediction_distribution,
    plot_prediction_scatter,
    plot_residual_histogram,
    plot_residual_vs_target,
    plot_rul_curve,
    plot_single_bearing_prediction,
)

def _safe_load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid PHM2012 RUL model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config_loss.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
        help="Checkpoint file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["valid", "test"],
        help="Dataset split used for evaluation.",
    )
    return parser.parse_args()


def _build_model(cfg: Dict, sensor_dim: int, device: torch.device) -> HybridRULModel:
    m = cfg["model"]
    model_cfg = ModelConfig(
        sensor_dim=sensor_dim,
        tcn_channels=int(m["tcn_channels"]),
        tcn_kernel_size=int(m["tcn_kernel_size"]),
        tcn_dilations=tuple(m["tcn_dilations"]),
        transformer_embed_dim=int(m["transformer_embed_dim"]),
        transformer_heads=int(m["transformer_heads"]),
        transformer_layers=int(m["transformer_layers"]),
        transformer_ffn_dim=int(m["transformer_ffn_dim"]),
        dropout=float(m["dropout"]),
        backbone_variant=str(m.get("backbone_variant", "tcn_transformer")),
        use_hi=bool(m.get("use_hi", True)),
        hi_input_source=str(m.get("hi_input_source", "tcn")),
        hi_output_temperature=float(m.get("hi_output_temperature", 1.2)),
        use_attention=bool(m.get("use_attention", True)),
        attention_use_hi_bias=bool(m.get("attention_use_hi_bias", True)),
        attention_use_hi_logit=bool(m.get("attention_use_hi_logit", True)),
        attention_use_temporal_gate=bool(m.get("attention_use_temporal_gate", True)),
        attention_use_recency_bias=bool(m.get("attention_use_recency_bias", True)),
        attention_conditioning_gain=float(m.get("attention_conditioning_gain", 2.0)),
        attention_head_bias_gain=float(m.get("attention_head_bias_gain", 2.0)),
        attention_temperature=float(m.get("attention_temperature", 1.0)),
        attention_recency_strength=float(m.get("attention_recency_strength", 0.5)),
        head_hidden_dim=int(m.get("head_hidden_dim", 32)),
        output_activation=str(m.get("output_activation", "identity")),
    )
    return HybridRULModel(model_cfg).to(device)


def _attention_average(attention_maps: List[np.ndarray], window_size: int) -> np.ndarray:
    if not attention_maps:
        return np.eye(window_size, dtype=np.float32)
    stacked = np.stack(attention_maps, axis=0).astype(np.float32)
    return stacked.mean(axis=(0, 1))

def _figure_path(figures_dir: Path, filename: str) -> Path:
    path = figures_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    run_name = build_run_name(
        base_name=str(cfg["experiment"].get("name", "rul_experiment")),
        seed=seed,
        suffix=str(cfg["experiment"].get("tag", "")) or None,
    )
    paths = ensure_project_paths(ROOT, run_name=run_name)
    logger = configure_logging(paths["run_logs"] / f"evaluate_{args.split}.log")
    variant_tag = str(cfg["experiment"].get("tag", ""))
    logger.info("Evaluation variant: %s (%s)", VARIANT_LABELS.get(variant_tag, "main"), variant_tag or "default")

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

    split_dataset = prepared.test_dataset if args.split == "test" else prepared.valid_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, sensor_dim=prepared.feature_dim, device=device)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        fallback = paths.get("run_checkpoints", paths["checkpoints"]) / "best_model.pth"
        ckpt = fallback
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    payload = _safe_load_checkpoint(ckpt, device)
    model.load_state_dict(payload["model_state"])
    logger.info("Loaded checkpoint %s from epoch %s", ckpt, payload.get("epoch", "unknown"))

    loader = DataLoader(split_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    result = evaluate_model(model, loader, device)

    y_true = result.y_true.astype(np.float32)
    y_pred = result.y_pred.astype(np.float32)
    residuals = y_pred - y_true

    norm_metrics = compute_regression_metrics(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    norm_metrics["phm_score"] = phm2012_score(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    norm_metrics["num_samples"] = int(len(y_true))
    norm_metrics["pred_true_std_ratio"] = norm_metrics["pred_std"] / max(norm_metrics["true_std"], 1e-8)

    target_scale = float(prepared.target_scale)
    y_true_raw = y_true * target_scale
    y_pred_raw = y_pred * target_scale
    raw_metrics = compute_regression_metrics(torch.from_numpy(y_pred_raw), torch.from_numpy(y_true_raw))
    raw_metrics["phm_score"] = phm2012_score(torch.from_numpy(y_pred_raw), torch.from_numpy(y_true_raw))
    raw_metrics["target_scale"] = target_scale
    raw_metrics["pred_true_std_ratio"] = raw_metrics["pred_std"] / max(raw_metrics["true_std"], 1e-8)

    grouped_rows = grouped_regression_metrics(result.ids, y_true, y_pred)
    grouped_rows_raw = grouped_regression_metrics(result.ids, y_true_raw, y_pred_raw)
    for row_norm, row_raw in zip(grouped_rows, grouped_rows_raw):
        row_norm["rmse_raw"] = row_raw["rmse"]
        row_norm["mae_raw"] = row_raw["mae"]
        row_norm["mean_bias_raw"] = row_raw["mean_bias"]

    attention_mean = _attention_average(result.attention_maps, window_size=int(data_cfg.get("window_size", 40)))
    temporal_attention_examples = result.temporal_attention[: int(cfg.get("evaluation", {}).get("attention_num_samples", 3))]

    eval_dir = paths["run_results"] / f"evaluation_{args.split}"
    figures_dir = eval_dir / "figures"
    tables_dir = eval_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_rul_curve(y_true, y_pred, _figure_path(figures_dir, "prediction_timeseries_normalized.png"))
    plot_rul_curve(y_true_raw, y_pred_raw, _figure_path(figures_dir, "prediction_timeseries_raw.png"))
    plot_prediction_scatter(y_true, y_pred, _figure_path(figures_dir, "prediction_scatter_normalized.png"))
    plot_prediction_scatter(y_true_raw, y_pred_raw, _figure_path(figures_dir, "prediction_scatter_raw.png"))
    plot_residual_histogram(residuals, _figure_path(figures_dir, "residual_histogram_normalized.png"))
    plot_residual_histogram(y_pred_raw - y_true_raw, _figure_path(figures_dir, "residual_histogram_raw.png"))
    plot_residual_vs_target(y_true, residuals, _figure_path(figures_dir, "residual_vs_target_normalized.png"))
    plot_residual_vs_target(y_true_raw, y_pred_raw - y_true_raw, _figure_path(figures_dir, "residual_vs_target_raw.png"))
    plot_prediction_distribution(y_true, y_pred, _figure_path(figures_dir, "prediction_distribution_normalized.png"))
    plot_prediction_distribution(y_true_raw, y_pred_raw, _figure_path(figures_dir, "prediction_distribution_raw.png"))
    plot_hi_curve(result.hi, _figure_path(figures_dir, "health_indicator_curve.png"))
    plot_attention(attention_mean, _figure_path(figures_dir, "attention_heatmap_mean.png"))

    for idx, temporal_weights in enumerate(temporal_attention_examples, start=1):
        plot_attention_weights(
            np.asarray(temporal_weights, dtype=np.float32),
            _figure_path(figures_dir, f"temporal_attention_sample_{idx}.png"),
            title=f"Temporal Attention Weights Sample {idx}",
        )

    grouped = group_predictions_by_id(result.ids, y_true_raw, y_pred_raw, result.hi)
    hi_summary = {
        "hi_mean": float(np.mean(result.hi)),
        "hi_std": float(np.std(result.hi)),
        "hi_min": float(np.min(result.hi)),
        "hi_max": float(np.max(result.hi)),
    }
    corr = np.corrcoef(y_true, result.hi)[0, 1] if len(y_true) > 1 else np.nan
    corr_pred = np.corrcoef(y_pred, result.hi)[0, 1] if len(y_pred) > 1 else np.nan
    hi_summary["corr_hi_true_rul"] = float(corr) if np.isfinite(corr) else float("nan")
    hi_summary["corr_hi_pred_rul"] = float(corr_pred) if np.isfinite(corr_pred) else float("nan")

    scatter_rows = []
    for i, (bid, yt, yp, hi) in enumerate(zip(result.ids, y_true_raw, y_pred_raw, result.hi)):
        scatter_rows.append(
            {
                "sample_index": i,
                "group": bid,
                "true_rul_raw": float(yt),
                "pred_rul_raw": float(yp),
                "error_raw": float(yp - yt),
                "health_indicator": float(hi),
            }
        )
    for bearing_id, arrays in list(grouped.items())[:3]:
        plot_single_bearing_prediction(
            bearing_id,
            arrays["y_true"],
            arrays["y_pred"],
            _figure_path(figures_dir, f"bearing_{bearing_id}_timeseries.png"),
        )

    save_json(
        tables_dir / "metrics.json",
        {
            "split": args.split,
            "checkpoint": str(ckpt),
            "normalized": norm_metrics,
            "raw": raw_metrics,
        },
    )
    save_csv_rows(tables_dir / "grouped_metrics.csv", grouped_rows)
    save_csv_rows(tables_dir / "prediction_scatter_raw.csv", scatter_rows)
    save_json(tables_dir / "hi_summary.json", hi_summary)

    logger.info("TEST RESULTS (%s, normalized scale)", args.split.upper())
    logger.info("RMSE: %.6f", norm_metrics["rmse"])
    logger.info("MAE: %.6f", norm_metrics["mae"])
    logger.info("R2: %.6f", norm_metrics["r2"])
    logger.info("Mean Bias: %+.6f", norm_metrics["mean_bias"])
    logger.info("Pred Std: %.6f", norm_metrics["pred_std"])
    logger.info("True Std: %.6f", norm_metrics["true_std"])
    logger.info("Pred/True Std Ratio: %.6f", norm_metrics["pred_true_std_ratio"])
    logger.info("Evaluation normalized metrics: %s", norm_metrics)
    logger.info("Evaluation raw metrics: %s", raw_metrics)
    logger.info("HI summary: %s", hi_summary)
    logger.info("Saved evaluation artifacts under %s", eval_dir)


if __name__ == "__main__":
    main()
