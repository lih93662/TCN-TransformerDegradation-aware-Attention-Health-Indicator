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

from src.checkpoint_compat import (
    detect_checkpoint_head_variant,
    load_model_state_strict,
    resolve_rul_head_mode,
)
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
    parser.add_argument(
        "--max-attention-samples",
        type=int,
        default=None,
        help="Optional cap on number of attention samples to store/export.",
    )
    parser.add_argument(
        "--skip-attention-export",
        action="store_true",
        help="Skip attention averaging/plots to reduce memory use.",
    )
    parser.add_argument(
        "--disable-linear-calibration",
        action="store_true",
        help="Disable post-hoc linear calibration fitted on validation predictions.",
    )
    return parser.parse_args()


def _build_model(
    cfg: Dict,
    sensor_dim: int,
    device: torch.device,
    checkpoint_head_variant: str | None = None,
) -> HybridRULModel:
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
        use_fixed_hi=bool(m.get("use_fixed_hi", False)),
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
        rul_head_mode=resolve_rul_head_mode(
            m,
            default="hi_guided_residual",
            checkpoint_variant=checkpoint_head_variant,
        ),
        hi_residual_scale=float(m.get("hi_residual_scale", 0.3)),
        output_activation=str(m.get("output_activation", "identity")),
    )
    return HybridRULModel(model_cfg).to(device)


def _attention_average(attention_maps: List[np.ndarray], window_size: int) -> np.ndarray:
    if not attention_maps:
        return np.eye(window_size, dtype=np.float32)
    running_sum: np.ndarray | None = None
    count = 0
    for amap in attention_maps:
        arr = np.asarray(amap, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        elif arr.ndim == 2:
            pass
        else:
            continue
        if running_sum is None:
            running_sum = np.zeros_like(arr, dtype=np.float32)
        running_sum += arr
        count += 1
    if running_sum is None or count == 0:
        return np.eye(window_size, dtype=np.float32)
    return running_sum / float(count)

def _figure_path(figures_dir: Path, filename: str) -> Path:
    path = figures_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fit_linear_calibration(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    x = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if x.size < 2 or float(np.std(x)) < 1e-8:
        return 1.0, 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-12:
        return 1.0, 0.0
    a = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    b = float(y_mean - a * x_mean)
    return a, b


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})

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
        scaler_mode=str(data_cfg.get("scaler_mode", "standard")),
        valid_bearing_ids=data_cfg.get("valid_bearing_ids"),
        valid_split_variant=data_cfg.get("valid_split_variant"),
        late_stage_threshold=float(data_cfg.get("late_stage_threshold", 0.2)),
        late_stage_oversample_factor=float(data_cfg.get("late_stage_oversample_factor", 1.0)),
        balance_train_rul_bins=bool(data_cfg.get("balance_train_rul_bins", False)),
        rul_bin_edges=data_cfg.get("rul_bin_edges"),
        train_balance_mode=str(data_cfg.get("train_balance_mode", "hybrid")),
        train_balance_target=str(data_cfg.get("train_balance_target", "median")),
        train_balance_custom_count=data_cfg.get("train_balance_custom_count"),
        train_balance_seed=data_cfg.get("train_balance_seed"),
        auto_valid_bearings_count=int(data_cfg.get("auto_valid_bearings_count", 1)),
        max_auto_valid_tries=int(data_cfg.get("max_auto_valid_tries", 16)),
        min_valid_active_bins=int(data_cfg.get("min_valid_active_bins", 3)),
        valid_skew_ratio_warn=float(data_cfg.get("valid_skew_ratio_warn", 25.0)),
    )

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        fallback = paths.get("run_checkpoints", paths["checkpoints"]) / "best_model.pth"
        ckpt = fallback
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    split_dataset = prepared.test_dataset if args.split == "test" else prepared.valid_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = _safe_load_checkpoint(ckpt, device)
    checkpoint_head_variant = detect_checkpoint_head_variant(payload["model_state"])
    model = _build_model(
        cfg,
        sensor_dim=prepared.feature_dim,
        device=device,
        checkpoint_head_variant=checkpoint_head_variant,
    )
    logger.info("Checkpoint head variant detected: %s", checkpoint_head_variant)
    load_model_state_strict(model, payload)
    logger.info("Loaded checkpoint %s from epoch %s", ckpt, payload.get("epoch", "unknown"))

    loader = DataLoader(split_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    cfg_max_attention = eval_cfg.get("max_attention_samples")
    max_attention_samples = (
        args.max_attention_samples
        if args.max_attention_samples is not None
        else (None if cfg_max_attention is None else int(cfg_max_attention))
    )
    skip_attention_export = bool(eval_cfg.get("skip_attention_export", False)) or bool(args.skip_attention_export)
    logger.info(
        "Attention export settings | skip=%s | max_attention_samples=%s | streaming_average=%s",
        skip_attention_export,
        max_attention_samples if max_attention_samples is not None else "all",
        True,
    )
    result = evaluate_model(
        model,
        loader,
        device,
        max_attention_samples=max_attention_samples,
        skip_attention_export=skip_attention_export,
    )

    y_true = result.y_true.astype(np.float32)
    y_pred = result.y_pred.astype(np.float32)
    residuals = y_pred - y_true
    calibration = {"enabled": False, "a": 1.0, "b": 0.0, "source_split": "none"}
    if args.split == "test" and not args.disable_linear_calibration:
        logger.info("Fitting linear calibration on validation split and applying to test split.")
        valid_loader = DataLoader(prepared.valid_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        valid_result = evaluate_model(
            model,
            valid_loader,
            device,
            max_attention_samples=0,
            skip_attention_export=True,
        )
        a, b = _fit_linear_calibration(valid_result.y_pred.astype(np.float32), valid_result.y_true.astype(np.float32))
        calibration = {"enabled": True, "a": float(a), "b": float(b), "source_split": "valid"}
        logger.info("Linear calibration coefficients: a=%.6f, b=%.6f", a, b)
    elif args.disable_linear_calibration:
        logger.info("Linear calibration disabled by CLI flag.")
    else:
        logger.info("Linear calibration skipped because evaluation split is '%s'.", args.split)

    y_pred_cal = (calibration["a"] * y_pred + calibration["b"]).astype(np.float32)

    norm_metrics = compute_regression_metrics(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    norm_metrics["phm_score"] = phm2012_score(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    norm_metrics["num_samples"] = int(len(y_true))
    norm_metrics["pred_true_std_ratio"] = norm_metrics["pred_std"] / max(norm_metrics["true_std"], 1e-8)
    norm_metrics_cal = compute_regression_metrics(torch.from_numpy(y_pred_cal), torch.from_numpy(y_true))
    norm_metrics_cal["phm_score"] = phm2012_score(torch.from_numpy(y_pred_cal), torch.from_numpy(y_true))
    norm_metrics_cal["num_samples"] = int(len(y_true))
    norm_metrics_cal["pred_true_std_ratio"] = norm_metrics_cal["pred_std"] / max(norm_metrics_cal["true_std"], 1e-8)

    target_scale = float(prepared.target_scale)
    y_true_raw = y_true * target_scale
    y_pred_raw = y_pred * target_scale
    y_pred_raw_cal = y_pred_cal * target_scale
    raw_metrics = compute_regression_metrics(torch.from_numpy(y_pred_raw), torch.from_numpy(y_true_raw))
    raw_metrics["phm_score"] = phm2012_score(torch.from_numpy(y_pred_raw), torch.from_numpy(y_true_raw))
    raw_metrics["target_scale"] = target_scale
    raw_metrics["pred_true_std_ratio"] = raw_metrics["pred_std"] / max(raw_metrics["true_std"], 1e-8)
    raw_metrics_cal = compute_regression_metrics(torch.from_numpy(y_pred_raw_cal), torch.from_numpy(y_true_raw))
    raw_metrics_cal["phm_score"] = phm2012_score(torch.from_numpy(y_pred_raw_cal), torch.from_numpy(y_true_raw))
    raw_metrics_cal["target_scale"] = target_scale
    raw_metrics_cal["pred_true_std_ratio"] = raw_metrics_cal["pred_std"] / max(raw_metrics_cal["true_std"], 1e-8)

    grouped_rows = grouped_regression_metrics(result.ids, y_true, y_pred)
    grouped_rows_cal = grouped_regression_metrics(result.ids, y_true, y_pred_cal)
    grouped_rows_raw = grouped_regression_metrics(result.ids, y_true_raw, y_pred_raw)
    grouped_rows_raw_cal = grouped_regression_metrics(result.ids, y_true_raw, y_pred_raw_cal)
    for row_norm, row_raw, row_norm_cal, row_raw_cal in zip(grouped_rows, grouped_rows_raw, grouped_rows_cal, grouped_rows_raw_cal):
        row_norm["rmse_raw"] = row_raw["rmse"]
        row_norm["mae_raw"] = row_raw["mae"]
        row_norm["mean_bias_raw"] = row_raw["mean_bias"]
        row_norm["rmse_calibrated"] = row_norm_cal["rmse"]
        row_norm["mae_calibrated"] = row_norm_cal["mae"]
        row_norm["mean_bias_calibrated"] = row_norm_cal["mean_bias"]
        row_norm["rmse_raw_calibrated"] = row_raw_cal["rmse"]
        row_norm["mae_raw_calibrated"] = row_raw_cal["mae"]
        row_norm["mean_bias_raw_calibrated"] = row_raw_cal["mean_bias"]

    attention_mean = _attention_average(result.attention_maps, window_size=int(data_cfg.get("window_size", 40)))
    temporal_attention_examples = result.temporal_attention[: int(eval_cfg.get("attention_num_samples", 3))]
    logger.info("Collected %d attention samples for export.", len(result.attention_maps))

    eval_dir = paths["run_results"] / f"evaluation_{args.split}"
    figures_dir = eval_dir / "figures"
    tables_dir = eval_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_rul_curve(y_true, y_pred, _figure_path(figures_dir, "prediction_timeseries_normalized.png"))
    plot_rul_curve(y_true, y_pred_cal, _figure_path(figures_dir, "prediction_timeseries_normalized_calibrated.png"))
    plot_rul_curve(y_true_raw, y_pred_raw, _figure_path(figures_dir, "prediction_timeseries_raw.png"))
    plot_rul_curve(y_true_raw, y_pred_raw_cal, _figure_path(figures_dir, "prediction_timeseries_raw_calibrated.png"))
    plot_prediction_scatter(y_true, y_pred, _figure_path(figures_dir, "prediction_scatter_normalized.png"))
    plot_prediction_scatter(y_true, y_pred_cal, _figure_path(figures_dir, "prediction_scatter_normalized_calibrated.png"))
    plot_prediction_scatter(y_true_raw, y_pred_raw, _figure_path(figures_dir, "prediction_scatter_raw.png"))
    plot_prediction_scatter(y_true_raw, y_pred_raw_cal, _figure_path(figures_dir, "prediction_scatter_raw_calibrated.png"))
    plot_residual_histogram(residuals, _figure_path(figures_dir, "residual_histogram_normalized.png"))
    plot_residual_histogram(y_pred_raw - y_true_raw, _figure_path(figures_dir, "residual_histogram_raw.png"))
    plot_residual_vs_target(y_true, residuals, _figure_path(figures_dir, "residual_vs_target_normalized.png"))
    plot_residual_vs_target(y_true_raw, y_pred_raw - y_true_raw, _figure_path(figures_dir, "residual_vs_target_raw.png"))
    plot_prediction_distribution(y_true, y_pred, _figure_path(figures_dir, "prediction_distribution_normalized.png"))
    plot_prediction_distribution(y_true_raw, y_pred_raw, _figure_path(figures_dir, "prediction_distribution_raw.png"))
    plot_hi_curve(result.hi, _figure_path(figures_dir, "health_indicator_curve.png"))
    if not skip_attention_export:
        plot_attention(attention_mean, _figure_path(figures_dir, "attention_heatmap_mean.png"))
        for idx, temporal_weights in enumerate(temporal_attention_examples, start=1):
            plot_attention_weights(
                np.asarray(temporal_weights, dtype=np.float32),
                _figure_path(figures_dir, f"temporal_attention_sample_{idx}.png"),
                title=f"Temporal Attention Weights Sample {idx}",
            )
    else:
        logger.info("Skipping attention export artifacts by configuration.")

    grouped = group_predictions_by_id(result.ids, y_true_raw, y_pred_raw, result.hi)
    hi_summary = {
        "hi_mean": float(np.mean(result.hi)),
        "hi_std": float(np.std(result.hi)),
        "hi_min": float(np.min(result.hi)),
        "hi_max": float(np.max(result.hi)),
    }
    corr = np.corrcoef(y_true, result.hi)[0, 1] if len(y_true) > 1 else np.nan
    corr_deg = np.corrcoef(1.0 - y_true, result.hi)[0, 1] if len(y_true) > 1 else np.nan
    corr_pred = np.corrcoef(y_pred, result.hi)[0, 1] if len(y_pred) > 1 else np.nan
    hi_summary["corr_hi_true_rul"] = float(corr) if np.isfinite(corr) else float("nan")
    hi_summary["corr_hi_true_degradation"] = float(corr_deg) if np.isfinite(corr_deg) else float("nan")
    hi_summary["corr_hi_pred_rul"] = float(corr_pred) if np.isfinite(corr_pred) else float("nan")

    scatter_rows = []
    for i, (bid, yt, yp, yp_cal, hi) in enumerate(zip(result.ids, y_true_raw, y_pred_raw, y_pred_raw_cal, result.hi)):
        scatter_rows.append(
            {
                "sample_index": i,
                "group": bid,
                "true_rul_raw": float(yt),
                "pred_rul_raw": float(yp),
                "pred_rul_raw_calibrated": float(yp_cal),
                "error_raw": float(yp - yt),
                "error_raw_calibrated": float(yp_cal - yt),
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
            "normalized_calibrated": norm_metrics_cal,
            "raw": raw_metrics,
            "raw_calibrated": raw_metrics_cal,
            "calibration": calibration,
        },
    )
    save_csv_rows(tables_dir / "grouped_metrics.csv", grouped_rows)
    save_csv_rows(tables_dir / "prediction_scatter_raw.csv", scatter_rows)
    save_json(tables_dir / "hi_summary.json", hi_summary)

    logger.info("TEST RESULTS (%s, normalized scale)", args.split.upper())
    logger.info(
        "Original | RMSE %.6f | MAE %.6f | R2 %.6f | Bias %+.6f | Pred/True Std Ratio %.6f",
        norm_metrics["rmse"],
        norm_metrics["mae"],
        norm_metrics["r2"],
        norm_metrics["mean_bias"],
        norm_metrics["pred_true_std_ratio"],
    )
    logger.info(
        "Calibrated | RMSE %.6f | MAE %.6f | R2 %.6f | Bias %+.6f | Pred/True Std Ratio %.6f",
        norm_metrics_cal["rmse"],
        norm_metrics_cal["mae"],
        norm_metrics_cal["r2"],
        norm_metrics_cal["mean_bias"],
        norm_metrics_cal["pred_true_std_ratio"],
    )
    logger.info("Evaluation normalized metrics: %s", norm_metrics)
    logger.info("Evaluation normalized metrics (calibrated): %s", norm_metrics_cal)
    logger.info("Evaluation raw metrics: %s", raw_metrics)
    logger.info("Evaluation raw metrics (calibrated): %s", raw_metrics_cal)
    logger.info("HI summary: %s", hi_summary)
    logger.info("Saved evaluation artifacts under %s", eval_dir)


if __name__ == "__main__":
    main()
