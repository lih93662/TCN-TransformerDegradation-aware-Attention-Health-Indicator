"""Training entrypoint for PHM2012 RUL prediction project.

Usage:
    python scripts/train.py --config configs/config_loss.yaml
    python scripts/train.py --config configs/config_loss.yaml --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig, count_parameters
from src.checkpoint_compat import resolve_rul_head_mode
from src.preprocess import prepare_datasets, summarize_dataset
from src.trainer import Trainer, TrainerConfig
from src.utils import (
    build_run_name,
    configure_logging,
    ensure_project_paths,
    get_device,
    load_yaml,
    save_csv_rows,
    save_json,
    set_seed,
)
from src.visualization import plot_training_loss, plot_training_metrics


VARIANT_LABELS = {
    "ablation_backbone": "A",
    "ablation_attention": "B",
    "ablation_loss": "C",
    "ablation_full": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid PHM2012 RUL model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config_loss.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional seed override for repeated experiments.",
    )
    return parser.parse_args()


def build_model_config(cfg: Dict, sensor_dim: int) -> ModelConfig:
    model_cfg_raw = cfg["model"]
    return ModelConfig(
        sensor_dim=sensor_dim,
        tcn_channels=int(model_cfg_raw["tcn_channels"]),
        tcn_kernel_size=int(model_cfg_raw["tcn_kernel_size"]),
        tcn_dilations=tuple(model_cfg_raw["tcn_dilations"]),
        transformer_embed_dim=int(model_cfg_raw["transformer_embed_dim"]),
        transformer_heads=int(model_cfg_raw["transformer_heads"]),
        transformer_layers=int(model_cfg_raw["transformer_layers"]),
        transformer_ffn_dim=int(model_cfg_raw["transformer_ffn_dim"]),
        dropout=float(model_cfg_raw["dropout"]),
        backbone_variant=str(model_cfg_raw.get("backbone_variant", "tcn_transformer")),
        use_hi=bool(model_cfg_raw.get("use_hi", True)),
        use_fixed_hi=bool(model_cfg_raw.get("use_fixed_hi", False)),
        hi_input_source=str(model_cfg_raw.get("hi_input_source", "tcn")),
        hi_output_temperature=float(model_cfg_raw.get("hi_output_temperature", 1.2)),
        use_attention=bool(model_cfg_raw.get("use_attention", True)),
        attention_use_hi_bias=bool(model_cfg_raw.get("attention_use_hi_bias", True)),
        attention_use_hi_logit=bool(model_cfg_raw.get("attention_use_hi_logit", True)),
        attention_use_temporal_gate=bool(model_cfg_raw.get("attention_use_temporal_gate", True)),
        attention_use_recency_bias=bool(model_cfg_raw.get("attention_use_recency_bias", True)),
        attention_conditioning_gain=float(model_cfg_raw.get("attention_conditioning_gain", 2.0)),
        attention_head_bias_gain=float(model_cfg_raw.get("attention_head_bias_gain", 2.0)),
        attention_temperature=float(model_cfg_raw.get("attention_temperature", 1.0)),
        attention_recency_strength=float(model_cfg_raw.get("attention_recency_strength", 0.5)),
        head_hidden_dim=int(model_cfg_raw.get("head_hidden_dim", 32)),
        use_hi_in_rul_head=bool(model_cfg_raw.get("use_hi_in_rul_head", False)),
        rul_head_mode=resolve_rul_head_mode(model_cfg_raw, default="hi_guided_residual"),
        hi_residual_scale=float(model_cfg_raw.get("hi_residual_scale", 0.3)),
        output_activation=str(model_cfg_raw.get("output_activation", "identity")),
    )


def build_trainer_config(cfg: Dict) -> TrainerConfig:
    train_cfg_raw = cfg["train"]
    use_mae_term = train_cfg_raw.get("use_mae_term")
    use_bias_regularization = train_cfg_raw.get("use_bias_regularization")
    mae_weight = float(train_cfg_raw.get("loss_mae_weight", 0.3))
    bias_weight = float(train_cfg_raw.get("bias_regularization_weight", 0.0))
    loss_name = str(train_cfg_raw.get("loss_name", "mse"))
    if use_mae_term is not None:
        use_mae_term = bool(use_mae_term)
        loss_name = "mse_mae" if use_mae_term else "mse"
        if not use_mae_term:
            mae_weight = 0.0
    if use_bias_regularization is not None:
        use_bias_regularization = bool(use_bias_regularization)
        if not use_bias_regularization:
            bias_weight = 0.0
    return TrainerConfig(
        lr=float(train_cfg_raw["lr"]),
        batch_size=int(train_cfg_raw["batch_size"]),
        epochs=int(train_cfg_raw["epochs"]),
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
        early_stopping_patience=int(train_cfg_raw.get("early_stopping_patience", 5)),
        early_stopping_min_epochs=int(train_cfg_raw.get("early_stopping_min_epochs", 10)),
        early_stopping_min_delta=float(train_cfg_raw.get("early_stopping_min_delta", 0.0)),
        grad_clip_norm=float(train_cfg_raw.get("grad_clip_norm", 0.0)),
        scheduler_factor=float(train_cfg_raw.get("scheduler_factor", 0.5)),
        scheduler_patience=int(train_cfg_raw.get("scheduler_patience", 3)),
        weight_decay=float(train_cfg_raw.get("weight_decay", 0.0)),
        loss_name=loss_name,
        loss_mse_weight=float(train_cfg_raw.get("loss_mse_weight", 1.0)),
        loss_mae_weight=mae_weight,
        huber_delta=float(train_cfg_raw.get("huber_delta", 0.1)),
        bias_regularization_weight=bias_weight,
        std_regularization_weight=float(train_cfg_raw.get("std_regularization_weight", 0.0)),
        correlation_regularization_weight=float(train_cfg_raw.get("correlation_regularization_weight", 0.0)),
        hi_supervision_weight=float(train_cfg_raw.get("hi_supervision_weight", 0.0)),
        hi_rank_weight=float(train_cfg_raw.get("hi_rank_weight", 0.0)),
        hi_variance_weight=float(train_cfg_raw.get("hi_variance_weight", 0.03)),
        hi_variance_floor=float(train_cfg_raw.get("hi_variance_floor", 0.08)),
        hi_smoothness_weight=float(train_cfg_raw.get("hi_smoothness_weight", 0.01)),
        residual_regularization_weight=float(train_cfg_raw.get("residual_regularization_weight", 0.0)),
        hi_supervision_mode=str(train_cfg_raw.get("hi_supervision_mode", "weak")),
        hi_target_mode=str(train_cfg_raw.get("hi_target_mode", "degradation")),
    )


def run_single_seed(cfg: Dict, seed: int) -> Dict[str, float | int | str]:
    exp_cfg = cfg["experiment"]
    run_name = build_run_name(
        base_name=str(exp_cfg.get("name", "rul_experiment")),
        seed=seed,
        suffix=str(exp_cfg.get("tag", "")) or None,
    )
    project_paths = ensure_project_paths(ROOT, run_name=run_name)
    logger = configure_logging(project_paths["run_logs"] / "train.log")

    set_seed(seed)
    variant_tag = str(exp_cfg.get("tag", ""))
    logger.info("Run name: %s", run_name)
    logger.info("Experiment variant: %s (%s)", VARIANT_LABELS.get(variant_tag, "main"), variant_tag or "default")
    logger.info("Reproducibility seed: %d", seed)
    logger.info("Preparing datasets from PHM2012 files...")
    data_cfg = cfg["data"]
    prepared = prepare_datasets(
        dataset_root=data_cfg["dataset_root"],
        window_size=int(data_cfg["window_size"]),
        stride=int(data_cfg["stride"]),
        max_rul=int(data_cfg["max_rul"]),
        valid_ratio=float(data_cfg["valid_ratio"]),
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
    stats = summarize_dataset(prepared)
    logger.info("Dataset summary: %s", stats)

    model_cfg = build_model_config(cfg, sensor_dim=prepared.feature_dim)
    model = HybridRULModel(model_cfg)
    n_params = count_parameters(model)
    logger.info("Model parameter count: %d", n_params)

    trainer_cfg = build_trainer_config(cfg)
    device = get_device()
    logger.info("Using device: %s", device)

    checkpoint_path = project_paths["run_checkpoints"] / "best_model.pth"
    trainer = Trainer(
        model=model,
        config=trainer_cfg,
        device=device,
        checkpoint_path=checkpoint_path,
        logger=logger,
        logs_dir=project_paths["run_logs"],
        figures_dir=project_paths["run_figures"],
    )

    history = trainer.fit(prepared.train_dataset, prepared.valid_dataset)
    metrics = trainer.evaluate(prepared.test_dataset)

    plot_training_loss(history, project_paths["run_figures"] / "training_loss_curve.png")
    plot_training_metrics(history, project_paths["run_figures"] / "training_metrics_curve.png")

    save_json(project_paths["run_logs"] / "history.json", {"history": history})
    save_json(project_paths["run_logs"] / "dataset_summary.json", stats)
    save_json(project_paths["run_logs"] / "resolved_config.json", cfg)
    save_json(project_paths["run_results"] / "test_metrics.json", metrics)
    save_json(
        project_paths["run_results"] / "run_metadata.json",
        {
            "run_name": run_name,
            "seed": seed,
            "num_parameters": n_params,
            "target_scale": prepared.target_scale,
            "model_config": model_cfg.__dict__,
            "trainer_config": trainer_cfg.__dict__,
        },
    )
    logger.info("Training complete for %s. Artifacts written under %s", run_name, project_paths["run_dir"])

    return {
        "run_name": run_name,
        "seed": seed,
        "num_parameters": n_params,
        **metrics,
    }


def summarize_seed_results(records: List[Dict[str, float | int | str]]) -> Dict[str, Dict[str, float]]:
    metric_names = ["rmse", "mae", "r2", "mean_bias", "phm_score"]
    summary: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        values = [float(r[metric]) for r in records]
        summary[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    config_seeds = cfg.get("experiment", {}).get("seeds")
    if args.seeds:
        seeds = list(args.seeds)
    elif config_seeds:
        seeds = [int(s) for s in config_seeds]
    else:
        seeds = [int(cfg["experiment"].get("seed", 42))]

    all_records: List[Dict[str, float | int | str]] = []
    shared_paths = ensure_project_paths(ROOT)
    summary_logger = configure_logging(shared_paths["logs"] / "train_summary.log")
    summary_logger.info("Running seeds: %s", seeds)

    for seed in seeds:
        record = run_single_seed(cfg, seed)
        all_records.append(record)
        summary_logger.info("Completed seed %s with metrics: %s", seed, record)

    save_csv_rows(shared_paths["results"] / "multi_seed_results.csv", all_records)
    aggregated = summarize_seed_results(all_records)
    save_json(shared_paths["results"] / "multi_seed_summary.json", aggregated)
    summary_logger.info("Aggregate summary: %s", aggregated)


if __name__ == "__main__":
    main()
