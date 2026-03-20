"""Training entrypoint for PHM2012 RUL prediction project."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig, count_parameters
from src.preprocess import prepare_datasets, summarize_dataset
from src.trainer import Trainer, TrainerConfig
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
from src.visualization import plot_training_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid PHM2012 RUL model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file.",
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
        attention_mode=str(model_cfg_raw.get("attention_mode", "degradation")),
        attention_temperature=float(model_cfg_raw.get("attention_temperature", 1.0)),
        attention_use_qk_norm=bool(model_cfg_raw.get("attention_use_qk_norm", False)),
        attention_bias_scale=float(model_cfg_raw.get("attention_bias_scale", 1.0)),
        prediction_activation=str(model_cfg_raw.get("prediction_activation", "identity")),
    )


def build_trainer_config(cfg: Dict) -> TrainerConfig:
    train_cfg_raw = cfg["train"]
    return TrainerConfig(
        lr=float(train_cfg_raw["lr"]),
        batch_size=int(train_cfg_raw["batch_size"]),
        epochs=int(train_cfg_raw["epochs"]),
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
        early_stopping_patience=int(train_cfg_raw.get("early_stopping_patience", 5)),
        grad_clip_norm=float(train_cfg_raw.get("grad_clip_norm", 0.0)),
        scheduler_factor=float(train_cfg_raw.get("scheduler_factor", 0.5)),
        scheduler_patience=int(train_cfg_raw.get("scheduler_patience", 3)),
        scheduler_min_lr=float(train_cfg_raw.get("scheduler_min_lr", 1e-6)),
        weight_decay=float(train_cfg_raw.get("weight_decay", 0.0)),
        loss_name=str(train_cfg_raw.get("loss_name", "mse")),
        loss_mse_weight=float(train_cfg_raw.get("loss_mse_weight", 1.0)),
        loss_mae_weight=float(train_cfg_raw.get("loss_mae_weight", 0.3)),
        loss_huber_delta=float(train_cfg_raw.get("loss_huber_delta", 1.0)),
        bias_regularization_weight=float(train_cfg_raw.get("bias_regularization_weight", 0.0)),
    )


def run_single_seed(cfg: Dict, seed: int, run_name: str, multi_seed: bool) -> Dict[str, float]:
    exp_cfg = cfg.get("experiment", {})
    use_run_name_dir = bool(exp_cfg.get("use_run_name_dir", multi_seed))
    project_paths = ensure_project_paths(ROOT, run_name if use_run_name_dir else None)
    logger = configure_logging(project_paths["logs"] / "train.log")
    set_seed(seed)

    logger.info("Starting run '%s' with seed=%d", run_name, seed)
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
    )
    stats = summarize_dataset(prepared)
    logger.info("Dataset summary: %s", stats)

    model_cfg = build_model_config(cfg, sensor_dim=prepared.feature_dim)
    model = HybridRULModel(model_cfg)
    logger.info("Model parameter count: %d", count_parameters(model))

    trainer_cfg = build_trainer_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    trainer = Trainer(
        model=model,
        config=trainer_cfg,
        device=device,
        checkpoint_path=project_paths["checkpoints"] / "best_model.pth",
        logger=logger,
        logs_dir=project_paths["logs"],
    )

    history = trainer.fit(prepared.train_dataset, prepared.valid_dataset)
    metrics = trainer.evaluate(prepared.test_dataset)

    plot_training_loss(history, project_paths["figures"] / "training_loss_curve.png")
    save_json(project_paths["logs"] / "history.json", {"history": history})
    save_json(project_paths["logs"] / "test_metrics.json", metrics)
    save_json(project_paths["logs"] / "run_metadata.json", {"seed": seed, "run_name": run_name, "dataset": stats})

    result = {
        "seed": seed,
        "run_name": run_name,
        **metrics,
        "best_epoch": trainer.best_epoch,
        "best_valid_rmse": trainer.best_val_rmse,
        "output_dir": str(project_paths["outputs"]),
    }
    logger.info("Training complete for '%s'. Artifacts written under %s", run_name, project_paths["outputs"])
    return result


def summarize_multi_seed(results: List[Dict[str, float]], summary_path: Path) -> None:
    numeric_keys = ["rmse", "mae", "r2", "phm_score", "pred_mean", "pred_std", "true_mean", "true_std", "mean_error", "best_valid_rmse"]
    summary = {
        "num_runs": len(results),
        "runs": results,
        "aggregate": {},
    }
    for key in numeric_keys:
        values = [float(r[key]) for r in results if key in r]
        if not values:
            continue
        summary["aggregate"][key] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values) if len(values) > 1 else 0.0),
        }
    save_json(summary_path, summary)
    save_csv_rows(
        summary_path.with_suffix(".csv"),
        [
            {
                "metric": metric,
                "mean": payload["mean"],
                "std": payload["std"],
            }
            for metric, payload in summary["aggregate"].items()
        ],
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    exp_cfg = cfg.get("experiment", {})
    seeds = resolve_seeds(exp_cfg)
    base_name = str(exp_cfg.get("name", "tcn_transformer_degradation_attention_hi"))
    multi_seed = len(seeds) > 1

    results: List[Dict[str, float]] = []
    for seed in seeds:
        run_name = experiment_run_name(base_name, seed, multi_seed=multi_seed)
        results.append(run_single_seed(cfg, seed=seed, run_name=run_name, multi_seed=multi_seed))

    if multi_seed:
        summary_paths = ensure_project_paths(ROOT, run_name=base_name)
        summarize_multi_seed(results, summary_paths["results"] / "multi_seed_summary.json")


if __name__ == "__main__":
    import torch

    main()
