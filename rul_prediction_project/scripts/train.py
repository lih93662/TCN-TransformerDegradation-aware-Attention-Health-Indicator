"""Training entrypoint for PHM2012 RUL prediction project.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig, count_parameters
from src.preprocess import prepare_datasets, summarize_dataset
from src.trainer import Trainer, TrainerConfig
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, save_json, set_seed
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


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    project_paths = ensure_project_paths(ROOT)
    logger = configure_logging(project_paths["logs"] / "train.log")

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    logger.info("Preparing datasets from PHM2012 files...")
    data_cfg = cfg["data"]
    prepared = prepare_datasets(
        dataset_root=data_cfg["dataset_root"],
        window_size=int(data_cfg["window_size"]),
        stride=int(data_cfg["stride"]),
        max_rul=int(data_cfg["max_rul"]),
        valid_ratio=float(data_cfg["valid_ratio"]),
        seed=seed,
    )
    stats = summarize_dataset(prepared)
    logger.info("Dataset summary: %s", stats)

    model_cfg_raw = cfg["model"]
    model_cfg = ModelConfig(
        sensor_dim=prepared.feature_dim,
        tcn_channels=int(model_cfg_raw["tcn_channels"]),
        tcn_kernel_size=int(model_cfg_raw["tcn_kernel_size"]),
        tcn_dilations=tuple(model_cfg_raw["tcn_dilations"]),
        transformer_embed_dim=int(model_cfg_raw["transformer_embed_dim"]),
        transformer_heads=int(model_cfg_raw["transformer_heads"]),
        transformer_layers=int(model_cfg_raw["transformer_layers"]),
        transformer_ffn_dim=int(model_cfg_raw["transformer_ffn_dim"]),
        dropout=float(model_cfg_raw["dropout"]),
    )
    model = HybridRULModel(model_cfg)

    n_params = count_parameters(model)
    logger.info("Model parameter count: %d", n_params)

    train_cfg_raw = cfg["train"]
    trainer_cfg = TrainerConfig(
        lr=float(train_cfg_raw["lr"]),
        batch_size=int(train_cfg_raw["batch_size"]),
        epochs=int(train_cfg_raw["epochs"]),
        num_workers=int(train_cfg_raw.get("num_workers", 0)),
    )

    device = get_device()
    logger.info("Using device: %s", device)

    trainer = Trainer(
        model=model,
        config=trainer_cfg,
        device=device,
        checkpoint_path=project_paths["checkpoints"] / "best_model.pth",
        logger=logger,
    )

    history = trainer.fit(prepared.train_dataset, prepared.valid_dataset)
    metrics = trainer.evaluate(prepared.test_dataset)

    plot_training_loss(
        history,
        project_paths["figures"] / "training_loss_curve.png",
    )

    save_json(project_paths["logs"] / "history.json", {"history": history})
    save_json(project_paths["logs"] / "test_metrics.json", metrics)
    logger.info("Training complete. Artifacts written under outputs/.")


if __name__ == "__main__":
    main()
