"""Lightweight synthetic smoke test for train/eval pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PHM2012RULDataset
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import configure_logging, ensure_project_paths, set_seed


def make_synth_dataset(n: int, window: int, sensors: int, seed: int, prefix: str) -> PHM2012RULDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, window, sensors)).astype(np.float32)
    trend = np.linspace(1.0, 0.0, window, dtype=np.float32).reshape(1, window, 1)
    weighted = (x * trend).mean(axis=(1, 2))
    y = (weighted - weighted.min()) / max(weighted.max() - weighted.min(), 1e-6)
    ids = [f"{prefix}_{i//8}" for i in range(n)]
    return PHM2012RULDataset(x, y.astype(np.float32), ids)


def main() -> None:
    set_seed(123)
    paths = ensure_project_paths(ROOT, run_name="smoke_train_eval")
    logger = configure_logging(paths["run_logs"] / "smoke.log")

    train_ds = make_synth_dataset(n=128, window=32, sensors=2, seed=1, prefix="train")
    valid_ds = make_synth_dataset(n=64, window=32, sensors=2, seed=2, prefix="valid")
    test_ds = make_synth_dataset(n=64, window=32, sensors=2, seed=3, prefix="test")

    model = HybridRULModel(
        ModelConfig(
            sensor_dim=2,
            tcn_channels=32,
            transformer_embed_dim=64,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ffn_dim=128,
            dropout=0.1,
            use_hi=True,
            use_attention=True,
        )
    )
    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            lr=1e-3,
            batch_size=16,
            epochs=2,
            early_stopping_patience=2,
            early_stopping_min_epochs=1,
            grad_clip_norm=1.0,
            loss_name="mse_mae",
            loss_mse_weight=1.0,
            loss_mae_weight=0.2,
            std_regularization_weight=0.02,
        ),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_path=paths["run_checkpoints"] / "best_model.pth",
        logger=logger,
        logs_dir=paths["run_logs"],
        figures_dir=paths["run_figures"],
    )

    trainer.fit(train_ds, valid_ds)
    metrics = trainer.evaluate(test_ds)
    logger.info("Smoke metrics: %s", metrics)
    print("smoke_ok", metrics["rmse"], metrics["mae"])


if __name__ == "__main__":
    main()
