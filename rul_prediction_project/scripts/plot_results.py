"""Plotting script to generate publication-style figures from outputs/log files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_model, group_predictions_by_id
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.checkpoint_compat import detect_checkpoint_head_variant, load_model_state_strict, resolve_rul_head_mode
from src.preprocess import prepare_datasets
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, set_seed
from src.visualization import (
    plot_health_indicator_curve,
    plot_rul_prediction_curve,
    plot_single_bearing_prediction,
    plot_training_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument(
        "--history",
        type=str,
        default=str(ROOT / "outputs" / "logs" / "history.json"),
    )
    return parser.parse_args()


def _load_history(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("history", [])


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)
    logger = configure_logging(paths["logs"] / "plot_results.log")

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    history = _load_history(Path(args.history))
    if history:
        plot_training_loss(history, paths["figures"] / "training_loss_curve.png")
        logger.info("Saved training loss curve")

    prepared = prepare_datasets(
        dataset_root=cfg["data"]["dataset_root"],
        window_size=int(cfg["data"]["window_size"]),
        stride=int(cfg["data"]["stride"]),
        max_rul=int(cfg["data"]["max_rul"]),
        valid_ratio=float(cfg["data"]["valid_ratio"]),
        seed=seed,
    )

    ckpt = Path(args.checkpoint)
    payload = torch.load(ckpt, map_location=get_device()) if ckpt.exists() else None
    checkpoint_head_variant = (
        detect_checkpoint_head_variant(payload["model_state"])
        if payload is not None
        else None
    )

    model_cfg = ModelConfig(
        sensor_dim=prepared.feature_dim,
        tcn_channels=int(cfg["model"]["tcn_channels"]),
        tcn_kernel_size=int(cfg["model"]["tcn_kernel_size"]),
        tcn_dilations=tuple(cfg["model"]["tcn_dilations"]),
        transformer_embed_dim=int(cfg["model"]["transformer_embed_dim"]),
        transformer_heads=int(cfg["model"]["transformer_heads"]),
        transformer_layers=int(cfg["model"]["transformer_layers"]),
        transformer_ffn_dim=int(cfg["model"]["transformer_ffn_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        rul_head_mode=resolve_rul_head_mode(
            cfg["model"],
            default="feature_only",
            checkpoint_variant=checkpoint_head_variant,
        ),
        use_hi_auxiliary=bool(cfg["model"].get("use_hi_auxiliary", cfg["model"].get("use_hi", True))),
        use_hi_in_rul_head=bool(cfg["model"].get("use_hi_in_rul_head", False)),
        use_degradation_attention=bool(cfg["model"].get("use_degradation_attention", cfg["model"].get("use_attention", True))),
    )
    model = HybridRULModel(model_cfg)
    device = get_device()
    if payload is not None:
        if checkpoint_head_variant is not None:
            logger.info("Checkpoint head variant detected: %s", checkpoint_head_variant)
        load_model_state_strict(model, payload)
        logger.info("Loaded checkpoint for plotting")
    else:
        logger.warning("Checkpoint %s not found. Using randomly initialized weights.", ckpt)

    model = model.to(device)
    loader = DataLoader(prepared.test_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    result = evaluate_model(model, loader, device)

    plot_rul_prediction_curve(result.y_true, result.y_pred, paths["figures"] / "rul_prediction_curve.png")
    plot_health_indicator_curve(result.hi, paths["figures"] / "health_indicator_curve.png")

    grouped = group_predictions_by_id(result.ids, result.y_true, result.y_pred, result.hi)
    if grouped:
        first_id = sorted(grouped.keys())[0]
        plot_single_bearing_prediction(
            first_id,
            grouped[first_id]["y_true"],
            grouped[first_id]["y_pred"],
            paths["figures"] / "example_bearing_prediction.png",
        )
        logger.info("Saved single bearing plot for %s", first_id)

    logger.info("All figures generated under outputs/figures")


if __name__ == "__main__":
    main()
