"""Evaluation entrypoint for saved hybrid RUL model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_model, group_predictions_by_id
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.preprocess import prepare_datasets
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid PHM2012 RUL model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
        help="Checkpoint file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)
    logger = configure_logging(paths["logs"] / "evaluate.log")

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    prepared = prepare_datasets(
        dataset_root=data_cfg["dataset_root"],
        window_size=int(data_cfg["window_size"]),
        stride=int(data_cfg["stride"]),
        max_rul=int(data_cfg["max_rul"]),
        valid_ratio=float(data_cfg["valid_ratio"]),
        seed=seed,
    )

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
    device = get_device()
    model = model.to(device)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state"])
    logger.info("Loaded checkpoint from epoch %s", payload.get("epoch", "unknown"))

    loader = DataLoader(prepared.test_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    result = evaluate_model(model, loader, device)

    grouped = group_predictions_by_id(
        ids=result.ids,
        y_true=result.y_true,
        y_pred=result.y_pred,
        hi=result.hi,
    )

    summary = {
        "rmse": result.rmse,
        "mae": result.mae,
        "phm_score": result.phm_score,
        "num_samples": len(result.y_true),
        "num_bearings": len(grouped),
    }
    save_json(paths["logs"] / "evaluation_metrics.json", summary)
    logger.info("Evaluation summary: %s", summary)


if __name__ == "__main__":
    main()
