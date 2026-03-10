"""Evaluation entrypoint for saved hybrid PHM2012 RUL model.

This script performs three responsibilities:
1. Load prepared PHM2012 test windows and checkpoint.
2. Run inference and compute RMSE/MAE/PHM score.
3. Automatically generate visualization artifacts:
   - outputs/rul_curve.png
   - outputs/hi_curve.png
   - outputs/attention_heatmap.png
"""

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

from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.loss import compute_regression_metrics, phm2012_score
from src.preprocess import prepare_datasets
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, save_json, set_seed
from src.visualization import plot_attention, plot_hi_curve, plot_rul_curve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """

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


def _build_model(cfg: Dict, sensor_dim: int, device: torch.device) -> HybridRULModel:
    """Construct model from configuration.

    Args:
        cfg: Full config dictionary.
        sensor_dim: Input sensor channel count.
        device: Torch device.

    Returns:
        Instantiated model moved to ``device``.
    """

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
    )
    return HybridRULModel(model_cfg).to(device)


def _run_inference_with_attention(
    model: HybridRULModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and aggregate outputs needed for metrics and plotting.

    Args:
        model: Loaded hybrid model.
        loader: Test dataloader.
        device: Torch device.

    Returns:
        Tuple ``(y_true, y_pred, hi, attention_avg)`` where
        - ``y_true`` shape: ``(N,)``
        - ``y_pred`` shape: ``(N,)``
        - ``hi`` shape: ``(N,)``
        - ``attention_avg`` shape: ``(T,T)``

    Notes:
        - All tensors are moved to CPU numpy arrays before plotting.
        - Attention map is averaged over batch and heads across all batches.
    """

    model.eval()

    y_true_chunks: List[np.ndarray] = []
    y_pred_chunks: List[np.ndarray] = []
    hi_chunks: List[np.ndarray] = []

    attn_sum: torch.Tensor | None = None
    attn_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            out = model(x)
            pred = out["pred"]
            hi = out["hi"]
            attn = out["attn_map"]  # (B, H, T, T)

            y_true_chunks.append(y.detach().cpu().numpy().reshape(-1))
            y_pred_chunks.append(pred.detach().cpu().numpy().reshape(-1))
            hi_chunks.append(hi.detach().cpu().numpy().reshape(-1))

            # Average attention over batch and heads for a compact heatmap.
            attn_mean = attn.mean(dim=(0, 1)).detach().cpu()
            if attn_sum is None:
                attn_sum = attn_mean
            else:
                attn_sum = attn_sum + attn_mean
            attn_count += 1

    if not y_true_chunks:
        raise RuntimeError("No samples produced during evaluation.")

    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    hi_vals = np.concatenate(hi_chunks, axis=0)

    if attn_sum is None or attn_count == 0:
        attention_avg = np.eye(1, dtype=np.float32)
    else:
        attention_avg = (attn_sum / float(attn_count)).numpy().astype(np.float32)

    return y_true, y_pred, hi_vals, attention_avg


def main() -> None:
    """Main entrypoint for evaluation + visualization export."""

    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)
    logger = configure_logging(paths["logs"] / "evaluate.log")

    seed = int(cfg["experiment"].get("seed", 42))
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

    logger.info("Dataset summary during evaluation: %s", {
        "train_samples": len(prepared.train_dataset),
        "valid_samples": len(prepared.valid_dataset),
        "test_samples": len(prepared.test_dataset),
        "num_sensors": prepared.feature_dim,
        "detected_sensor_columns": prepared.detected_sensor_columns,
    })

    device = get_device()
    model = _build_model(cfg, sensor_dim=prepared.feature_dim, device=device)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state"])
    logger.info("Loaded checkpoint from epoch %s", payload.get("epoch", "unknown"))

    loader = DataLoader(
        prepared.test_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
    )

    y_true, y_pred, hi_vals, attn_avg = _run_inference_with_attention(model, loader, device)

    # Compute metrics in torch for consistency with existing project utilities.
    y_true_t = torch.from_numpy(y_true)
    y_pred_t = torch.from_numpy(y_pred)
    metrics = compute_regression_metrics(y_pred_t, y_true_t)
    metrics["phm_score"] = phm2012_score(y_pred_t, y_true_t)
    metrics["num_samples"] = int(len(y_true))

    save_json(paths["logs"] / "evaluation_metrics.json", metrics)
    logger.info("Evaluation summary: %s", metrics)

    # Required automatic figures.
    plot_rul_curve(y_true, y_pred, paths["outputs"] / "rul_curve.png")
    plot_hi_curve(hi_vals, paths["outputs"] / "hi_curve.png")
    plot_attention(attn_avg, paths["outputs"] / "attention_heatmap.png")
    logger.info("Saved figures: outputs/rul_curve.png, outputs/hi_curve.png, outputs/attention_heatmap.png")


if __name__ == "__main__":
    main()
