"""Benchmark experiment runner for comparative RUL modeling.

This script trains/evaluates a set of model variants and exports a benchmark
result table to:

    outputs/results/benchmark_results.csv

Compared variants:
1. TCN
2. TCN + Transformer
3. TCN + Transformer + Degradation Attention
4. Full Hybrid Model (TCN + Transformer + Degradation Attention + HI)

The benchmark is intended for reproducible ablation studies in research papers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.degradation_attention import DegradationAwareAttention
from src.evaluator import evaluate_model
from src.health_indicator import HealthIndicatorNet
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.loss import CompositeRULLoss
from src.preprocess import prepare_datasets
from src.tcn import TCNEncoder
from src.transformer_encoder import TransformerTemporalEncoder
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, save_json, set_seed


@dataclass
class BenchmarkRecord:
    """One row in benchmark output table."""

    model_name: str
    rmse: float
    mae: float
    phm_score: float
    params: int


class TCNOnlyModel(nn.Module):
    """Baseline model with TCN encoder and simple regression head.

    Input shape:
        ``(batch, time, sensors)``

    Output:
        Dict with ``pred`` shape ``(batch,1)`` and ``hi`` placeholder.
    """

    def __init__(self, sensor_dim: int, channels: int = 64):
        super().__init__()
        self.tcn = TCNEncoder(
            input_dim=sensor_dim,
            channels=channels,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, pooled = self.tcn(x)
        pred = self.head(pooled)
        hi = torch.sigmoid(pred * 0.0 + 0.5)
        return {"pred": pred, "hi": hi}


class TCNTransformerModel(nn.Module):
    """Model variant combining TCN + Transformer without custom attention/HI."""

    def __init__(self, sensor_dim: int, channels: int = 64, embed: int = 128):
        super().__init__()
        self.tcn = TCNEncoder(
            input_dim=sensor_dim,
            channels=channels,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            dropout=0.1,
        )
        self.transformer = TransformerTemporalEncoder(
            input_dim=channels,
            embedding_dim=embed,
            heads=4,
            layers=2,
            ffn_dim=256,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(channels + embed, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tcn_seq, tcn_pool = self.tcn(x)
        _, tr_pool = self.transformer(tcn_seq)
        fused = torch.cat([tcn_pool, tr_pool], dim=-1)
        pred = self.head(fused)
        hi = torch.sigmoid(pred * 0.0 + 0.5)
        return {"pred": pred, "hi": hi}


class TCNTransformerAttentionModel(nn.Module):
    """Model variant adding degradation-aware attention but no explicit HI fusion."""

    def __init__(self, sensor_dim: int, channels: int = 64, embed: int = 128):
        super().__init__()
        self.tcn = TCNEncoder(
            input_dim=sensor_dim,
            channels=channels,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            dropout=0.1,
        )
        self.transformer = TransformerTemporalEncoder(
            input_dim=channels,
            embedding_dim=embed,
            heads=4,
            layers=2,
            ffn_dim=256,
            dropout=0.1,
        )
        self.hi_proxy = HealthIndicatorNet(sensor_dim=sensor_dim)
        self.attn = DegradationAwareAttention(embed_dim=embed, num_heads=4, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(channels + embed + embed, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tcn_seq, tcn_pool = self.tcn(x)
        tr_seq, tr_pool = self.transformer(tcn_seq)
        hi, _ = self.hi_proxy(x)
        da_feat, _ = self.attn(tr_seq, hi)
        pred = self.head(torch.cat([tcn_pool, tr_pool, da_feat], dim=-1))
        return {"pred": pred, "hi": hi}


def count_params(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger,
    model_name: str,
) -> nn.Module:
    """Train one benchmark model with shared protocol.

    Args:
        model: Model instance.
        train_loader: Training dataloader.
        valid_loader: Validation dataloader.
        epochs: Number of epochs.
        lr: Learning rate.
        device: Torch device.
        logger: Logger instance.
        model_name: Name for logs.

    Returns:
        Best model weights loaded in returned model.
    """

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CompositeRULLoss(rmse_weight=1.0, mae_weight=0.3)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0

        pbar = tqdm(train_loader, desc=f"{model_name} Train {epoch:03d}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out["pred"], y).total
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                out = model(x)
                loss = criterion(out["pred"], y).total
                va_loss += float(loss.item())

        tr_loss /= max(1, len(train_loader))
        va_loss /= max(1, len(valid_loader))

        logger.info("%s | epoch %03d | train %.4f | valid %.4f", model_name, epoch, tr_loss, va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_benchmark_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Evaluate benchmark model on test split.

    Returns:
        Tuple ``(rmse, mae, phm_score)``.
    """

    result = evaluate_model(model, test_loader, device)
    return result.rmse, result.mae, result.phm_score


def build_model_variants(sensor_dim: int, cfg: Dict) -> List[Tuple[str, nn.Module]]:
    """Construct all benchmarked model variants.

    Args:
        sensor_dim: Number of sensor channels.
        cfg: Full config dict.

    Returns:
        List of ``(name, model)`` tuples.
    """

    model_cfg = cfg["model"]

    full = HybridRULModel(
        ModelConfig(
            sensor_dim=sensor_dim,
            tcn_channels=int(model_cfg["tcn_channels"]),
            tcn_kernel_size=int(model_cfg["tcn_kernel_size"]),
            tcn_dilations=tuple(model_cfg["tcn_dilations"]),
            transformer_embed_dim=int(model_cfg["transformer_embed_dim"]),
            transformer_heads=int(model_cfg["transformer_heads"]),
            transformer_layers=int(model_cfg["transformer_layers"]),
            transformer_ffn_dim=int(model_cfg["transformer_ffn_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
    )

    return [
        ("TCN", TCNOnlyModel(sensor_dim=sensor_dim, channels=int(model_cfg["tcn_channels"]))),
        (
            "TCN+Transformer",
            TCNTransformerModel(
                sensor_dim=sensor_dim,
                channels=int(model_cfg["tcn_channels"]),
                embed=int(model_cfg["transformer_embed_dim"]),
            ),
        ),
        (
            "TCN+Transformer+Attention",
            TCNTransformerAttentionModel(
                sensor_dim=sensor_dim,
                channels=int(model_cfg["tcn_channels"]),
                embed=int(model_cfg["transformer_embed_dim"]),
            ),
        ),
        ("FullHybrid", full),
    ]


def save_benchmark_results(path: Path, records: Sequence[BenchmarkRecord]) -> None:
    """Save benchmark rows to CSV.

    Args:
        path: Output CSV path.
        records: Sequence of benchmark records.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "rmse", "mae", "phm_score", "params"])
        for r in records:
            writer.writerow([r.model_name, f"{r.rmse:.6f}", f"{r.mae:.6f}", f"{r.phm_score:.6f}", r.params])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run benchmark experiments")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def main() -> None:
    """Main benchmark workflow."""

    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)
    logger = configure_logging(paths["logs"] / "benchmark.log")

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    # Prepare data once to ensure fair comparison across all variants.
    prepared = prepare_datasets(
        dataset_root=cfg["data"]["dataset_root"],
        window_size=int(cfg["data"]["window_size"]),
        stride=int(cfg["data"]["stride"]),
        max_rul=int(cfg["data"]["max_rul"]),
        valid_ratio=float(cfg["data"]["valid_ratio"]),
        seed=seed,
    )

    batch_size = int(args.batch_size or cfg["train"]["batch_size"])
    epochs = int(args.epochs or cfg["train"]["epochs"])
    lr = float(args.lr or cfg["train"]["lr"])

    train_loader = DataLoader(prepared.train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(prepared.valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(prepared.test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()
    logger.info("Benchmark device: %s", device)
    logger.info("Benchmark setup: batch_size=%d epochs=%d lr=%g", batch_size, epochs, lr)

    variants = build_model_variants(prepared.feature_dim, cfg)

    results: List[BenchmarkRecord] = []
    for name, model in variants:
        logger.info("Running benchmark for model: %s", name)
        trained_model = train_one_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            logger=logger,
            model_name=name,
        )

        rmse, mae, phm = evaluate_benchmark_model(trained_model, test_loader, device)
        params = count_params(trained_model)

        record = BenchmarkRecord(model_name=name, rmse=rmse, mae=mae, phm_score=phm, params=params)
        results.append(record)
        logger.info("%s | RMSE %.4f | MAE %.4f | PHM %.4f | Params %d", name, rmse, mae, phm, params)

    csv_path = paths["outputs"] / "results" / "benchmark_results.csv"
    save_benchmark_results(csv_path, results)

    json_rows = [r.__dict__ for r in results]
    save_json(paths["outputs"] / "results" / "benchmark_results.json", {"results": json_rows})

    logger.info("Benchmark complete. Results saved to %s", csv_path)


if __name__ == "__main__":
    main()
