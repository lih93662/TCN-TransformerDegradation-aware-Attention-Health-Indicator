"""Benchmark experiment runner for comparative RUL modeling.

Compared model families (as requested):
- TCN
- BiLSTM
- Transformer
- TCN+Transformer
- Proposed Model (Hybrid: TCN+Transformer+Degradation Attention+HI)

Outputs:
- outputs/results/benchmark_results.csv
- outputs/results/benchmark_results.json
- outputs/benchmark_results.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.loss import CompositeRULLoss
from src.preprocess import prepare_datasets
from src.tcn import TCNEncoder
from src.transformer_encoder import TransformerTemporalEncoder
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, save_json, set_seed


@dataclass
class BenchmarkRecord:
    """One benchmark result row."""

    model_name: str
    rmse: float
    mae: float
    phm_score: float
    params: int


def phm_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PHM challenge score."""

    d = pred.view(-1) - target.view(-1)
    neg = torch.exp(-d / 13.0) - 1.0
    pos = torch.exp(d / 10.0) - 1.0
    return float(torch.where(d < 0, neg, pos).sum().item())


class TCNModel(nn.Module):
    """TCN baseline regression model."""

    def __init__(self, sensor_dim: int, channels: int = 64):
        super().__init__()
        self.tcn = TCNEncoder(input_dim=sensor_dim, channels=channels, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.1)
        self.head = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, pooled = self.tcn(x)
        pred = self.head(pooled)
        return {"pred": pred, "hi": torch.sigmoid(pred * 0.0 + 0.5)}


class BiLSTMModel(nn.Module):
    """BiLSTM baseline regression model."""

    def __init__(self, sensor_dim: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.LSTM(sensor_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, _ = self.rnn(x)
        pooled = out.mean(dim=1)
        pred = self.head(pooled)
        return {"pred": pred, "hi": torch.sigmoid(pred * 0.0 + 0.5)}


class TransformerModel(nn.Module):
    """Transformer-only baseline."""

    def __init__(self, sensor_dim: int, embed: int = 128):
        super().__init__()
        self.encoder = TransformerTemporalEncoder(input_dim=sensor_dim, embedding_dim=embed, heads=4, layers=2, ffn_dim=256, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(embed, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, pooled = self.encoder(x)
        pred = self.head(pooled)
        return {"pred": pred, "hi": torch.sigmoid(pred * 0.0 + 0.5)}


class TCNTransformerModel(nn.Module):
    """Combined TCN+Transformer baseline."""

    def __init__(self, sensor_dim: int, channels: int = 64, embed: int = 128):
        super().__init__()
        self.tcn = TCNEncoder(input_dim=sensor_dim, channels=channels, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.1)
        self.tr = TransformerTemporalEncoder(input_dim=channels, embedding_dim=embed, heads=4, layers=2, ffn_dim=256, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(channels + embed, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tcn_seq, tcn_pool = self.tcn(x)
        _, tr_pool = self.tr(tcn_seq)
        pred = self.head(torch.cat([tcn_pool, tr_pool], dim=-1))
        return {"pred": pred, "hi": torch.sigmoid(pred * 0.0 + 0.5)}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Compute RMSE/MAE/PHM on a dataloader."""

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            p = model(x)["pred"]
            preds.append(p)
            trues.append(y)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    rmse = float(torch.sqrt(torch.mean((pred - true) ** 2)).item())
    mae = float(torch.mean(torch.abs(pred - true)).item())
    return rmse, mae, phm_score(pred, true)


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
    """Train model with best-valid-loss selection."""

    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    crit = CompositeRULLoss(rmse_weight=1.0, mae_weight=0.3)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{model_name} Train {epoch:03d}", leave=False)
        tr = 0.0
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out["pred"], y).total
            loss.backward()
            opt.step()
            tr += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                va += float(crit(model(x)["pred"], y).total.item())

        tr /= max(1, len(train_loader))
        va /= max(1, len(valid_loader))
        logger.info("%s | epoch %03d | train %.4f | valid %.4f", model_name, epoch, tr, va)

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_model_variants(sensor_dim: int, cfg: Dict) -> List[Tuple[str, nn.Module]]:
    """Build required benchmark model list in requested order."""

    m = cfg["model"]
    proposed = HybridRULModel(
        ModelConfig(
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
    )

    return [
        ("TCN", TCNModel(sensor_dim=sensor_dim, channels=int(m["tcn_channels"]))),
        ("BiLSTM", BiLSTMModel(sensor_dim=sensor_dim, hidden=64)),
        ("Transformer", TransformerModel(sensor_dim=sensor_dim, embed=int(m["transformer_embed_dim"]))),
        (
            "TCN+Transformer",
            TCNTransformerModel(sensor_dim=sensor_dim, channels=int(m["tcn_channels"]), embed=int(m["transformer_embed_dim"])),
        ),
        ("Proposed Model", proposed),
    ]


def save_benchmark_results(path: Path, records: Sequence[BenchmarkRecord]) -> None:
    """Write benchmark table to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "rmse", "mae", "phm_score", "params"])
        for r in records:
            w.writerow([r.model_name, f"{r.rmse:.6f}", f"{r.mae:.6f}", f"{r.phm_score:.6f}", r.params])


def plot_benchmark_results(records: Sequence[BenchmarkRecord], out_path: Path) -> None:
    """Plot RMSE/MAE bar chart and save to outputs/benchmark_results.png."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = [r.model_name for r in records]
    rmse = np.asarray([r.rmse for r in records], dtype=np.float32)
    mae = np.asarray([r.mae for r in records], dtype=np.float32)

    x = np.arange(len(models))
    width = 0.36

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width / 2, rmse, width=width, label="RMSE", color="#2563eb")
    plt.bar(x + width / 2, mae, width=width, label="MAE", color="#dc2626")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Metric Value")
    plt.title("Benchmark Comparison (RMSE / MAE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark experiments")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)
    logger = configure_logging(paths["logs"] / "benchmark.log")

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    prepared = prepare_datasets(
        dataset_root=cfg["data"]["dataset_root"],
        window_size=int(cfg["data"].get("window_size", 40)),
        stride=int(cfg["data"].get("stride", 10)),
        max_rul=int(cfg["data"].get("max_rul", 125)),
        valid_ratio=float(cfg["data"].get("valid_ratio", 0.2)),
        seed=seed,
        sensor_dim=int(cfg["data"].get("sensor_dim", 2)),
        max_windows_per_bearing=int(cfg["data"].get("max_windows_per_bearing", 20000)),
    )

    batch_size = int(args.batch_size or cfg["train"]["batch_size"])
    epochs = int(args.epochs or cfg["train"]["epochs"])
    lr = float(args.lr or cfg["train"]["lr"])

    train_loader = DataLoader(prepared.train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(prepared.valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(prepared.test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()
    logger.info("Benchmark setup | device=%s batch=%d epochs=%d lr=%g", device, batch_size, epochs, lr)

    variants = build_model_variants(prepared.feature_dim, cfg)
    results: List[BenchmarkRecord] = []

    for name, model in variants:
        logger.info("Running benchmark model: %s", name)
        model = train_one_model(model, train_loader, valid_loader, epochs, lr, device, logger, name)
        rmse, mae, phm = evaluate_model_metrics(model, test_loader, device)
        rec = BenchmarkRecord(model_name=name, rmse=rmse, mae=mae, phm_score=phm, params=count_params(model))
        results.append(rec)
        logger.info("%s | RMSE %.4f | MAE %.4f | PHM %.4f", name, rmse, mae, phm)

    csv_path = paths["outputs"] / "results" / "benchmark_results.csv"
    save_benchmark_results(csv_path, results)
    save_json(paths["outputs"] / "results" / "benchmark_results.json", {"results": [r.__dict__ for r in results]})

    fig_path = paths["outputs"] / "benchmark_results.png"
    plot_benchmark_results(results, fig_path)
    logger.info("Saved benchmark table to %s", csv_path)
    logger.info("Saved benchmark figure to %s", fig_path)


if __name__ == "__main__":
    main()
