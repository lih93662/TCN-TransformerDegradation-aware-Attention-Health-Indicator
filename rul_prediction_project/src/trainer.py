"""Training loop implementation for hybrid RUL model.

This module provides:
- GPU-aware training/validation loops.
- Best-checkpoint saving.
- Epoch-level structured logs.
- Early stopping based on validation RMSE to reduce overfitting.
- Per-batch loss CSV logging and epoch attention diagnostics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluator import evaluate_model
from .loss import CompositeRULLoss


@dataclass
class TrainerConfig:
    """Trainer hyperparameter container."""

    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 0
    early_stopping_patience: int = 5
    grad_clip_norm: float = 0.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3


class Trainer:
    """Encapsulated trainer with checkpointing and early stopping.

    Early stopping behavior:
    - Monitors ``valid_rmse``.
    - Stops if it does not improve for ``patience`` epochs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        device: torch.device,
        checkpoint_path: Path,
        logger,
        logs_dir: Path,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.logger = logger
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = CompositeRULLoss(rmse_weight=1.0, mae_weight=0.3)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=1e-6,
        )

        self.batch_loss_csv = self.logs_dir / "batch_loss_log.csv"
        self.attn_stats_csv = self.logs_dir / "attention_epoch_stats.csv"
        self.attn_dir = self.logs_dir / "attention_maps"
        self.attn_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, float]] = []
        self.best_val_total = float("inf")
        self.best_val_rmse = float("inf")
        self.no_improve_epochs = 0

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        """Create dataloader with consistent options."""

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    @staticmethod
    def _append_rows_csv(path: Path, rows: List[Dict[str, float]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float], np.ndarray | None]:
        """Run one train/validation epoch and return metrics + diagnostics."""

        mode = "train" if train else "valid"
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0
        batch_rows: List[Dict[str, float]] = []

        attn_min_sum = 0.0
        attn_max_sum = 0.0
        attn_mean_sum = 0.0
        attn_std_sum = 0.0
        attn_batches = 0
        attn_map_sum: np.ndarray | None = None

        pbar = tqdm(loader, desc=f"{mode.title()} Epoch {epoch:03d}", leave=False)
        grad_context = torch.enable_grad() if train else torch.no_grad()

        with grad_context:
            for batch_idx, batch in enumerate(pbar, start=1):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                out = self.model(x)
                loss_out = self.criterion(out["pred"], y)

                if train:
                    loss_out.total.backward()
                    if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.optimizer.step()

                total_loss += float(loss_out.total.item())
                total_rmse += float(loss_out.rmse.item())
                total_mae += float(loss_out.mae.item())

                attn = out.get("attn_map")
                if attn is not None:
                    attn_min = float(attn.min().item())
                    attn_max = float(attn.max().item())
                    attn_mean = float(attn.mean().item())
                    attn_std = float(attn.std().item())
                    attn_min_sum += attn_min
                    attn_max_sum += attn_max
                    attn_mean_sum += attn_mean
                    attn_std_sum += attn_std
                    attn_batches += 1

                    attn_2d = attn.detach().mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
                    attn_map_sum = attn_2d if attn_map_sum is None else (attn_map_sum + attn_2d)
                else:
                    attn_min = float("nan")
                    attn_max = float("nan")

                current_lr = self.optimizer.param_groups[0]["lr"]
                batch_rows.append(
                    {
                        "epoch": epoch,
                        "phase": mode,
                        "batch": batch_idx,
                        "total_loss": float(loss_out.total.item()),
                        "rmse": float(loss_out.rmse.item()),
                        "mae": float(loss_out.mae.item()),
                        "lr": float(current_lr),
                        "attn_min": attn_min,
                        "attn_max": attn_max,
                    }
                )

                pbar.set_postfix(
                    {
                        "total": f"{loss_out.total.item():.4f}",
                        "rmse": f"{loss_out.rmse.item():.4f}",
                        "mae": f"{loss_out.mae.item():.4f}",
                        "attn_min": f"{attn_min:.3f}" if np.isfinite(attn_min) else "nan",
                        "attn_max": f"{attn_max:.3f}" if np.isfinite(attn_max) else "nan",
                    }
                )

        n = max(1, len(loader))
        metrics = {
            f"{mode}_total": total_loss / n,
            f"{mode}_rmse": total_rmse / n,
            f"{mode}_mae": total_mae / n,
        }

        attn_stats = {
            "epoch": float(epoch),
            "phase": 0.0 if train else 1.0,
            "attn_min": attn_min_sum / max(1, attn_batches),
            "attn_max": attn_max_sum / max(1, attn_batches),
            "attn_mean": attn_mean_sum / max(1, attn_batches),
            "attn_std": attn_std_sum / max(1, attn_batches),
            "num_batches_with_attention": float(attn_batches),
        }

        attn_avg = None
        if attn_map_sum is not None and attn_batches > 0:
            attn_avg = (attn_map_sum / float(attn_batches)).astype(np.float32)

        return metrics, batch_rows, attn_stats, attn_avg

    def _save_checkpoint(self, epoch: int, valid_total: float, valid_rmse: float) -> None:
        """Persist best checkpoint payload."""

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_valid_total": valid_total,
            "best_valid_rmse": valid_rmse,
            "config": self.config.__dict__,
        }
        torch.save(payload, self.checkpoint_path)

    def fit(self, train_dataset, valid_dataset) -> List[Dict[str, float]]:
        """Execute training loop with early stopping and return history."""

        train_loader = self._make_loader(train_dataset, shuffle=True)
        valid_loader = self._make_loader(valid_dataset, shuffle=False)

        for epoch in range(1, self.config.epochs + 1):
            tr, tr_rows, tr_attn_stats, tr_attn_avg = self._run_epoch(train_loader, epoch, train=True)
            va, va_rows, va_attn_stats, va_attn_avg = self._run_epoch(valid_loader, epoch, train=False)

            self._append_rows_csv(self.batch_loss_csv, tr_rows + va_rows)
            self._append_rows_csv(
                self.attn_stats_csv,
                [
                    {
                        "epoch": epoch,
                        "phase": "train",
                        **{k: v for k, v in tr_attn_stats.items() if k not in {"epoch", "phase"}},
                    },
                    {
                        "epoch": epoch,
                        "phase": "valid",
                        **{k: v for k, v in va_attn_stats.items() if k not in {"epoch", "phase"}},
                    },
                ],
            )

            if tr_attn_avg is not None:
                np.save(self.attn_dir / f"epoch_{epoch:03d}_train.npy", tr_attn_avg)
            if va_attn_avg is not None:
                np.save(self.attn_dir / f"epoch_{epoch:03d}_valid.npy", va_attn_avg)

            epoch_row = {**tr, **va, "epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(epoch_row)

            self.logger.info(
                (
                    "Epoch %03d | lr %.6f | train_total %.4f train_rmse %.4f train_mae %.4f | "
                    "valid_total %.4f valid_rmse %.4f valid_mae %.4f"
                ),
                epoch,
                self.optimizer.param_groups[0]["lr"],
                tr["train_total"],
                tr["train_rmse"],
                tr["train_mae"],
                va["valid_total"],
                va["valid_rmse"],
                va["valid_mae"],
            )
            self.logger.info(
                "Epoch %03d attention stats | train min/max %.5f/%.5f | valid min/max %.5f/%.5f",
                epoch,
                tr_attn_stats["attn_min"],
                tr_attn_stats["attn_max"],
                va_attn_stats["attn_min"],
                va_attn_stats["attn_max"],
            )

            self.scheduler.step(va["valid_total"])

            if va["valid_total"] < self.best_val_total:
                self.best_val_total = va["valid_total"]
                self._save_checkpoint(epoch, self.best_val_total, va["valid_rmse"])
                self.logger.info(
                    "Saved checkpoint at epoch %03d with valid_total=%.4f",
                    epoch,
                    self.best_val_total,
                )

            if va["valid_rmse"] < self.best_val_rmse:
                self.best_val_rmse = va["valid_rmse"]
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1
                self.logger.info(
                    "EarlyStopping monitor valid_rmse did not improve (%d/%d)",
                    self.no_improve_epochs,
                    self.config.early_stopping_patience,
                )

            if self.no_improve_epochs >= self.config.early_stopping_patience:
                self.logger.info(
                    "Early stopping triggered at epoch %03d (best valid_rmse=%.4f)",
                    epoch,
                    self.best_val_rmse,
                )
                break

        return self.history

    def evaluate(self, test_dataset) -> Dict[str, float]:
        """Evaluate best model checkpoint against test dataset."""

        if self.checkpoint_path.exists():
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(payload["model_state"])
            self.logger.info(
                "Loaded best checkpoint from epoch %s", payload.get("epoch", "<unknown>")
            )

        test_loader = self._make_loader(test_dataset, shuffle=False)
        result = evaluate_model(self.model, test_loader, self.device)
        metrics = {
            "rmse": result.rmse,
            "mae": result.mae,
            "phm_score": result.phm_score,
        }
        self.logger.info(
            "Test metrics | RMSE %.4f | MAE %.4f | PHM Score %.4f",
            result.rmse,
            result.mae,
            result.phm_score,
        )
        return metrics
