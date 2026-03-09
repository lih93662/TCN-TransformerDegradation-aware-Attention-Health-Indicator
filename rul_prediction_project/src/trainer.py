"""Training loop implementation for hybrid RUL model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluator import evaluate_model
from .loss import CompositeRULLoss


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 0


class Trainer:
    """Encapsulated trainer with logging/checkpointing support."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        device: torch.device,
        checkpoint_path: Path,
        logger,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.logger = logger

        self.criterion = CompositeRULLoss(rmse_weight=1.0, mae_weight=0.3)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

        self.history: List[Dict[str, float]] = []
        self.best_val = float("inf")

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", leave=False)
        for batch in pbar:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss_out = self.criterion(out["pred"], y)
            loss_out.total.backward()
            self.optimizer.step()

            total_loss += float(loss_out.total.item())
            total_rmse += float(loss_out.rmse.item())
            total_mae += float(loss_out.mae.item())
            pbar.set_postfix(
                {
                    "total": f"{loss_out.total.item():.4f}",
                    "rmse": f"{loss_out.rmse.item():.4f}",
                    "mae": f"{loss_out.mae.item():.4f}",
                }
            )

        n = max(1, len(loader))
        return {
            "train_total": total_loss / n,
            "train_rmse": total_rmse / n,
            "train_mae": total_mae / n,
        }

    def _validate_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0

        pbar = tqdm(loader, desc=f"Valid Epoch {epoch:03d}", leave=False)
        with torch.no_grad():
            for batch in pbar:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                out = self.model(x)
                loss_out = self.criterion(out["pred"], y)

                total_loss += float(loss_out.total.item())
                total_rmse += float(loss_out.rmse.item())
                total_mae += float(loss_out.mae.item())
                pbar.set_postfix(
                    {
                        "total": f"{loss_out.total.item():.4f}",
                        "rmse": f"{loss_out.rmse.item():.4f}",
                        "mae": f"{loss_out.mae.item():.4f}",
                    }
                )

        n = max(1, len(loader))
        return {
            "valid_total": total_loss / n,
            "valid_rmse": total_rmse / n,
            "valid_mae": total_mae / n,
        }

    def _save_checkpoint(self, epoch: int, valid_total: float) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_valid_total": valid_total,
            "config": self.config.__dict__,
        }
        torch.save(payload, self.checkpoint_path)

    def fit(self, train_dataset, valid_dataset) -> List[Dict[str, float]]:
        """Execute full training loop and return metric history."""

        train_loader = self._make_loader(train_dataset, shuffle=True)
        valid_loader = self._make_loader(valid_dataset, shuffle=False)

        for epoch in range(1, self.config.epochs + 1):
            tr = self._train_epoch(train_loader, epoch)
            va = self._validate_epoch(valid_loader, epoch)

            epoch_row = {**tr, **va, "epoch": epoch}
            self.history.append(epoch_row)

            self.logger.info(
                (
                    "Epoch %03d | train_total %.4f train_rmse %.4f train_mae %.4f | "
                    "valid_total %.4f valid_rmse %.4f valid_mae %.4f"
                ),
                epoch,
                tr["train_total"],
                tr["train_rmse"],
                tr["train_mae"],
                va["valid_total"],
                va["valid_rmse"],
                va["valid_mae"],
            )

            if va["valid_total"] < self.best_val:
                self.best_val = va["valid_total"]
                self._save_checkpoint(epoch, self.best_val)
                self.logger.info(
                    "Saved new best checkpoint at epoch %03d with valid_total=%.4f",
                    epoch,
                    self.best_val,
                )

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
