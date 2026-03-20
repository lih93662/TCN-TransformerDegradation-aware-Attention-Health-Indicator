"""Training loop implementation for hybrid RUL model."""

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
from .loss import RegressionLoss
from .visualization import plot_rul_curve


@dataclass
class TrainerConfig:
    """Trainer hyperparameter container."""

    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 0
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    weight_decay: float = 0.0
    loss_name: str = "mse"
    loss_mse_weight: float = 1.0
    loss_mae_weight: float = 0.3
    loss_huber_delta: float = 1.0
    bias_regularization_weight: float = 0.0
    collapse_std_threshold: float = 1e-4


class Trainer:
    """Encapsulated trainer with checkpointing and early stopping."""

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

        self.criterion = RegressionLoss(
            mode=config.loss_name,
            mse_weight=config.loss_mse_weight,
            mae_weight=config.loss_mae_weight,
            huber_delta=config.loss_huber_delta,
            bias_weight=config.bias_regularization_weight,
        )
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )

        self.batch_loss_csv = self.logs_dir / "batch_loss_log.csv"
        self.epoch_summary_csv = self.logs_dir / "epoch_summary.csv"
        self.attn_stats_csv = self.logs_dir / "attention_epoch_stats.csv"
        self.attn_dir = self.logs_dir / "attention_maps"
        self.attn_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir = self.logs_dir / "raw_predictions"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.curve_dir = self.logs_dir / "prediction_curves"
        self.curve_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, float]] = []
        self.best_val_total = float("inf")
        self.best_val_rmse = float("inf")
        self.best_epoch = 0
        self.no_improve_epochs = 0

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
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

    @staticmethod
    def _aggregate_attention(attn: torch.Tensor | None) -> tuple[float, float, float, float, np.ndarray | None]:
        if attn is None:
            return float("nan"), float("nan"), float("nan"), float("nan"), None
        attn_min = float(attn.min().item())
        attn_max = float(attn.max().item())
        attn_mean = float(attn.mean().item())
        attn_std = float(attn.std().item())
        attn_avg = attn.detach().mean(dim=0)
        while attn_avg.ndim > 2:
            attn_avg = attn_avg.mean(dim=0)
        return attn_min, attn_max, attn_mean, attn_std, attn_avg.cpu().numpy().astype(np.float32)

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float], np.ndarray | None]:
        mode = "train" if train else "valid"
        self.model.train(mode=train)

        total_loss = 0.0
        total_primary = 0.0
        total_rmse = 0.0
        total_mae = 0.0
        total_mean_error = 0.0
        total_bias_penalty = 0.0
        total_pred_mean = 0.0
        total_true_mean = 0.0
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
                pred = out["pred"]
                loss_out = self.criterion(pred, y)

                if train:
                    loss_out.total.backward()
                    grad_norm = float("nan")
                    if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.grad_clip_norm
                            ).item()
                        )
                    self.optimizer.step()
                else:
                    grad_norm = float("nan")

                pred_mean = float(pred.mean().item())
                true_mean = float(y.mean().item())
                total_loss += float(loss_out.total.item())
                total_primary += float(loss_out.primary.item())
                total_rmse += float(loss_out.rmse.item())
                total_mae += float(loss_out.mae.item())
                total_mean_error += float(loss_out.mean_error.item())
                total_bias_penalty += float(loss_out.bias_penalty.item())
                total_pred_mean += pred_mean
                total_true_mean += true_mean

                attn = out.get("attn_map")
                attn_min, attn_max, attn_mean, attn_std, attn_avg = self._aggregate_attention(attn)
                if attn_avg is not None:
                    attn_min_sum += attn_min
                    attn_max_sum += attn_max
                    attn_mean_sum += attn_mean
                    attn_std_sum += attn_std
                    attn_batches += 1
                    attn_map_sum = attn_avg if attn_map_sum is None else (attn_map_sum + attn_avg)

                current_lr = self.optimizer.param_groups[0]["lr"]
                batch_rows.append(
                    {
                        "epoch": epoch,
                        "phase": mode,
                        "batch": batch_idx,
                        "total_loss": float(loss_out.total.item()),
                        "primary_loss": float(loss_out.primary.item()),
                        "rmse": float(loss_out.rmse.item()),
                        "mae": float(loss_out.mae.item()),
                        "mean_error": float(loss_out.mean_error.item()),
                        "bias_penalty": float(loss_out.bias_penalty.item()),
                        "pred_mean": pred_mean,
                        "true_mean": true_mean,
                        "lr": float(current_lr),
                        "attn_min": attn_min,
                        "attn_max": attn_max,
                        "attn_mean": attn_mean,
                        "attn_std": attn_std,
                        "grad_norm": grad_norm,
                    }
                )

                pbar.set_postfix(
                    {
                        "total": f"{loss_out.total.item():.4f}",
                        "rmse": f"{loss_out.rmse.item():.4f}",
                        "bias": f"{loss_out.mean_error.item():+.4f}",
                    }
                )

        n = max(1, len(loader))
        metrics = {
            f"{mode}_total": total_loss / n,
            f"{mode}_primary": total_primary / n,
            f"{mode}_rmse": total_rmse / n,
            f"{mode}_mae": total_mae / n,
            f"{mode}_mean_error": total_mean_error / n,
            f"{mode}_bias_penalty": total_bias_penalty / n,
            f"{mode}_pred_mean": total_pred_mean / n,
            f"{mode}_true_mean": total_true_mean / n,
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

    def _collect_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        ids: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                out = self.model(x)
                preds.append(out["pred"].detach().cpu().numpy().reshape(-1))
                trues.append(y.detach().cpu().numpy().reshape(-1))
                ids.extend(list(batch["id"]))

        if not preds:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []

        return np.concatenate(preds).astype(np.float32), np.concatenate(trues).astype(np.float32), ids

    def _save_prediction_debug(self, epoch: int, phase: str, loader: DataLoader) -> Dict[str, float]:
        pred, true, ids = self._collect_predictions(loader)
        csv_path = self.pred_dir / f"epoch_{epoch:03d}_{phase}.csv"
        rows = []
        for idx, (bearing_id, y_true, y_pred) in enumerate(zip(ids, true, pred)):
            rows.append(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "sample_index": idx,
                    "bearing_id": bearing_id,
                    "true_rul": float(y_true),
                    "pred_rul": float(y_pred),
                    "error": float(y_pred - y_true),
                }
            )
        self._append_rows_csv(csv_path, rows)

        stats = {
            "pred_mean": float(np.mean(pred)) if len(pred) else float("nan"),
            "pred_std": float(np.std(pred)) if len(pred) else float("nan"),
            "true_mean": float(np.mean(true)) if len(true) else float("nan"),
            "true_std": float(np.std(true)) if len(true) else float("nan"),
            "mean_error": float(np.mean(pred - true)) if len(pred) else float("nan"),
        }

        if len(pred):
            plot_rul_curve(true, pred, self.curve_dir / f"epoch_{epoch:03d}_{phase}_all.png")
            seen = []
            for bearing_id in ids:
                if bearing_id not in seen:
                    seen.append(bearing_id)
                if len(seen) >= 3:
                    break
            for bearing_id in seen:
                idx = [i for i, bid in enumerate(ids) if bid == bearing_id]
                plot_rul_curve(
                    true[idx],
                    pred[idx],
                    self.curve_dir / f"epoch_{epoch:03d}_{phase}_{bearing_id}.png",
                )

        if len(pred) and stats["pred_std"] < self.config.collapse_std_threshold:
            self.logger.warning(
                "Prediction collapse detected for %s epoch %03d: pred_std=%.8f",
                phase,
                epoch,
                stats["pred_std"],
            )

        self.logger.info(
            (
                "%s epoch %03d raw prediction stats | pred mean/std %.6f/%.6f | "
                "true mean/std %.6f/%.6f | mean error %+ .6f"
            ),
            phase.title(),
            epoch,
            stats["pred_mean"],
            stats["pred_std"],
            stats["true_mean"],
            stats["true_std"],
            stats["mean_error"],
        )
        return stats

    def _save_checkpoint(self, epoch: int, valid_total: float, valid_rmse: float) -> None:
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

            valid_pred_stats = self._save_prediction_debug(epoch, "valid", valid_loader)

            epoch_row = {
                **tr,
                **va,
                **valid_pred_stats,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_row)
            self._append_rows_csv(self.epoch_summary_csv, [epoch_row])

            self.logger.info(
                (
                    "Epoch %03d | lr %.6f | train_total %.4f train_rmse %.4f train_mae %.4f train_bias %+ .4f | "
                    "valid_total %.4f valid_rmse %.4f valid_mae %.4f valid_bias %+ .4f"
                ),
                epoch,
                self.optimizer.param_groups[0]["lr"],
                tr["train_total"],
                tr["train_rmse"],
                tr["train_mae"],
                tr["train_mean_error"],
                va["valid_total"],
                va["valid_rmse"],
                va["valid_mae"],
                va["valid_mean_error"],
            )
            self.logger.info(
                (
                    "Epoch %03d attention stats | train min/max %.5f/%.5f mean/std %.5f/%.5f | "
                    "valid min/max %.5f/%.5f mean/std %.5f/%.5f"
                ),
                epoch,
                tr_attn_stats["attn_min"],
                tr_attn_stats["attn_max"],
                tr_attn_stats["attn_mean"],
                tr_attn_stats["attn_std"],
                va_attn_stats["attn_min"],
                va_attn_stats["attn_max"],
                va_attn_stats["attn_mean"],
                va_attn_stats["attn_std"],
            )

            self.scheduler.step(va["valid_rmse"])

            improved_total = va["valid_total"] < self.best_val_total
            improved_rmse = va["valid_rmse"] < self.best_val_rmse
            if improved_total:
                self.best_val_total = va["valid_total"]
            if improved_rmse:
                self.best_val_rmse = va["valid_rmse"]
                self.best_epoch = epoch
                self.no_improve_epochs = 0
                self._save_checkpoint(epoch, self.best_val_total, self.best_val_rmse)
                self.logger.info(
                    "Saved checkpoint at epoch %03d with valid_rmse=%.4f and valid_total=%.4f",
                    epoch,
                    self.best_val_rmse,
                    va["valid_total"],
                )
            else:
                self.no_improve_epochs += 1
                self.logger.info(
                    "EarlyStopping monitor valid_rmse did not improve (%d/%d)",
                    self.no_improve_epochs,
                    self.config.early_stopping_patience,
                )

            if self.no_improve_epochs >= self.config.early_stopping_patience:
                self.logger.info(
                    "Early stopping triggered at epoch %03d (best epoch %03d, best valid_rmse=%.4f)",
                    epoch,
                    self.best_epoch,
                    self.best_val_rmse,
                )
                break

        return self.history

    def evaluate(self, test_dataset) -> Dict[str, float]:
        if self.checkpoint_path.exists():
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(payload["model_state"])
            self.logger.info("Loaded best checkpoint from epoch %s", payload.get("epoch", "<unknown>"))

        test_loader = self._make_loader(test_dataset, shuffle=False)
        self._save_prediction_debug(0, "test", test_loader)
        result = evaluate_model(self.model, test_loader, self.device)
        metrics = {
            "rmse": result.rmse,
            "mae": result.mae,
            "r2": result.r2,
            "phm_score": result.phm_score,
            "pred_mean": result.pred_mean,
            "pred_std": result.pred_std,
            "true_mean": result.true_mean,
            "true_std": result.true_std,
            "mean_error": result.mean_error,
        }
        self.logger.info(
            "Test metrics | RMSE %.4f | MAE %.4f | R2 %.4f | PHM Score %.4f | bias %+ .4f",
            result.rmse,
            result.mae,
            result.r2,
            result.phm_score,
            result.mean_error,
        )
        return metrics
