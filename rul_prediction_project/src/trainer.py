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

from .checkpoint_compat import load_model_state_strict
from .evaluator import evaluate_model
from .loss import RawRegressionLoss, compute_regression_metrics
from .utils import save_csv_rows, save_json
from .visualization import plot_rul_curve


@dataclass
class TrainerConfig:
    """Trainer hyperparameter container."""

    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 0
    early_stopping_patience: int = 5
    early_stopping_min_epochs: int = 10
    early_stopping_min_delta: float = 0.0
    grad_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    weight_decay: float = 0.0
    loss_name: str = "mse"
    loss_mse_weight: float = 1.0
    loss_mae_weight: float = 0.3
    huber_delta: float = 0.1
    bias_regularization_weight: float = 0.0
    std_regularization_weight: float = 0.0
    correlation_regularization_weight: float = 0.0
    hi_supervision_weight: float = 0.0
    hi_rank_weight: float = 0.0
    hi_monotonic_weight: float = 0.1
    hi_variance_weight: float = 0.05
    hi_variance_floor: float = 0.05
    hi_smoothness_weight: float = 0.01
    residual_regularization_weight: float = 0.0
    hi_supervision_mode: str = "weak"
    hi_target_mode: str = "health"
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
        figures_dir: Path | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.logger = logger
        self.logs_dir = logs_dir
        self.figures_dir = figures_dir or logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = RawRegressionLoss(
            mode=config.loss_name,
            huber_delta=config.huber_delta,
            mse_weight=config.loss_mse_weight,
            mae_weight=config.loss_mae_weight,
            bias_regularization_weight=config.bias_regularization_weight,
            std_regularization_weight=config.std_regularization_weight,
            correlation_regularization_weight=config.correlation_regularization_weight,
        )
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if str(config.hi_supervision_mode).lower() not in {"weak", "direct"}:
            raise ValueError("train.hi_supervision_mode must be one of: weak, direct")
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=1e-6,
        )

        self.batch_loss_csv = self.logs_dir / "batch_loss_log.csv"
        self.epoch_metrics_csv = self.logs_dir / "epoch_metrics.csv"
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

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float], np.ndarray | None]:
        mode = "train" if train else "valid"
        self.model.train(mode=train)

        totals = {
            "loss": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "bias": 0.0,
            "bias_penalty": 0.0,
            "std_penalty": 0.0,
            "corr_penalty": 0.0,
            "corr_value": 0.0,
            "hi_supervision": 0.0,
            "hi_rank": 0.0,
            "hi_monotonic": 0.0,
            "hi_var_penalty": 0.0,
            "hi_smoothness": 0.0,
            "residual_penalty": 0.0,
        }
        batch_rows: List[Dict[str, float]] = []

        attn_min_sum = 0.0
        attn_max_sum = 0.0
        attn_mean_sum = 0.0
        attn_std_sum = 0.0
        attn_cond_hi_std_sum = 0.0
        attn_head_bias_std_sum = 0.0
        attn_degradation_bias_std_sum = 0.0
        attn_delta_sum = 0.0
        attn_entropy_sum = 0.0
        attn_batches = 0
        attn_map_sum: np.ndarray | None = None
        hi_mean_sum = 0.0
        hi_std_sum = 0.0
        hi_min_sum = 0.0
        hi_max_sum = 0.0
        hi_batches = 0

        pbar = tqdm(loader, desc=f"{mode.title()} Epoch {epoch:03d}/{self.config.epochs:03d}", leave=False)
        grad_context = torch.enable_grad() if train else torch.no_grad()

        with grad_context:
            for batch_idx, batch in enumerate(pbar, start=1):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                fixed_hi_mode = bool(getattr(getattr(self.model, "cfg", None), "use_fixed_hi", False))

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                out = self.model(x, fixed_hi=y.detach() if fixed_hi_mode else None)
                loss_out = self.criterion(out["pred"], y)
                hi_sup = torch.tensor(0.0, device=self.device)
                hi_rank = torch.tensor(0.0, device=self.device)
                hi_var_pen = torch.tensor(0.0, device=self.device)
                hi_smoothness = torch.tensor(0.0, device=self.device)
                hi_target = y
                supervision_mode = str(self.config.hi_supervision_mode).lower()
                if (
                    (not fixed_hi_mode)
                    and supervision_mode == "direct"
                    and self.config.hi_supervision_weight > 0
                    and out.get("hi") is not None
                ):
                    hi_sup = torch.nn.functional.mse_loss(out["hi"], hi_target)
                if (not fixed_hi_mode) and self.config.hi_rank_weight > 0 and out.get("hi_temporal") is not None:
                    hi_temporal_rank = out["hi_temporal"].squeeze(-1)
                    if hi_temporal_rank.size(1) > 1:
                        # Early windows should be at least as healthy as late windows.
                        hi_rank = torch.nn.functional.softplus(
                            hi_temporal_rank[:, -1] - hi_temporal_rank[:, 0]
                        ).mean()
                if (not fixed_hi_mode) and self.config.hi_variance_weight > 0 and out.get("hi") is not None:
                    hi_std = torch.std(out["hi"].view(-1), unbiased=False)
                    hi_var_pen = torch.relu(torch.tensor(self.config.hi_variance_floor, device=self.device) - hi_std)
                hi_temporal = out.get("hi_temporal")
                hi_temporal_logit = out.get("hi_temporal_logit")
                residual_penalty = torch.tensor(0.0, device=self.device)
                if (
                    str(getattr(self.model, "rul_head_mode", "plain")).lower() == "hi_guided_residual"
                    and self.config.residual_regularization_weight > 0
                    and out.get("residual_component") is not None
                ):
                    residual_penalty = out["residual_component"].abs().mean()
                hi_monotonic = torch.tensor(0.0, device=self.device)
                if (not fixed_hi_mode) and hi_temporal is not None and hi_temporal.size(1) > 1:
                    hi_seq = hi_temporal.squeeze(-1)
                    hi_monotonic = torch.relu(hi_seq[:, 1:] - hi_seq[:, :-1]).mean()
                if (not fixed_hi_mode) and self.config.hi_smoothness_weight > 0 and hi_temporal is not None and hi_temporal.size(1) > 1:
                    smooth_src = hi_temporal_logit if hi_temporal_logit is not None else hi_temporal
                    hi_delta = smooth_src[:, 1:, :] - smooth_src[:, :-1, :]
                    hi_smoothness = hi_delta.pow(2).mean()

                total_loss = (
                    loss_out.total
                    + self.config.hi_supervision_weight * hi_sup
                    + self.config.hi_rank_weight * hi_rank
                    + self.config.hi_monotonic_weight * hi_monotonic
                    + self.config.hi_variance_weight * hi_var_pen
                    + self.config.hi_smoothness_weight * hi_smoothness
                    + self.config.residual_regularization_weight * residual_penalty
                )

                if train:
                    total_loss.backward()
                    grad_norm = float("nan")
                    if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm).item())
                    self.optimizer.step()
                else:
                    grad_norm = float("nan")

                totals["loss"] += float(total_loss.item())
                totals["rmse"] += float(loss_out.rmse.item())
                totals["mae"] += float(loss_out.mae.item())
                totals["mse"] += float(loss_out.mse.item())
                totals["bias"] += float(loss_out.mean_bias.item())
                totals["bias_penalty"] += float(loss_out.bias_penalty.item())
                totals["std_penalty"] += float(loss_out.std_penalty.item())
                totals["corr_penalty"] += float(loss_out.corr_penalty.item())
                totals["corr_value"] += float(loss_out.corr_value.item())
                totals["hi_supervision"] += float(hi_sup.item())
                totals["hi_rank"] += float(hi_rank.item())
                totals["hi_monotonic"] += float(hi_monotonic.item())
                totals["hi_var_penalty"] += float(hi_var_pen.item())
                totals["hi_smoothness"] += float(hi_smoothness.item())
                totals["residual_penalty"] += float(residual_penalty.item())

                hi = out.get("hi")
                if hi is not None:
                    hi_mean_sum += float(hi.mean().item())
                    hi_std_sum += float(hi.std().item())
                    hi_min_sum += float(hi.min().item())
                    hi_max_sum += float(hi.max().item())
                    hi_batches += 1

                attn = out.get("attn_map")
                attn_debug = out.get("attention_debug")
                if attn is not None:
                    attn_min = float(attn.min().item())
                    attn_max = float(attn.max().item())
                    attn_mean = float(attn.mean().item())
                    attn_std = float(attn.std().item())
                    attn_min_sum += attn_min
                    attn_max_sum += attn_max
                    attn_mean_sum += attn_mean
                    attn_std_sum += attn_std
                    if attn_debug is not None:
                        attn_cond_hi_std_sum += float(attn_debug["hi_cond_std"].item())
                        attn_head_bias_std_sum += float(attn_debug["head_bias_std"].item())
                        attn_degradation_bias_std_sum += float(attn_debug["degradation_bias_std"].item())
                        attn_delta_sum += float(attn_debug["attn_delta_l1"].item())
                        attn_entropy_sum += float(attn_debug["attn_entropy"].item())
                    attn_batches += 1
                    attn_2d = attn.detach().mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
                    attn_map_sum = attn_2d if attn_map_sum is None else (attn_map_sum + attn_2d)
                else:
                    attn_min = float("nan")
                    attn_max = float("nan")
                    attn_mean = float("nan")
                    attn_std = float("nan")

                current_lr = self.optimizer.param_groups[0]["lr"]
                batch_rows.append(
                    {
                        "epoch": epoch,
                        "phase": mode,
                        "batch": batch_idx,
                        "total_loss": float(total_loss.item()),
                        "rmse": float(loss_out.rmse.item()),
                        "mae": float(loss_out.mae.item()),
                        "mse": float(loss_out.mse.item()),
                        "mean_bias": float(loss_out.mean_bias.item()),
                        "bias_penalty": float(loss_out.bias_penalty.item()),
                        "std_penalty": float(loss_out.std_penalty.item()),
                        "corr_penalty": float(loss_out.corr_penalty.item()),
                        "corr_value": float(loss_out.corr_value.item()),
                        "hi_supervision_loss": float(hi_sup.item()),
                        "hi_rank_loss": float(hi_rank.item()),
                        "hi_variance_penalty": float(hi_var_pen.item()),
                        "hi_smoothness_loss": float(hi_smoothness.item()),
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
                        "loss": f"{total_loss.item():.4f}",
                        "rmse": f"{loss_out.rmse.item():.4f}",
                        "bias": f"{loss_out.mean_bias.item():+.4f}",
                    }
                )

        n = max(1, len(loader))
        metrics = {
            f"{mode}_total": totals["loss"] / n,
            f"{mode}_rmse": totals["rmse"] / n,
            f"{mode}_mae": totals["mae"] / n,
            f"{mode}_mse": totals["mse"] / n,
            f"{mode}_mean_bias": totals["bias"] / n,
            f"{mode}_bias_penalty": totals["bias_penalty"] / n,
            f"{mode}_std_penalty": totals["std_penalty"] / n,
            f"{mode}_corr_penalty": totals["corr_penalty"] / n,
            f"{mode}_corr_value": totals["corr_value"] / n,
            f"{mode}_hi_supervision": totals["hi_supervision"] / n,
            f"{mode}_hi_rank": totals["hi_rank"] / n,
            f"{mode}_hi_monotonic": totals["hi_monotonic"] / n,
            f"{mode}_hi_var_penalty": totals["hi_var_penalty"] / n,
            f"{mode}_hi_smoothness": totals["hi_smoothness"] / n,
            f"{mode}_hi_mean": hi_mean_sum / max(1, hi_batches),
            f"{mode}_hi_std": hi_std_sum / max(1, hi_batches),
            f"{mode}_hi_min": hi_min_sum / max(1, hi_batches),
            f"{mode}_hi_max": hi_max_sum / max(1, hi_batches),
        }

        attn_stats = {
            "epoch": float(epoch),
            "attn_min": attn_min_sum / max(1, attn_batches),
            "attn_max": attn_max_sum / max(1, attn_batches),
            "attn_mean": attn_mean_sum / max(1, attn_batches),
            "attn_std": attn_std_sum / max(1, attn_batches),
            "cond_hi_std": attn_cond_hi_std_sum / max(1, attn_batches),
            "head_bias_std": attn_head_bias_std_sum / max(1, attn_batches),
            "degradation_bias_std": attn_degradation_bias_std_sum / max(1, attn_batches),
            "attn_delta_l1": attn_delta_sum / max(1, attn_batches),
            "attn_entropy": attn_entropy_sum / max(1, attn_batches),
            "num_batches_with_attention": float(attn_batches),
        }

        attn_avg = None
        if attn_map_sum is not None and attn_batches > 0:
            attn_avg = (attn_map_sum / float(attn_batches)).astype(np.float32)

        return metrics, batch_rows, attn_stats, attn_avg

    def _collect_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        his: List[np.ndarray] = []
        ids: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                fixed_hi_mode = bool(getattr(getattr(self.model, "cfg", None), "use_fixed_hi", False))
                out = self.model(x, fixed_hi=y.detach() if fixed_hi_mode else None)
                preds.append(out["pred"].detach().cpu().numpy().reshape(-1))
                trues.append(y.detach().cpu().numpy().reshape(-1))
                his.append(out["hi"].detach().cpu().numpy().reshape(-1))
                ids.extend(list(batch["id"]))

        if not preds:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                [],
            )

        return (
            np.concatenate(preds).astype(np.float32),
            np.concatenate(trues).astype(np.float32),
            np.concatenate(his).astype(np.float32),
            ids,
        )

    def _save_prediction_debug(self, epoch: int, phase: str, loader: DataLoader) -> Dict[str, float]:
        pred, true, hi, ids = self._collect_predictions(loader)
        csv_path = self.pred_dir / f"epoch_{epoch:03d}_{phase}.csv"
        rows = []
        for idx, (bearing_id, y_true, y_pred, hi_i) in enumerate(zip(ids, true, pred, hi)):
            rows.append(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "sample_index": idx,
                    "bearing_id": bearing_id,
                    "true_rul": float(y_true),
                    "pred_rul": float(y_pred),
                    "error": float(y_pred - y_true),
                    "hi": float(hi_i),
                }
            )
        save_csv_rows(csv_path, rows)

        if len(pred):
            regression_stats = compute_regression_metrics(
                torch.from_numpy(pred.astype(np.float32)),
                torch.from_numpy(true.astype(np.float32)),
            )
        else:
            regression_stats = {
                "pred_mean": float("nan"),
                "pred_std": float("nan"),
                "true_mean": float("nan"),
                "true_std": float("nan"),
                "mean_bias": float("nan"),
                "r2": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "mse": float("nan"),
                "pred_min": float("nan"),
                "pred_max": float("nan"),
                "true_min": float("nan"),
                "true_max": float("nan"),
            }
        hi_true_corr = float(np.corrcoef(hi, true)[0, 1]) if len(hi) > 1 else float("nan")
        hi_true_degradation_corr = float(np.corrcoef(hi, 1.0 - true)[0, 1]) if len(hi) > 1 else float("nan")
        hi_pred_corr = float(np.corrcoef(hi, pred)[0, 1]) if len(hi) > 1 else float("nan")
        pred_true_corr = float(np.corrcoef(pred, true)[0, 1]) if len(hi) > 1 else float("nan")
        regression_stats["hi_mean"] = float(np.mean(hi)) if len(hi) else float("nan")
        regression_stats["hi_std"] = float(np.std(hi)) if len(hi) else float("nan")
        regression_stats["hi_true_corr"] = hi_true_corr if np.isfinite(hi_true_corr) else float("nan")
        regression_stats["hi_true_degradation_corr"] = (
            hi_true_degradation_corr if np.isfinite(hi_true_degradation_corr) else float("nan")
        )
        regression_stats["hi_pred_corr"] = hi_pred_corr if np.isfinite(hi_pred_corr) else float("nan")
        regression_stats["pred_true_corr"] = pred_true_corr if np.isfinite(pred_true_corr) else float("nan")
        regression_stats["warning_shortcut"] = bool(
            np.isfinite(regression_stats["hi_pred_corr"]) and regression_stats["hi_pred_corr"] > 0.95
        )
        regression_stats["warning_hi_collapse"] = bool(
            np.isfinite(regression_stats["hi_std"]) and regression_stats["hi_std"] < 0.01
        )

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

        if len(pred) and regression_stats["pred_std"] < self.config.collapse_std_threshold:
            self.logger.warning(
                "Prediction collapse detected for %s epoch %03d: pred_std=%.8f",
                phase,
                epoch,
                regression_stats["pred_std"],
            )
        if regression_stats["warning_shortcut"]:
            self.logger.warning(
                "Possible shortcut for %s epoch %03d: corr(pred, hi)=%.4f (>0.95)",
                phase,
                epoch,
                regression_stats["hi_pred_corr"],
            )
        if regression_stats["warning_hi_collapse"]:
            self.logger.warning(
                "HI collapse for %s epoch %03d: std(hi)=%.6f (<0.01)",
                phase,
                epoch,
                regression_stats["hi_std"],
            )

        self.logger.info(
            (
                "%s epoch %03d prediction stats | rmse %.6f | mae %.6f | bias %+.6f | "
                "pred mean/std %.6f/%.6f | true mean/std %.6f/%.6f | hi mean/std %.6f/%.6f | "
                "hi-true(RUL) corr %.4f | hi-true(degradation) corr %.4f | hi-pred corr %.4f"
                " | pred-true corr %.4f"
            ),
            phase.title(),
            epoch,
            regression_stats["rmse"],
            regression_stats["mae"],
            regression_stats["mean_bias"],
            regression_stats["pred_mean"],
            regression_stats["pred_std"],
            regression_stats["true_mean"],
            regression_stats["true_std"],
            regression_stats["hi_mean"],
            regression_stats["hi_std"],
            regression_stats["hi_true_corr"],
            regression_stats["hi_true_degradation_corr"],
            regression_stats["hi_pred_corr"],
            regression_stats["pred_true_corr"],
        )
        return regression_stats

    def _save_checkpoint(self, epoch: int, valid_total: float, valid_rmse: float) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_valid_total": valid_total,
            "best_valid_rmse": valid_rmse,
            "config": self.config.__dict__,
        }
        torch.save(payload, self.checkpoint_path)

    def fit(self, train_dataset, valid_dataset) -> List[Dict[str, float]]:
        train_loader = self._make_loader(train_dataset, shuffle=True)
        valid_loader = self._make_loader(valid_dataset, shuffle=False)

        for epoch in range(1, self.config.epochs + 1):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            tr, tr_rows, tr_attn_stats, tr_attn_avg = self._run_epoch(train_loader, epoch, train=True)
            va, va_rows, va_attn_stats, va_attn_avg = self._run_epoch(valid_loader, epoch, train=False)

            self._append_rows_csv(self.batch_loss_csv, tr_rows + va_rows)
            self._append_rows_csv(
                self.attn_stats_csv,
                [
                    {"epoch": epoch, "phase": "train", **tr_attn_stats},
                    {"epoch": epoch, "phase": "valid", **va_attn_stats},
                ],
            )

            if tr_attn_avg is not None:
                np.save(self.attn_dir / f"epoch_{epoch:03d}_train.npy", tr_attn_avg)
            if va_attn_avg is not None:
                np.save(self.attn_dir / f"epoch_{epoch:03d}_valid.npy", va_attn_avg)

            valid_pred_stats = self._save_prediction_debug(epoch, "valid", valid_loader)
            # Use full-validation aggregation metrics (not batch-averaged surrogates)
            # for scheduler and early-stopping decisions.
            va["valid_rmse"] = float(valid_pred_stats.get("rmse", va["valid_rmse"]))
            va["valid_mae"] = float(valid_pred_stats.get("mae", va["valid_mae"]))
            va["valid_mse"] = float(valid_pred_stats.get("mse", va["valid_mse"]))
            va["valid_mean_bias"] = float(valid_pred_stats.get("mean_bias", va["valid_mean_bias"]))
            self.scheduler.step(va["valid_rmse"])
            new_lr = float(self.optimizer.param_groups[0]["lr"])
            lr_changed = abs(new_lr - current_lr) > 1e-12

            epoch_row = {
                "epoch": epoch,
                **tr,
                **va,
                **{f"valid_pred_{k}": v for k, v in valid_pred_stats.items()},
                "lr": new_lr,
            }
            self.history.append(epoch_row)
            save_csv_rows(self.epoch_metrics_csv, self.history)
            save_json(self.logs_dir / "history.json", {"history": self.history})

            self.logger.info(
                (
                    "Epoch %03d/%03d | lr %.6f%s | train rmse %.4f mae %.4f bias %+.4f std_pen %.6f corr %.4f "
                    "hi_sup %.6f hi_rank %.6f hi_mono %.6f hi_var %.6f hi_smooth %.6f | "
                    "valid rmse %.4f mae %.4f bias %+.4f std_pen %.6f corr %.4f "
                    "hi_sup %.6f hi_rank %.6f hi_mono %.6f hi_var %.6f hi_smooth %.6f"
                ),
                epoch,
                self.config.epochs,
                new_lr,
                " (scheduler step)" if lr_changed else "",
                tr["train_rmse"],
                tr["train_mae"],
                tr["train_mean_bias"],
                tr["train_std_penalty"],
                tr["train_corr_value"],
                tr["train_hi_supervision"],
                tr["train_hi_rank"],
                tr["train_hi_monotonic"],
                tr["train_hi_var_penalty"],
                tr["train_hi_smoothness"],
                va["valid_rmse"],
                va["valid_mae"],
                va["valid_mean_bias"],
                va["valid_std_penalty"],
                va["valid_corr_value"],
                va["valid_hi_supervision"],
                va["valid_hi_rank"],
                va["valid_hi_monotonic"],
                va["valid_hi_var_penalty"],
                va["valid_hi_smoothness"],
            )
            self.logger.info(
                "Epoch %03d HI stats | train mean/std/min/max %.4f/%.4f/%.4f/%.4f | valid %.4f/%.4f/%.4f/%.4f",
                epoch,
                tr["train_hi_mean"],
                tr["train_hi_std"],
                tr["train_hi_min"],
                tr["train_hi_max"],
                va["valid_hi_mean"],
                va["valid_hi_std"],
                va["valid_hi_min"],
                va["valid_hi_max"],
            )
            self.logger.info(
                (
                    "Epoch %03d/%03d attention stats | train min/max %.5f/%.5f mean/std %.5f/%.5f "
                    "cond_hi_std %.5f head_bias_std %.5f degr_bias_std %.5f delta_l1 %.6f entropy %.6f | "
                    "valid min/max %.5f/%.5f mean/std %.5f/%.5f cond_hi_std %.5f "
                    "head_bias_std %.5f degr_bias_std %.5f delta_l1 %.6f entropy %.6f"
                ),
                epoch,
                self.config.epochs,
                tr_attn_stats["attn_min"],
                tr_attn_stats["attn_max"],
                tr_attn_stats["attn_mean"],
                tr_attn_stats["attn_std"],
                tr_attn_stats["cond_hi_std"],
                tr_attn_stats["head_bias_std"],
                tr_attn_stats["degradation_bias_std"],
                tr_attn_stats["attn_delta_l1"],
                tr_attn_stats["attn_entropy"],
                va_attn_stats["attn_min"],
                va_attn_stats["attn_max"],
                va_attn_stats["attn_mean"],
                va_attn_stats["attn_std"],
                va_attn_stats["cond_hi_std"],
                va_attn_stats["head_bias_std"],
                va_attn_stats["degradation_bias_std"],
                va_attn_stats["attn_delta_l1"],
                va_attn_stats["attn_entropy"],
            )

            improved_total = va["valid_total"] < self.best_val_total
            min_delta = max(0.0, float(self.config.early_stopping_min_delta))
            improved_rmse = va["valid_rmse"] < (self.best_val_rmse - min_delta)
            if improved_total:
                self.best_val_total = va["valid_total"]

            if improved_rmse:
                self.best_val_rmse = va["valid_rmse"]
                self.best_epoch = epoch
                self.no_improve_epochs = 0
                self._save_checkpoint(epoch, self.best_val_total, self.best_val_rmse)
                self.logger.info(
                    "Saved new best checkpoint at epoch %03d with valid_rmse=%.4f",
                    epoch,
                    self.best_val_rmse,
                )
            else:
                self.no_improve_epochs += 1
                self.logger.info(
                    "EarlyStopping monitor valid_rmse did not improve (%d/%d)",
                    self.no_improve_epochs,
                    self.config.early_stopping_patience,
                )

            if epoch < int(self.config.early_stopping_min_epochs):
                continue

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
            try:
                payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                payload = torch.load(self.checkpoint_path, map_location=self.device)
            load_model_state_strict(self.model, payload)
            self.logger.info("Loaded best checkpoint from epoch %s", payload.get("epoch", "<unknown>"))

        test_loader = self._make_loader(test_dataset, shuffle=False)
        self._save_prediction_debug(0, "test", test_loader)
        result = evaluate_model(self.model, test_loader, self.device)
        regression_metrics = compute_regression_metrics(
            torch.from_numpy(result.y_pred.astype(np.float32)),
            torch.from_numpy(result.y_true.astype(np.float32)),
        )
        metrics = {
            "rmse": regression_metrics["rmse"],
            "mae": regression_metrics["mae"],
            "r2": regression_metrics["r2"],
            "mean_bias": regression_metrics["mean_bias"],
            "pred_std": regression_metrics["pred_std"],
            "true_std": regression_metrics["true_std"],
            "pred_true_std_ratio": regression_metrics["pred_std"] / max(regression_metrics["true_std"], 1e-8),
            "phm_score": result.phm_score,
            "best_epoch": self.best_epoch,
        }
        self.logger.info("TEST RESULTS")
        self.logger.info("RMSE: %.6f", metrics["rmse"])
        self.logger.info("MAE: %.6f", metrics["mae"])
        self.logger.info("R2: %.6f", metrics["r2"])
        self.logger.info("Mean Bias: %+.6f", metrics["mean_bias"])
        self.logger.info("Pred Std: %.6f", metrics["pred_std"])
        self.logger.info("True Std: %.6f", metrics["true_std"])
        self.logger.info("Pred/True Std Ratio: %.6f", metrics["pred_true_std_ratio"])
        self.logger.info("PHM Score: %.6f", metrics["phm_score"])
        return metrics
