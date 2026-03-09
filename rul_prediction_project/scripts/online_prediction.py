"""Online RUL prediction simulation script.

This script simulates real-time industrial monitoring by continuously consuming
new vibration signal chunks, applying preprocessing + optional time-frequency
feature extraction, and running model inference to estimate Remaining Useful
Life (RUL).

Pipeline:
    raw signal -> standardization -> sliding window update -> model inference

Usage example:
    python scripts/online_prediction.py \
      --config configs/config.yaml \
      --checkpoint outputs/checkpoints/best_model.pth \
      --input_file /path/to/new_signal.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.time_frequency import TimeFrequencyFeatureExtractor, compute_spectral_statistics
from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.preprocess import StandardScaler, prepare_datasets
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, set_seed


@dataclass
class OnlineState:
    """State container for streaming prediction.

    Attributes:
        buffer: Latest rolling window of standardized sensor samples.
        timestamps: Optional cycle/timestamp indices associated with buffer rows.
    """

    buffer: np.ndarray
    timestamps: List[int]


class OnlineRULPredictor:
    """Stateful online predictor that updates with incoming samples.

    Input sample shape:
        single observation ``(sensors,)``

    Internal window shape:
        ``(window_size, sensors)``

    Output:
        scalar RUL prediction per update step once buffer is full.
    """

    def __init__(
        self,
        model: HybridRULModel,
        scaler: StandardScaler,
        window_size: int,
        device: torch.device,
        use_time_frequency: bool = False,
    ):
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.device = device
        self.use_time_frequency = use_time_frequency

        self.state: Optional[OnlineState] = None
        self.tf_extractor = TimeFrequencyFeatureExtractor() if use_time_frequency else None

    def reset(self, sensor_dim: int) -> None:
        """Reset streaming state for a new sequence.

        Args:
            sensor_dim: Number of sensor channels.
        """

        self.state = OnlineState(
            buffer=np.zeros((0, sensor_dim), dtype=np.float32),
            timestamps=[],
        )

    def _ensure_state(self, sensor_dim: int) -> None:
        if self.state is None:
            self.reset(sensor_dim)

    def update(self, sample: np.ndarray, timestamp: int) -> Optional[Dict[str, float]]:
        """Update buffer with one new sample and predict RUL if window is ready.

        Args:
            sample: Raw sensor sample of shape ``(sensors,)``.
            timestamp: Current cycle index.

        Returns:
            Optional prediction dict containing ``pred_rul`` and ``hi``.
            Returns ``None`` until enough samples are accumulated.
        """

        if sample.ndim != 1:
            raise ValueError("sample must be 1D array with shape (sensors,)")

        self._ensure_state(sample.shape[0])
        assert self.state is not None

        scaled = self.scaler.transform(sample.reshape(1, -1)).reshape(-1)
        self.state.buffer = np.vstack([self.state.buffer, scaled[None, :]])
        self.state.timestamps.append(timestamp)

        if len(self.state.timestamps) > self.window_size:
            self.state.buffer = self.state.buffer[-self.window_size :]
            self.state.timestamps = self.state.timestamps[-self.window_size :]

        if self.state.buffer.shape[0] < self.window_size:
            return None

        x = torch.from_numpy(self.state.buffer).float().unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            pred = float(out["pred"].item())
            hi = float(out["hi"].item())

            result = {
                "timestamp": float(timestamp),
                "pred_rul": pred,
                "health_indicator": hi,
            }

            if self.use_time_frequency and self.tf_extractor is not None:
                tf_out = self.tf_extractor(x)
                stats = compute_spectral_statistics(tf_out.fused)
                result["spectral_energy"] = float(stats["spectral_energy"].mean().item())
                result["spectral_centroid"] = float(stats["spectral_centroid"].mean().item())

            return result


def load_signal_file(path: Path) -> pd.DataFrame:
    """Load incoming vibration signal file.

    Supports CSV/TXT files with numeric sensor columns and optional cycle column.

    Args:
        path: File path.

    Returns:
        DataFrame sorted by cycle if present.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, sep=None, engine="python")

    numeric_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    frame = frame[numeric_cols].copy()

    if "cycle" not in [c.lower() for c in frame.columns]:
        frame.insert(0, "cycle", np.arange(1, len(frame) + 1))

    # Normalize cycle column name.
    for c in list(frame.columns):
        if c.lower() == "cycle" and c != "cycle":
            frame = frame.rename(columns={c: "cycle"})
    frame = frame.sort_values("cycle").reset_index(drop=True)
    return frame


def build_model_from_config(cfg: Dict, sensor_dim: int, checkpoint_path: Path, device: torch.device) -> HybridRULModel:
    """Instantiate hybrid model and load checkpoint weights.

    Args:
        cfg: Full config dictionary.
        sensor_dim: Number of sensor channels.
        checkpoint_path: Path to saved model checkpoint.
        device: Torch device.

    Returns:
        Loaded ``HybridRULModel``.
    """

    model_cfg = cfg["model"]
    arch_cfg = ModelConfig(
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

    model = HybridRULModel(arch_cfg).to(device)

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state"])
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return model


def simulate_streaming(
    predictor: OnlineRULPredictor,
    frame: pd.DataFrame,
    interval_sec: float = 0.0,
) -> List[Dict[str, float]]:
    """Simulate real-time inference over sequential rows.

    Args:
        predictor: Online predictor instance.
        frame: DataFrame with ``cycle`` and sensor columns.
        interval_sec: Optional sleep interval to mimic real-time speed.

    Returns:
        List of prediction dictionaries.
    """

    sensor_cols = [c for c in frame.columns if c != "cycle"]
    results: List[Dict[str, float]] = []

    for _, row in frame.iterrows():
        ts = int(row["cycle"])
        sample = row[sensor_cols].to_numpy(dtype=np.float32)
        pred = predictor.update(sample, timestamp=ts)
        if pred is not None:
            results.append(pred)
        if interval_sec > 0:
            time.sleep(interval_sec)

    return results


def save_online_results(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    """Save online prediction records into CSV.

    Args:
        path: Output CSV path.
        rows: Sequence of prediction dicts.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for online simulation."""

    parser = argparse.ArgumentParser(description="Online RUL prediction simulation")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument("--input_file", type=str, required=True, help="Incoming vibration signal CSV/TXT")
    parser.add_argument("--interval_sec", type=float, default=0.0, help="Sleep interval between samples")
    parser.add_argument("--use_time_frequency", action="store_true", help="Compute TF diagnostics in output")
    return parser.parse_args()


def main() -> None:
    """Entry point for online RUL prediction simulation."""

    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)
    logger = configure_logging(paths["logs"] / "online_prediction.log")

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    # Build scaler from training data statistics to mimic deployment process.
    prepared = prepare_datasets(
        dataset_root=cfg["data"]["dataset_root"],
        window_size=int(cfg["data"]["window_size"]),
        stride=int(cfg["data"]["stride"]),
        max_rul=int(cfg["data"]["max_rul"]),
        valid_ratio=float(cfg["data"]["valid_ratio"]),
        seed=seed,
    )

    input_frame = load_signal_file(Path(args.input_file))
    sensor_dim = len([c for c in input_frame.columns if c != "cycle"])

    device = get_device()
    model = build_model_from_config(cfg, sensor_dim, Path(args.checkpoint), device)

    predictor = OnlineRULPredictor(
        model=model,
        scaler=prepared.scaler,
        window_size=int(cfg["data"]["window_size"]),
        device=device,
        use_time_frequency=bool(args.use_time_frequency),
    )

    results = simulate_streaming(
        predictor,
        input_frame,
        interval_sec=float(args.interval_sec),
    )

    out_csv = paths["outputs"] / "results" / "online_predictions.csv"
    save_online_results(out_csv, results)

    if results:
        logger.info("Processed %d online predictions", len(results))
        logger.info("Last prediction: %s", results[-1])
    else:
        logger.warning("No predictions generated (insufficient samples for full window)")

    logger.info("Online prediction results saved to %s", out_csv)


if __name__ == "__main__":
    main()
