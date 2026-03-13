"""Online inference script for PHM2012 RUL prediction.

This script is intentionally focused on robust batch-style online inference over
all CSV files in a directory. It uses the trained hybrid model:

TCN + Transformer + Degradation-aware Attention + Health Indicator.

Main workflow:
1) Load config and checkpoint.
2) Discover all CSV files under --input_dir.
3) For each file, load and normalize first two sensor channels.
4) Run sliding-window inference to predict RUL and HI at each cycle.
5) Save prediction CSV and two visualization curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.preprocess import prepare_datasets
from src.utils import get_device, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments namespace.
    """

    parser = argparse.ArgumentParser(description="PHM2012 directory online inference")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing PHM2012 CSV files.",
    )
    return parser.parse_args()


def load_signal_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one PHM2012 CSV signal file.

    Steps implemented per requirement:
    - read CSV with pandas
    - select first two numeric sensor columns
    - rename to sensor_0 / sensor_1 (internally)
    - normalize channels with per-file z-score
    - return normalized signal and cycle vector

    Args:
        path: CSV file path.

    Returns:
        Tuple:
            signal: normalized array of shape (T, 2)
            cycles: cycle index array of shape (T,)
    """

    frame = pd.read_csv(path)

    # Keep numeric columns only so text/meta fields do not break inference.
    numeric_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if len(numeric_cols) < 2:
        raise ValueError(f"File {path.name} does not contain at least two numeric columns.")

    # Use cycle column if available; otherwise create one.
    cycle_col = None
    for c in numeric_cols:
        if str(c).lower() == "cycle":
            cycle_col = c
            break

    if cycle_col is None:
        cycles = np.arange(1, len(frame) + 1, dtype=np.int32)
        sensor_candidates = numeric_cols
    else:
        cycles = frame[cycle_col].to_numpy(dtype=np.int32)
        sensor_candidates = [c for c in numeric_cols if c != cycle_col]

    if len(sensor_candidates) < 2:
        raise ValueError(f"File {path.name} has fewer than two usable sensor columns.")

    # Required mapping to sensor_0 and sensor_1.
    sensor_0 = frame[sensor_candidates[0]].to_numpy(dtype=np.float32)
    sensor_1 = frame[sensor_candidates[1]].to_numpy(dtype=np.float32)
    signal = np.stack([sensor_0, sensor_1], axis=1)

    # Normalize each sensor channel (z-score) for stable model input scale.
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    signal = (signal - mean) / np.clip(std, 1e-8, None)

    return signal.astype(np.float32), cycles


def predict_sequence(
    model: HybridRULModel,
    signal: np.ndarray,
    cycles: np.ndarray,
    window_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run sliding-window RUL and HI prediction for one signal sequence.

    Sliding window logic:
        for t in range(window_size, len(signal) + 1):
            window = signal[t-window_size:t]

    Args:
        model: Loaded PyTorch model.
        signal: Normalized signal array of shape (T, 2).
        cycles: Cycle array of shape (T,).
        window_size: Inference window length.
        device: CUDA/CPU device.

    Returns:
        DataFrame with columns:
            cycle, predicted_rul, health_indicator
    """

    rows: List[Dict[str, float]] = []
    model.eval()

    # Iterate from first full window to end-of-sequence.
    with torch.no_grad():
        for t in range(window_size, len(signal) + 1):
            # Build trailing window [t-window_size : t].
            window = signal[t - window_size : t]
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)

            # Model forward pass returns both RUL prediction and HI score.
            out = model(x)
            pred_rul = float(out["pred"].item())
            hi = float(out["hi"].item())

            rows.append(
                {
                    "cycle": int(cycles[t - 1]),
                    "predicted_rul": pred_rul,
                    "health_indicator": hi,
                }
            )

    return pd.DataFrame(rows)


def plot_rul_curve(pred_df: pd.DataFrame, save_path: Path) -> None:
    """Plot predicted RUL curve.

    Args:
        pred_df: Prediction dataframe.
        save_path: Output PNG path.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(pred_df["cycle"], pred_df["predicted_rul"], linewidth=2.0)
    plt.xlabel("cycle")
    plt.ylabel("predicted_rul")
    plt.title("RUL Prediction Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_hi_curve(pred_df: pd.DataFrame, save_path: Path) -> None:
    """Plot health indicator curve.

    Args:
        pred_df: Prediction dataframe.
        save_path: Output PNG path.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(pred_df["cycle"], pred_df["health_indicator"], linewidth=2.0, color="purple")
    plt.xlabel("cycle")
    plt.ylabel("health_indicator")
    plt.title("Health Indicator Curve")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def build_model(cfg: Dict, checkpoint_path: Path, device: torch.device, sensor_dim: int = 2) -> HybridRULModel:
    """Instantiate hybrid model and load checkpoint weights."""

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

    model = HybridRULModel(model_cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model


def main() -> None:
    """Program entrypoint."""

    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print("Input directory not found:")
        print(str(input_dir))
        return

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in directory: {input_dir}")
        return

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading model from checkpoint...")
    device = get_device()

    # Build scaler context by preparing datasets (keeps pipeline consistent).
    # We only need this call for reproducibility consistency with training setup.
    _ = prepare_datasets(
        dataset_root=cfg["data"]["dataset_root"],
        window_size=int(cfg["data"].get("window_size", 40)),
        stride=int(cfg["data"].get("stride", 10)),
        max_rul=int(cfg["data"].get("max_rul", 125)),
        valid_ratio=float(cfg["data"].get("valid_ratio", 0.2)),
        seed=seed,
        sensor_dim=2,
        max_windows_per_bearing=int(cfg["data"].get("max_windows_per_bearing", 20000)),
    )

    model = build_model(cfg, checkpoint, device=device, sensor_dim=2)

    window_size = int(cfg["data"].get("window_size", 40))
    out_dir = ROOT / "outputs" / "online_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in csv_files:
        print(f"Processing file: {file_path.name}")

        signal, cycles = load_signal_file(file_path)
        print(f"Signal length: {len(signal)}")

        pred_df = predict_sequence(
            model=model,
            signal=signal,
            cycles=cycles,
            window_size=window_size,
            device=device,
        )

        base = file_path.stem
        csv_path = out_dir / f"{base}_prediction.csv"
        rul_png = out_dir / f"{base}_rul_curve.png"
        hi_png = out_dir / f"{base}_hi_curve.png"

        pred_df.to_csv(csv_path, index=False)
        plot_rul_curve(pred_df, rul_png)
        plot_hi_curve(pred_df, hi_png)

        if len(pred_df):
            print(f"Predicted final RUL: {pred_df['predicted_rul'].iloc[-1]:.0f} cycles")
        else:
            print("Predicted final RUL: N/A (sequence shorter than window size)")

    print("\nResults saved to:")
    print(str(out_dir))


if __name__ == "__main__":
    main()
