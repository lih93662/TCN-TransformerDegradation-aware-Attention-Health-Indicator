"""Online PHM2012 directory inference for RUL prediction.

This script performs directory-based inference for the hybrid model:
TCN + Transformer + Degradation-aware Attention + Health Indicator.

Key behavior:
- Accepts --input_dir (absolute path only).
- Lists all CSV files under that directory.
- For each file, loads signals, builds sliding windows, predicts RUL + HI,
  optionally collects attention maps, and saves CSV/plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.preprocess import StandardScaler, prepare_datasets
from src.utils import get_device, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace with config, checkpoint, and input_dir.
    """

    parser = argparse.ArgumentParser(description="PHM2012 online directory inference")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Absolute path to PHM2012 test directory containing CSV files.",
    )
    return parser.parse_args()


def load_signal_file(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load one PHM2012 CSV file and map channels to sensor_0/sensor_1.

    Signal loading logic:
    - read CSV with pandas
    - keep numeric columns
    - use cycle column if present; otherwise create cycle index
    - select first two sensor columns
    - rename to sensor_0, sensor_1

    Args:
        path: CSV file path.

    Returns:
        Tuple of:
        - frame with columns [cycle, sensor_0, sensor_1]
        - cycle array (int32)
    """

    frame = pd.read_csv(path)
    num_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if len(num_cols) < 2:
        raise ValueError(f"{path.name} has fewer than two numeric columns.")

    cycle_col = next((c for c in num_cols if str(c).lower() == "cycle"), None)
    if cycle_col is None:
        cycles = np.arange(1, len(frame) + 1, dtype=np.int32)
        sensor_candidates = num_cols
    else:
        cycles = frame[cycle_col].to_numpy(dtype=np.int32)
        sensor_candidates = [c for c in num_cols if c != cycle_col]

    if len(sensor_candidates) < 2:
        raise ValueError(f"{path.name} has fewer than two usable sensor columns.")

    out = pd.DataFrame(
        {
            "cycle": cycles,
            "sensor_0": frame[sensor_candidates[0]].to_numpy(dtype=np.float32),
            "sensor_1": frame[sensor_candidates[1]].to_numpy(dtype=np.float32),
        }
    )
    return out, cycles


def build_model(cfg: Dict, checkpoint_path: Path, device: torch.device, sensor_dim: int = 2) -> HybridRULModel:
    """Build and load hybrid model from checkpoint."""

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


def predict_sequence(
    model: HybridRULModel,
    signal: np.ndarray,
    cycles: np.ndarray,
    scaler: StandardScaler,
    window_size: int,
    device: torch.device,
    extract_attention: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """Run sliding-window inference for one sequence.

    Sliding window creation:
    - Use trailing windows of length ``window_size``.
    - For each window, run model inference and store cycle/RUL/HI.

    Health indicator calculation:
    - HI is taken from model output key ``hi`` per window.

    Optional attention extraction:
    - If enabled, attention maps (attn_map) are averaged over batch/head/time.

    Args:
        model: Trained model.
        signal: Raw signal shape (T,2) from loaded file.
        cycles: Cycle index shape (T,).
        scaler: Training scaler (ensures same normalization as training).
        window_size: Sliding window length.
        device: Torch device.
        extract_attention: Whether to aggregate attention heatmap.

    Returns:
        Tuple:
        - prediction dataframe with columns cycle,predicted_rul,health_indicator
        - averaged attention matrix or None
    """

    # Normalize with the SAME scaler used in training.
    signal_norm = scaler.transform(signal.astype(np.float32))

    model.eval()
    rows: List[Dict[str, float]] = []

    attn_sum: torch.Tensor | None = None
    attn_count = 0

    with torch.no_grad():
        for t in range(window_size, len(signal_norm) + 1):
            window = signal_norm[t - window_size : t]
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)

            # Model inference produces RUL prediction and HI score.
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

            if extract_attention and "attn_map" in out:
                attn = out["attn_map"].mean(dim=(0, 1)).detach().cpu()
                if attn_sum is None:
                    attn_sum = attn
                else:
                    attn_sum = attn_sum + attn
                attn_count += 1

    pred_df = pd.DataFrame(rows)
    if attn_sum is None or attn_count == 0:
        return pred_df, None
    return pred_df, (attn_sum / float(attn_count)).numpy().astype(np.float32)


def plot_rul_curve(pred_df: pd.DataFrame, save_path: Path) -> None:
    """Plot predicted RUL curve for one bearing file."""

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
    """Plot health indicator curve for one bearing file."""

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


def plot_attention_heatmap(attn: np.ndarray, save_path: Path) -> None:
    """Plot optional attention heatmap when attention is available."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn, cmap="mako", cbar=True)
    plt.title("Attention Heatmap")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main() -> None:
    """Entrypoint for directory-based online inference."""

    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        print(f"Input directory must be an absolute path: {input_dir}")
        return
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found in directory.")
        return

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading model from checkpoint...")
    device = get_device()

    # Prepare datasets to obtain the same scaler used during training.
    prepared = prepare_datasets(
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

        signal_df, cycles = load_signal_file(file_path)
        signal = signal_df[["sensor_0", "sensor_1"]].to_numpy(dtype=np.float32)
        print(f"Signal length: {len(signal)} cycles")

        pred_df, attn = predict_sequence(
            model=model,
            signal=signal,
            cycles=cycles,
            scaler=prepared.scaler,
            window_size=window_size,
            device=device,
            extract_attention=True,
        )

        base = file_path.stem
        csv_path = out_dir / f"{base}_prediction.csv"
        rul_png = out_dir / f"{base}_rul_curve.png"
        hi_png = out_dir / f"{base}_hi_curve.png"
        attn_png = out_dir / f"{base}_attention_heatmap.png"

        pred_df.to_csv(csv_path, index=False)
        plot_rul_curve(pred_df, rul_png)
        plot_hi_curve(pred_df, hi_png)
        if attn is not None:
            plot_attention_heatmap(attn, attn_png)

        final_rul = pred_df["predicted_rul"].iloc[-1] if len(pred_df) else float("nan")
        print(f"Predicted final RUL: {final_rul:.0f}")
        print(f"Results saved to outputs/online_predictions/{base}_*")


if __name__ == "__main__":
    main()
