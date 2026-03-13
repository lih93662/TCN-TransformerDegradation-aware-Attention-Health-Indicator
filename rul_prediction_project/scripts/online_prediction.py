"""PHM2012 online/batch inference script for RUL prediction.

This script supports batch inference over all CSV files in a directory and keeps
optional backward compatibility with single-file input.

Model family:
    TCN + Transformer + Degradation-aware Attention + Health Indicator (HI)

Main steps per file:
1) Load CSV and extract first two numeric sensor columns.
2) Rename channels to sensor_0 / sensor_1.
3) Normalize signals with the same scaler used in training.
4) Run sliding-window inference (window_size from config).
5) Save predictions and plots to outputs/online_predictions/.
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


# -----------------------------------------------------------------------------
# Argument parsing and path handling
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(description="PHM2012 online batch inference")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory containing CSV files (relative or absolute path).",
    )
    # Backward compatibility (optional).
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Optional single CSV file path for backward compatibility.",
    )
    return parser.parse_args()


def resolve_input_directory(input_dir: str | None) -> Path | None:
    """Resolve input directory to absolute path.

    Args:
        input_dir: CLI input directory string.

    Returns:
        Absolute Path if provided, otherwise None.
    """

    if input_dir is None:
        return None

    p = Path(input_dir)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

def load_signal_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one PHM2012 CSV and return normalized two-channel signal.

    Loading rules:
    - Read with pandas.
    - Keep numeric columns.
    - Use first two numeric sensor columns (excluding optional cycle column).
    - Rename conceptually to sensor_0 / sensor_1.

    NOTE:
    The returned signal is *raw* two-channel numeric data; normalization with the
    training scaler is applied in ``predict_sequence`` to guarantee consistency
    with training preprocessing.

    Args:
        path: CSV file path.

    Returns:
        Tuple:
            signal_raw: np.ndarray of shape (T, 2)
            cycles: np.ndarray of shape (T,)
    """

    frame = pd.read_csv(path)

    # Select numeric columns only so textual metadata does not break inference.
    numeric_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if len(numeric_cols) < 2:
        raise ValueError(f"{path.name} has fewer than two numeric columns.")

    # If cycle exists, keep it for plotting/indexing and exclude it from sensors.
    cycle_col = next((c for c in numeric_cols if str(c).lower() == "cycle"), None)
    if cycle_col is None:
        cycles = np.arange(1, len(frame) + 1, dtype=np.int32)
        sensor_candidates = numeric_cols
    else:
        cycles = frame[cycle_col].to_numpy(dtype=np.int32)
        sensor_candidates = [c for c in numeric_cols if c != cycle_col]

    if len(sensor_candidates) < 2:
        raise ValueError(f"{path.name} has fewer than two usable sensor columns.")

    # Rename semantic mapping to sensor_0/sensor_1 (kept in comments/logic).
    sensor_0 = frame[sensor_candidates[0]].to_numpy(dtype=np.float32)
    sensor_1 = frame[sensor_candidates[1]].to_numpy(dtype=np.float32)
    signal_raw = np.stack([sensor_0, sensor_1], axis=1)

    return signal_raw, cycles


# -----------------------------------------------------------------------------
# Model creation and inference
# -----------------------------------------------------------------------------

def build_model(cfg: Dict, checkpoint_path: Path, device: torch.device, sensor_dim: int = 2) -> HybridRULModel:
    """Build HybridRULModel and load checkpoint weights.

    Args:
        cfg: Configuration dictionary.
        checkpoint_path: Path to checkpoint file.
        device: Torch device.
        sensor_dim: Input channel count.

    Returns:
        Loaded model on ``device``.
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

    model = HybridRULModel(model_cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model


def predict_sequence(
    model: HybridRULModel,
    signal_raw: np.ndarray,
    cycles: np.ndarray,
    scaler: StandardScaler,
    window_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Predict RUL and HI over a full sequence using sliding windows.

    Sliding window logic:
        for t in range(window_size, T + 1):
            window = signal_norm[t-window_size:t]

    Inference outputs:
    - predicted_rul from model['pred']
    - health_indicator from model['hi']
    - optional attention matrix (averaged over batch/head)

    Args:
        model: Loaded hybrid model.
        signal_raw: Raw two-channel signal, shape (T, 2).
        cycles: Cycle index array, shape (T,).
        scaler: Training scaler for consistent normalization.
        window_size: Sliding window size.
        device: Torch device.

    Returns:
        Tuple:
            prediction dataframe (cycle, predicted_rul, health_indicator)
            averaged attention heatmap matrix
    """

    # Normalize with training scaler (same normalization policy as training).
    signal = scaler.transform(signal_raw.astype(np.float32))

    rows: List[Dict[str, float]] = []
    attn_sum: torch.Tensor | None = None
    attn_count = 0

    model.eval()
    with torch.no_grad():
        for t in range(window_size, len(signal) + 1):
            # Build trailing temporal window ending at cycle t.
            window = signal[t - window_size : t]
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)

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

            # Extract/aggregate attention if available in output dictionary.
            if "attn_map" in out:
                attn = out["attn_map"].mean(dim=(0, 1)).detach().cpu()
                if attn_sum is None:
                    attn_sum = attn
                else:
                    attn_sum = attn_sum + attn
                attn_count += 1

    pred_df = pd.DataFrame(rows)

    if attn_sum is None or attn_count == 0:
        attn_avg = np.eye(window_size, dtype=np.float32)
    else:
        attn_avg = (attn_sum / float(attn_count)).numpy().astype(np.float32)

    return pred_df, attn_avg


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_rul_curve(results: pd.DataFrame, save_path: Path) -> None:
    """Plot predicted RUL vs cycle.

    Args:
        results: Prediction dataframe.
        save_path: Output PNG path.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(results["cycle"], results["predicted_rul"], linewidth=2.0)
    plt.xlabel("cycle")
    plt.ylabel("predicted_rul")
    plt.title("RUL Prediction Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_hi_curve(results: pd.DataFrame, save_path: Path) -> None:
    """Plot health indicator vs cycle.

    Args:
        results: Prediction dataframe.
        save_path: Output PNG path.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(results["cycle"], results["health_indicator"], color="purple", linewidth=2.0)
    plt.xlabel("cycle")
    plt.ylabel("health_indicator")
    plt.title("Health Indicator Curve")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_attention_heatmap(attentions: np.ndarray, save_path: Path) -> None:
    """Plot attention heatmap.

    Args:
        attentions: Attention matrix.
        save_path: Output PNG path.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attentions, cmap="mako", cbar=True)
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.title("Attention Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# Main application flow
# -----------------------------------------------------------------------------

def _discover_csv_files(input_dir: Path) -> List[Path]:
    """List CSV files directly under input directory (non-recursive)."""

    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def main() -> None:
    """Run online batch inference for all discovered CSV files."""

    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    # Resolve input directory (supports both relative and absolute input).
    input_dir = resolve_input_directory(args.input_dir)

    # Backward compatibility: allow single-file mode if --input_file is supplied.
    single_file = Path(args.input_file).resolve() if args.input_file else None

    if input_dir is None and single_file is None:
        print("Please provide --input_dir (or --input_file for backward compatibility).")
        return

    file_list: List[Path] = []
    if input_dir is not None:
        if not input_dir.exists():
            print(f"Input directory not found: {input_dir}")
            return
        file_list = _discover_csv_files(input_dir)
        if not file_list:
            print("No CSV files found in directory.")
            return
    elif single_file is not None:
        if not single_file.exists():
            print(f"Input file not found: {single_file}")
            return
        file_list = [single_file]

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading model from checkpoint...")
    device = get_device()

    # Prepare datasets once to obtain scaler fitted from training split.
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

    for csv_file in file_list:
        print(f"Processing file: {csv_file.name}")

        try:
            signal_raw, cycles = load_signal_file(csv_file)
        except Exception as exc:
            print(f"[Warning] Failed to read {csv_file.name}: {exc}")
            continue

        print(f"Signal length: {len(signal_raw)}")

        try:
            results, attentions = predict_sequence(
                model=model,
                signal_raw=signal_raw,
                cycles=cycles,
                scaler=prepared.scaler,
                window_size=window_size,
                device=device,
            )
        except Exception as exc:
            print(f"[Warning] Inference failed for {csv_file.name}: {exc}")
            continue

        bearing_name = csv_file.stem
        pred_csv = out_dir / f"{bearing_name}_prediction.csv"
        rul_png = out_dir / f"{bearing_name}_rul_curve.png"
        hi_png = out_dir / f"{bearing_name}_hi_curve.png"
        attn_png = out_dir / f"{bearing_name}_attention_heatmap.png"

        # Save per-cycle predictions.
        results.to_csv(pred_csv, index=False)

        # Save visualizations for RUL, HI, and attention.
        plot_rul_curve(results, rul_png)
        plot_hi_curve(results, hi_png)
        plot_attention_heatmap(attentions, attn_png)

        final_rul = float(results["predicted_rul"].iloc[-1]) if len(results) else float("nan")
        print(f"Predicted final RUL: {final_rul:.0f} cycles")
        print(f"Results saved to outputs/online_predictions/{bearing_name}_*")

    print("Results saved to outputs/online_predictions/")


if __name__ == "__main__":
    main()
