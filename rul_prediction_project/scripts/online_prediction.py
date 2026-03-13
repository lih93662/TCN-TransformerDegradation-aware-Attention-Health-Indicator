"""PHM2012 batch online inference script for RUL prediction.

This script performs batch inference over CSV files in an input directory using
an already trained hybrid model:

TCN + Transformer + Degradation-aware Attention + Health Indicator.

Design constraints implemented here:
- Accept only ``--input_dir``.
- Input directory must be provided as an absolute path.
- Process all CSV files in the directory tree.
- Save per-file CSV outputs and visualizations to outputs/online_predictions/.
"""

from __future__ import annotations

import argparse
import os
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
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description="PHM2012 RUL batch online inference")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to configuration YAML.",
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
        help="Absolute path to PHM2012 test directory.",
    )
    return parser.parse_args()


def discover_csv_files(input_dir: str) -> List[Path]:
    """Traverse input directory and return all CSV files.

    Path handling policy:
    - use ``os.path.abspath`` for normalization
    - require the original user-provided path to be absolute

    Args:
        input_dir: User input directory string.

    Returns:
        Sorted list of CSV file paths.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If path is not absolute.
    """

    if not os.path.isabs(input_dir):
        raise ValueError(f"Input directory must be an absolute path: {input_dir}")

    normalized = os.path.abspath(input_dir)
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"Input directory not found: {normalized}")

    csv_files: List[Path] = []
    for root, _, files in os.walk(normalized):
        for name in files:
            if name.lower().endswith(".csv"):
                csv_files.append(Path(root) / name)
            # non-CSV files are intentionally skipped

    csv_files.sort(key=lambda p: str(p))
    return csv_files


def load_signal_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one CSV file and extract two sensor channels.

    Signal loading details:
    1. Read CSV with pandas.
    2. Keep numeric columns only.
    3. Use optional ``cycle`` column when available, otherwise create one.
    4. Select first two numeric sensor columns.
    5. Conceptually rename channels to ``sensor_0`` and ``sensor_1``.

    Args:
        path: CSV file path.

    Returns:
        Tuple of:
        - signal_raw: numpy array shape (T, 2)
        - cycles: numpy array shape (T,)
    """

    frame = pd.read_csv(path)

    numeric_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if len(numeric_cols) < 2:
        raise ValueError(f"{path.name} has fewer than two numeric columns.")

    cycle_col = next((c for c in numeric_cols if str(c).lower() == "cycle"), None)
    if cycle_col is None:
        cycles = np.arange(1, len(frame) + 1, dtype=np.int32)
        sensor_candidates = numeric_cols
    else:
        cycles = frame[cycle_col].to_numpy(dtype=np.int32)
        sensor_candidates = [c for c in numeric_cols if c != cycle_col]

    if len(sensor_candidates) < 2:
        raise ValueError(f"{path.name} has fewer than two usable sensor columns.")

    sensor_0 = frame[sensor_candidates[0]].to_numpy(dtype=np.float32)
    sensor_1 = frame[sensor_candidates[1]].to_numpy(dtype=np.float32)
    signal_raw = np.stack([sensor_0, sensor_1], axis=1)

    return signal_raw, cycles


def build_model(cfg: Dict, checkpoint_path: Path, device: torch.device) -> HybridRULModel:
    """Build hybrid model and load checkpoint state."""

    m = cfg["model"]
    model_cfg = ModelConfig(
        sensor_dim=2,
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
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run sliding-window inference and collect RUL/HI/attention.

    Sliding-window process:
    - Normalize with training scaler (same normalization as training).
    - For each cycle position t >= window_size, build trailing window and run
      model forward pass.

    Model inference outputs:
    - ``pred`` for Remaining Useful Life.
    - ``hi`` for Health Indicator.
    - ``attn_map`` optionally used for heatmap generation.

    Args:
        model: Loaded model.
        signal: Raw signal array (T,2).
        cycles: Cycle indices (T,).
        scaler: Training scaler.
        window_size: Window size from config.
        device: Torch device.

    Returns:
        Tuple:
        - prediction dataframe with columns cycle,predicted_rul,health_indicator
        - averaged attention matrix
    """

    signal_norm = scaler.transform(signal.astype(np.float32))

    rows: List[Dict[str, float]] = []
    attn_sum: torch.Tensor | None = None
    attn_count = 0

    model.eval()
    with torch.no_grad():
        for t in range(window_size, len(signal_norm) + 1):
            window = signal_norm[t - window_size : t]
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

            if "attn_map" in out:
                attn = out["attn_map"].mean(dim=(0, 1)).detach().cpu()
                attn_sum = attn if attn_sum is None else (attn_sum + attn)
                attn_count += 1

    pred_df = pd.DataFrame(rows)

    if attn_sum is None or attn_count == 0:
        attn_avg = np.eye(window_size, dtype=np.float32)
    else:
        attn_avg = (attn_sum / float(attn_count)).numpy().astype(np.float32)

    return pred_df, attn_avg


def plot_rul_curve(results: pd.DataFrame, save_path: Path) -> None:
    """Plot predicted RUL vs cycle."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(results["cycle"], results["predicted_rul"], linewidth=2)
    plt.xlabel("cycle")
    plt.ylabel("predicted_rul")
    plt.title("RUL Prediction Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_hi_curve(results: pd.DataFrame, save_path: Path) -> None:
    """Plot health indicator vs cycle."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(results["cycle"], results["health_indicator"], color="purple", linewidth=2)
    plt.xlabel("cycle")
    plt.ylabel("health_indicator")
    plt.title("Health Indicator Curve")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_attention_heatmap(attentions: np.ndarray, save_path: Path) -> None:
    """Plot attention heatmap vs cycle/window indices."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attentions, cmap="mako", cbar=True)
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.title("Attention Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main() -> None:
    """Main function for PHM2012 batch inference."""

    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    try:
        csv_files = discover_csv_files(args.input_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        return
    except ValueError as exc:
        print(str(exc))
        return

    if not csv_files:
        print("No CSV files found in directory.")
        return

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading model from checkpoint...")
    device = get_device()

    # Prepare once to obtain the same scaler used in training.
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

    model = build_model(cfg, checkpoint, device)
    window_size = int(cfg["data"].get("window_size", 40))

    out_dir = ROOT / "outputs" / "online_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        print(f"Processing file: {csv_file.name}")

        try:
            signal, cycles = load_signal_file(csv_file)
        except Exception as exc:
            print(f"[Warning] Failed to read {csv_file.name}: {exc}")
            continue

        print(f"Signal length: {len(signal)}")

        try:
            results, attentions = predict_sequence(
                model=model,
                signal=signal,
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

        results.to_csv(pred_csv, index=False)
        plot_rul_curve(results, rul_png)
        plot_hi_curve(results, hi_png)
        plot_attention_heatmap(attentions, attn_png)

        final_rul = float(results["predicted_rul"].iloc[-1]) if len(results) else float("nan")
        print(f"Predicted final RUL: {final_rul:.0f} cycles")
        print("Results saved to outputs/online_predictions/")


if __name__ == "__main__":
    main()
