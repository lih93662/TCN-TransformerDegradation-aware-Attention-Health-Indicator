"""PHM2012 batch online inference script for RUL prediction.

This script supports batch inference over CSV files under an input directory and
(optional) single-file backward compatibility.

Model family:
    TCN + Transformer + Degradation-aware Attention + Health Indicator (HI)

Workflow per file:
1. Load CSV and extract two sensor channels (sensor_0, sensor_1).
2. Normalize with the same scaler used during training.
3. Run sliding-window inference to predict RUL and HI.
4. Aggregate attention maps (if available).
5. Save CSV + figures under outputs/online_predictions/.
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
from src.checkpoint_compat import (
    detect_checkpoint_head_variant,
    load_model_state_strict,
    resolve_rul_head_mode,
)
from src.preprocess import StandardScaler, prepare_datasets
from src.utils import get_device, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI namespace.
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
        help="Input directory containing CSV files (absolute or relative path).",
    )
    # Optional backward compatibility with earlier script usage.
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Optional single CSV input file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print window/attention diagnostics for debugging constant predictions.",
    )
    return parser.parse_args()


def resolve_input_dir(path_str: str) -> str:
    """Resolve input directory to absolute path.

    Resolution order for relative paths:
    1) Current working directory
    2) Project root directory (ROOT)

    Args:
        path_str: Input directory path (relative or absolute).

    Returns:
        Absolute path string.
    """

    if os.path.isabs(path_str):
        return path_str

    from_cwd = os.path.abspath(path_str)
    if os.path.exists(from_cwd):
        return from_cwd

    from_root = str((ROOT / path_str).resolve())
    return from_root


def discover_csv_files(input_dir: str) -> List[Path]:
    """Traverse directory tree and collect all CSV files.

    Args:
        input_dir: Input directory string.

    Returns:
        Sorted list of CSV paths.

    Raises:
        FileNotFoundError: If directory does not exist.
    """

    abs_dir = resolve_input_dir(input_dir)
    if not os.path.exists(abs_dir):
        raise FileNotFoundError(f"Input directory not found: {abs_dir}")

    csv_files: List[Path] = []
    for root, _, files in os.walk(abs_dir):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_files.append(Path(root) / file_name)
            # Non-CSV files are intentionally skipped.

    csv_files.sort(key=lambda p: str(p))
    return csv_files


def load_signal_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one CSV file and extract two numeric sensor channels.

    Signal loading logic:
    - Read CSV with pandas.
    - Keep only numeric columns.
    - Detect optional cycle column.
    - Select first two numeric sensor columns.
    - Map selected channels to semantic names: sensor_0 and sensor_1.

    Args:
        path: CSV file path.

    Returns:
        Tuple:
            signal_raw: Array (T, 2)
            cycles: Array (T,)
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

    # Rename semantics to sensor_0 and sensor_1.
    sensor_0 = frame[sensor_candidates[0]].to_numpy(dtype=np.float32)
    sensor_1 = frame[sensor_candidates[1]].to_numpy(dtype=np.float32)
    signal_raw = np.stack([sensor_0, sensor_1], axis=1)

    return signal_raw, cycles


def build_model(cfg: Dict, checkpoint_path: Path, device: torch.device) -> HybridRULModel:
    """Instantiate and load the hybrid model from checkpoint."""

    m = cfg["model"]
    payload = torch.load(checkpoint_path, map_location=device)
    checkpoint_head_variant = detect_checkpoint_head_variant(payload["model_state"])
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
        rul_head_mode=resolve_rul_head_mode(
            m,
            default="feature_only",
            checkpoint_variant=checkpoint_head_variant,
        ),
        use_hi_auxiliary=bool(m.get("use_hi_auxiliary", m.get("use_hi", True))),
        use_hi_in_rul_head=bool(m.get("use_hi_in_rul_head", False)),
        use_degradation_attention=bool(m.get("use_degradation_attention", m.get("use_attention", True))),
    )

    model = HybridRULModel(model_cfg).to(device)
    load_model_state_strict(model, payload)
    return model


def predict_sequence(
    model: HybridRULModel,
    signal: np.ndarray,
    cycles: np.ndarray,
    scaler: StandardScaler,
    window_size: int,
    device: torch.device,
    debug: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run sliding-window inference for one signal sequence.

    Sliding-window creation:
    - Normalize raw signal with training scaler.
    - For each t in [window_size, T], use window signal[t-window_size:t].

    Model inference:
    - Predict RUL via output key ``pred``.
    - Predict Health Indicator via output key ``hi``.
    - Optionally aggregate attention from output key ``attn_map``.

    Args:
        model: Loaded model.
        signal: Raw signal array (T,2).
        cycles: Cycle indices (T,).
        scaler: Training scaler.
        window_size: Window size from config.
        device: Torch device.

    Returns:
        Tuple:
            results dataframe (cycle,predicted_rul,health_indicator)
            attention heatmap matrix
    """

    signal_norm = scaler.transform(signal.astype(np.float32))

    if debug and len(signal_norm) >= window_size:
        first_window = signal_norm[:window_size]
        print(
            f"[Debug] first window shape={first_window.shape}, "
            f"sensor0(min,max)=({first_window[:,0].min():.4f},{first_window[:,0].max():.4f}), "
            f"sensor1(min,max)=({first_window[:,1].min():.4f},{first_window[:,1].max():.4f})"
        )

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
                attn_map = out["attn_map"]
                if debug and (t == window_size or t == len(signal_norm)):
                    print(
                        f"[Debug] cycle={int(cycles[t-1])} attention min/max="
                        f"({attn_map.min().item():.6f}, {attn_map.max().item():.6f})"
                    )
                attn = attn_map.mean(dim=(0, 1)).detach().cpu()
                attn_sum = attn if attn_sum is None else (attn_sum + attn)
                attn_count += 1

    results = pd.DataFrame(rows)
    if len(results):
        hi = results["health_indicator"].to_numpy(dtype=np.float32)
        hi_min, hi_max = float(np.min(hi)), float(np.max(hi))
        if hi_max > hi_min:
            results["health_indicator"] = (hi - hi_min) / (hi_max - hi_min)
    if attn_sum is None or attn_count == 0:
        attn_avg = np.eye(window_size, dtype=np.float32)
    else:
        attn_avg = (attn_sum / float(attn_count)).numpy().astype(np.float32)

    return results, attn_avg


def plot_rul_curve(results: pd.DataFrame, save_path: Path) -> None:
    """Plot predicted RUL vs cycle and save figure."""

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
    """Plot HI vs cycle and save figure."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(results["cycle"], results["health_indicator"], linewidth=2, color="purple")
    plt.xlabel("cycle")
    plt.ylabel("health_indicator")
    plt.title("Health Indicator Curve")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_attention_heatmap(attentions: np.ndarray, save_path: Path) -> None:
    """Plot attention heatmap and save figure."""

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
    """Batch process all CSV files from input directory (or single file fallback)."""

    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))

    file_list: List[Path] = []

    # Preferred mode: directory inference.
    if args.input_dir is not None:
        try:
            file_list = discover_csv_files(args.input_dir)
        except FileNotFoundError as exc:
            print(str(exc))
            return

        if not file_list:
            print("No CSV files found in directory.")
            return

    # Optional backward compatibility with single-file input.
    elif args.input_file is not None:
        single = Path(args.input_file)
        if not single.is_absolute() and not single.exists():
            single = ROOT / single
        if not single.exists():
            print(f"Input file not found: {single}")
            return
        file_list = [single]
    else:
        print("Please provide --input_dir (or --input_file for backward compatibility).")
        return

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading model from checkpoint...")
    device = get_device()

    # Build training-consistent scaler once.
    prepared = prepare_datasets(
        dataset_root=cfg["data"]["dataset_root"],
        window_size=int(cfg["data"].get("window_size", 40)),
        stride=int(cfg["data"].get("stride", 10)),
        max_rul=int(cfg["data"].get("max_rul", 125)),
        valid_ratio=float(cfg["data"].get("valid_ratio", 0.2)),
        seed=int(cfg["experiment"].get("seed", 42)),
        sensor_dim=2,
        max_windows_per_bearing=int(cfg["data"].get("max_windows_per_bearing", 20000)),
    )

    model = build_model(cfg, checkpoint, device)
    window_size = int(cfg["data"].get("window_size", 40))

    out_dir = ROOT / "outputs" / "online_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in file_list:
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
                debug=args.debug,
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

        first_rul = float(results["predicted_rul"].iloc[0]) if len(results) else float("nan")
        final_rul = float(results["predicted_rul"].iloc[-1]) if len(results) else float("nan")
        print(f"Predicted RUL first/last: {first_rul:.0f} -> {final_rul:.0f} cycles")
        print("Results saved to outputs/online_predictions/")


if __name__ == "__main__":
    main()
