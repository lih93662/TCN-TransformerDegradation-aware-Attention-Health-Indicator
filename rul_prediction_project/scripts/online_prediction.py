"""Online PHM2012 inference script.

This script supports two modes:
1. Single-file mode via ``--input_file``.
2. Directory mode via ``--input_dir`` (recommended for PHM2012 Full_Test_Set).

For each bearing file, the script:
- parses PHM2012 numeric channels,
- maps selected channels to ``sensor_0`` and ``sensor_1``,
- runs sliding-window online inference,
- predicts RUL + Health Indicator per cycle,
- aggregates attention maps,
- saves CSV and plots under ``outputs/online_predictions/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_rul_model import HybridRULModel, ModelConfig
from src.preprocess import StandardScaler, prepare_datasets
from src.utils import configure_logging, ensure_project_paths, get_device, load_yaml, set_seed
from src.visualization import plot_attention, plot_hi_curve, plot_rul_curve


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for online inference."""

    parser = argparse.ArgumentParser(description="PHM2012 online RUL prediction")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument("--input_file", type=str, default=None, help="Single PHM2012 CSV/TXT file")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing PHM2012 test CSV/TXT files")
    return parser.parse_args()


def _read_numeric_file(path: Path) -> pd.DataFrame:
    """Read one PHM2012 file and keep numeric columns only."""

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, sep=None, engine="python")

    num_cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    frame = frame[num_cols].copy()

    # Normalize/create cycle column for consistent online indexing.
    cycle_cols = [c for c in frame.columns if str(c).lower() == "cycle"]
    if cycle_cols:
        if cycle_cols[0] != "cycle":
            frame = frame.rename(columns={cycle_cols[0]: "cycle"})
    else:
        frame.insert(0, "cycle", np.arange(1, len(frame) + 1, dtype=np.int32))

    frame = frame.sort_values("cycle").reset_index(drop=True)
    return frame


def parse_phm2012_bearing_file(path: Path, sensor_dim: int = 2) -> pd.DataFrame:
    """Parse PHM2012 bearing file and map channels to canonical names.

    Args:
        path: Bearing CSV/TXT file path.
        sensor_dim: Number of vibration channels to keep.

    Returns:
        DataFrame with columns: ``cycle``, ``sensor_0``, ``sensor_1`` (for default sensor_dim=2).
    """

    raw = _read_numeric_file(path)

    sensor_candidates = [c for c in raw.columns if str(c).lower() not in {"cycle", "time"}]
    if len(sensor_candidates) < sensor_dim:
        raise ValueError(
            f"File {path.name} has only {len(sensor_candidates)} sensor columns; required {sensor_dim}."
        )

    selected = sensor_candidates[:sensor_dim]
    out = pd.DataFrame()
    out["cycle"] = raw["cycle"].to_numpy(dtype=np.int32)
    for i, c in enumerate(selected):
        out[f"sensor_{i}"] = raw[c].to_numpy(dtype=np.float32)

    return out


def discover_input_files(input_file: str | None, input_dir: str | None) -> List[Path]:
    """Collect input files from CLI options.

    Priority:
    - If ``input_dir`` is provided, process all CSV/TXT files inside recursively.
    - Otherwise process single ``input_file``.
    """

    files: List[Path] = []
    if input_dir:
        root = Path(input_dir)
        if not root.exists():
            raise FileNotFoundError(f"Input directory not found: {root}")
        files = sorted([*root.rglob("*.csv"), *root.rglob("*.txt")], key=lambda p: str(p))
    elif input_file:
        p = Path(input_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        files = [p]
    else:
        raise ValueError("Please provide either --input_file or --input_dir")

    if not files:
        raise RuntimeError("No input files discovered for online inference.")
    return files


def build_model_from_config(cfg: Dict, sensor_dim: int, checkpoint_path: Path, device: torch.device) -> HybridRULModel:
    """Instantiate and load model checkpoint."""

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
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model


def sliding_window_online_inference(
    model: HybridRULModel,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run sliding-window online predictions over one bearing sequence.

    Args:
        model: Trained hybrid model.
        frame: Parsed bearing dataframe with ``cycle`` and ``sensor_i`` columns.
        scaler: Standard scaler from training data.
        window_size: Window length.
        device: Torch device.

    Returns:
        Tuple:
        - predictions dataframe with columns ``cycle``, ``predicted_rul``, ``health_indicator``.
        - true_rul vector aligned with predictions.
        - averaged attention matrix.
    """

    sensor_cols = [c for c in frame.columns if c.startswith("sensor_")]
    signal = frame[sensor_cols].to_numpy(dtype=np.float32)
    signal = scaler.transform(signal)
    cycles = frame["cycle"].to_numpy(dtype=np.int32)

    model.eval()
    rows: List[Dict[str, float]] = []
    attn_sum: torch.Tensor | None = None
    attn_count = 0

    # Online inference: at each cycle t>=window_size, use trailing window [t-window_size, t).
    with torch.no_grad():
        for t in range(window_size, len(signal) + 1):
            window = signal[t - window_size : t]
            x = torch.from_numpy(window).unsqueeze(0).float().to(device)
            out = model(x)

            pred_rul = float(out["pred"].item())
            hi = float(out["hi"].item())
            cycle = int(cycles[t - 1])

            rows.append(
                {
                    "cycle": cycle,
                    "predicted_rul": pred_rul,
                    "health_indicator": hi,
                }
            )

            attn = out["attn_map"].mean(dim=(0, 1)).detach().cpu()
            if attn_sum is None:
                attn_sum = attn
            else:
                attn_sum = attn_sum + attn
            attn_count += 1

    pred_df = pd.DataFrame(rows)

    # Construct true RUL from file-end assumption for visualization.
    max_cycle = int(cycles.max())
    true_rul = np.asarray([max_cycle - int(c) for c in pred_df["cycle"].tolist()], dtype=np.float32)

    if attn_sum is None or attn_count == 0:
        attn_avg = np.eye(1, dtype=np.float32)
    else:
        attn_avg = (attn_sum / float(attn_count)).numpy().astype(np.float32)

    return pred_df, true_rul, attn_avg


def process_bearing_file(
    file_path: Path,
    model: HybridRULModel,
    scaler: StandardScaler,
    window_size: int,
    sensor_dim: int,
    device: torch.device,
    output_dir: Path,
    logger,
) -> None:
    """Process one bearing file end-to-end and save CSV + figures."""

    logger.info("Processing bearing file: %s", file_path.name)
    frame = parse_phm2012_bearing_file(file_path, sensor_dim=sensor_dim)
    logger.info("Total cycles: %d", len(frame))

    pred_df, true_rul, attn_avg = sliding_window_online_inference(
        model=model,
        frame=frame,
        scaler=scaler,
        window_size=window_size,
        device=device,
    )

    bearing_name = file_path.stem
    csv_path = output_dir / f"{bearing_name}_prediction.csv"
    pred_df.to_csv(csv_path, index=False)

    # Generate required visualizations per bearing.
    plot_rul_curve(true_rul, pred_df["predicted_rul"].to_numpy(dtype=np.float32), output_dir / f"{bearing_name}_rul_curve.png")
    plot_hi_curve(pred_df["health_indicator"].to_numpy(dtype=np.float32), output_dir / f"{bearing_name}_hi_curve.png")
    plot_attention(attn_avg, output_dir / f"{bearing_name}_attention_heatmap.png")

    final_rul = float(pred_df["predicted_rul"].iloc[-1]) if len(pred_df) else float("nan")
    logger.info("Predicted final RUL: %.3f", final_rul)


def main() -> None:
    """Main workflow for online inference over file or directory input."""

    args = parse_args()
    cfg = load_yaml(args.config)
    paths = ensure_project_paths(ROOT)

    online_dir = paths["outputs"] / "online_predictions"
    online_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(paths["logs"] / "online_prediction.log")
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    # Build scaler from training data preprocessing so online input is normalized identically.
    data_cfg = cfg["data"]
    prepared = prepare_datasets(
        dataset_root=data_cfg["dataset_root"],
        window_size=int(data_cfg.get("window_size", 40)),
        stride=int(data_cfg.get("stride", 10)),
        max_rul=int(data_cfg.get("max_rul", 125)),
        valid_ratio=float(data_cfg.get("valid_ratio", 0.2)),
        seed=seed,
        sensor_dim=int(data_cfg.get("sensor_dim", 2)),
        max_windows_per_bearing=int(data_cfg.get("max_windows_per_bearing", 20000)),
    )

    device = get_device()
    sensor_dim = int(data_cfg.get("sensor_dim", prepared.feature_dim))
    window_size = int(data_cfg.get("window_size", 40))

    model = build_model_from_config(
        cfg=cfg,
        sensor_dim=sensor_dim,
        checkpoint_path=Path(args.checkpoint),
        device=device,
    )

    input_files = discover_input_files(args.input_file, args.input_dir)
    logger.info("Discovered %d file(s) for online inference", len(input_files))

    for f in input_files:
        try:
            process_bearing_file(
                file_path=f,
                model=model,
                scaler=prepared.scaler,
                window_size=window_size,
                sensor_dim=sensor_dim,
                device=device,
                output_dir=online_dir,
                logger=logger,
            )
        except Exception as exc:
            logger.exception("Failed to process %s: %s", f, exc)

    logger.info("Saved results to: %s", online_dir)


if __name__ == "__main__":
    main()
