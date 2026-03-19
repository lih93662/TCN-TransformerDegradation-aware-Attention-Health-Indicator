"""Detect near-constant predictions from saved raw prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check raw prediction CSV for collapse")
    parser.add_argument("csv_path", type=str, help="Path to raw prediction CSV")
    parser.add_argument("--std-threshold", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv_path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        preds = [float(row["pred_rul"]) for row in reader]

    if not preds:
        raise ValueError("No pred_rul values found in CSV.")

    mean = sum(preds) / len(preds)
    var = sum((x - mean) ** 2 for x in preds) / len(preds)
    std = math.sqrt(var)

    print(f"Prediction mean: {mean:.8f}")
    print(f"Prediction std: {std:.8f}")
    if std < args.std_threshold:
        print("[Warning] Predictions appear collapsed / nearly constant.")
    else:
        print("Predictions show non-trivial variation.")


if __name__ == "__main__":
    main()
