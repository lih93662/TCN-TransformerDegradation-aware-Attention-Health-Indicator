# PHM2012 RUL Prediction Research Project

A research-oriented PyTorch implementation for Remaining Useful Life (RUL) prediction on the PHM 2012 IEEE Prognostics Challenge bearing dataset.

## Architecture

This repository implements a modular hybrid model family:

- **Temporal Convolutional Network (TCN)** for local temporal features.
- **Transformer Encoder** for global sequence modeling.
- **Degradation-Aware Attention** with HI-conditioned temporal bias.
- **Health Indicator (HI)** learned from statistical vibration features.
- **Multimodal Attention Fusion** across time/frequency/deep/HI features.

## Advanced Modules Added

- `src/features/time_frequency.py`
  - STFT feature extraction
  - CWT feature extraction
  - Time-frequency fusion/projector utilities
- `src/data/data_augmentation.py`
  - Gaussian noise
  - Time warping
  - Window slicing
  - Mixup
- `src/models/multimodal_fusion.py`
  - Attention-based multimodal fusion
  - HI-conditioned modality weighting
  - Fusion interpretability outputs
- `scripts/online_prediction.py`
  - Online/streaming RUL simulation pipeline
- `scripts/run_benchmark.py`
  - Multi-model benchmark experiments and CSV export
- `scripts/generate_paper_figures.py`
  - Publication-style figure generation under `outputs/paper_figures/`

## Folder Structure

```text
rul_prediction_project/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── ...
├── configs/
├── scripts/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   ├── paper_figures/
│   └── results/
├── requirements.txt
└── README.md
```

## Dataset

Default absolute dataset path used by the project:

```text
C:\peng\RULdata\ieee-phm-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset-master
```

The dataset is read directly from this location. No dataset duplication is performed.

## Training

```bash
python scripts/train.py --config configs/config.yaml
```

Training diagnostics are exported to:
- `outputs/logs/batch_loss_log.csv` (per-batch train/valid loss + LR)
- `outputs/logs/attention_epoch_stats.csv` (attention min/max/mean/std by epoch)
- `outputs/logs/attention_maps/epoch_*_{train|valid}.npy` (epoch-average attention maps)
- training stability knobs in `configs/config.yaml`: `grad_clip_norm`, `scheduler_factor`, `scheduler_patience`, `weight_decay`

## Evaluation

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

## Benchmark

```bash
python scripts/run_benchmark.py --config configs/config.yaml
```

Outputs:

- `outputs/results/benchmark_results.csv`
- `outputs/benchmark_results.png`

## Ablation Study

Legacy A/B/C/D ablations are preserved, and an extended publication-ready matrix
is available for optimization, HI pathway, attention internals, and backbone
comparisons.

See full matrix and commands:

- `docs/ablation_study.md`

Quick start:

```bash
python scripts/run_ablation.py --list
python scripts/run_ablation.py --suite legacy
python scripts/run_ablation.py --suite optimization
python scripts/run_ablation.py --suite all --dry-run
```

## Data Split & Sampling Experiments

Data-side controls live in `configs/config.yaml`:

- `valid_bearing_ids` / `valid_split_variant`: run-level validation split selection (anti-leakage preserved).
- `balance_train_rul_bins`: enable training-only RUL-stage balancing.
- `rul_bin_edges`: normalized RUL bins (default `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, plus open-ended final bin).
- `train_balance_mode`: `oversample | downsample | hybrid`.
- `train_balance_target`: `auto | min | median | max | custom`.
- `train_balance_custom_count`: target count per bin when `custom`.
- `train_balance_seed`: balancing seed (falls back to experiment seed).

Ready-to-run presets for data comparisons:

```bash
python scripts/train.py --config configs/data_compare/baseline_data.yaml
python scripts/train.py --config configs/data_compare/balanced_train.yaml
python scripts/train.py --config configs/data_compare/alt_valid_split.yaml
python scripts/train.py --config configs/data_compare/balanced_train_alt_valid.yaml
```

## Online Prediction

Single file:

```bash
python scripts/online_prediction.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth --input_file /path/to/new_signal.csv
```

PHM2012 directory (recommended):

```bash
python scripts/online_prediction.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth --input_dir Full_Test_Set
```

Debug mode for constant-prediction diagnosis:

```bash
python scripts/online_prediction.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth --input_dir Full_Test_Set --debug
```

Outputs are saved to `outputs/online_predictions/` per bearing:
- `*_prediction.csv`
- `*_rul_curve.png`
- `*_hi_curve.png`
- `*_attention_heatmap.png`

## Paper Figures

```bash
python scripts/generate_paper_figures.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Generated files in `outputs/paper_figures/`:

- `model_architecture.png`
- `health_indicator_curve.png`
- `rul_prediction_curve.png`
- `attention_heatmap.png`
- `sensor_importance.png`
- `ablation_results.png`

## Notes

- GPU is automatically used if available.
- Random seed setup is integrated.
- Checkpoint saved as `outputs/checkpoints/best_model.pth`.
- Metrics include RMSE, MAE, and PHM score.
