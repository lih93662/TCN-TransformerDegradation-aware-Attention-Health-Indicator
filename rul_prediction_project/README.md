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
