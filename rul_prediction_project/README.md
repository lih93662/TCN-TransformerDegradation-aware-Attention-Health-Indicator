# PHM2012 RUL Prediction Research Project

A research-oriented PyTorch implementation for Remaining Useful Life (RUL) prediction on the PHM 2012 IEEE Prognostics Challenge bearing dataset.

## Architecture

This repository implements the required hybrid model:

- **Temporal Convolutional Network (TCN)** for local temporal feature extraction.
- **Transformer Encoder** for global sequence dependency learning.
- **Degradation-Aware Attention** with HI-conditioned temporal bias.
- **Health Indicator (HI)** learned from statistical vibration features (RMS, variance, kurtosis, skewness).
- **Fusion + Regression Head** for final RUL prediction.

## Folder Structure

```text
rul_prediction_project/
├── src/
├── configs/
├── scripts/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── logs/
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

This will:

1. Prepare standardized sliding windows (`window_size=40`, `stride=1`).
2. Train for 50 epochs with Adam (`lr=3e-4`, `batch_size=64`).
3. Save best checkpoint to `outputs/checkpoints/best_model.pth`.
4. Save logs/metrics and generate training loss plot.

## Evaluation

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Metrics:

- RMSE
- MAE
- PHM 2012 Score

## Plotting

```bash
python scripts/plot_results.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Generated figures:

- `training_loss_curve.png`
- `rul_prediction_curve.png`
- `health_indicator_curve.png`
- `example_bearing_prediction.png`

## Notes

- GPU is automatically used if available.
- All modules include type hints and docstrings.
- Reproducible seed setup is integrated.
