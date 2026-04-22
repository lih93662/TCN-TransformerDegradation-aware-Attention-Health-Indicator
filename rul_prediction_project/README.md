# PHM2012 RUL Prediction (SCI Q3/Q4 Minimal Paper Package)

This repository predicts **Remaining Useful Life (RUL)** for rolling bearings from vibration signals using a stable hybrid deep-learning pipeline.

## 1) What the model does

- Input: run-to-failure vibration windows from PHM2012 bearings.
- Output: normalized and raw-scale RUL prediction.
- Core model: **TCN + Transformer** backbone, optional **degradation-aware attention**, and optional **HI auxiliary branch**.

## 2) Dataset

- Dataset: **PHM 2012 IEEE Prognostics Challenge bearing dataset** (run-to-failure).
- Spliting/preprocessing are configured in `configs/config.yaml` and reused by all baselines/ablations.
- Main preprocessing includes windowing, normalization/scaler fitting, and controlled train/valid/test split.

## 3) Key idea

- **TCN + Transformer** capture local + global temporal degradation cues.
- **Degradation-aware attention** reweights temporal evidence.
- **Health Indicator (HI)** is used as an **auxiliary signal only**.
- Final default setup prevents HI shortcut to the RUL head (`use_hi_in_rul_head: false`).

## 4) How to run

### Train

```bash
python scripts/train.py --config configs/config.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/runs/<run_name>/checkpoints/best_model.pth
```

### Baselines

```bash
python scripts/train.py --config configs/baseline_tcn.yaml
python scripts/train.py --config configs/baseline_transformer.yaml
python scripts/train.py --config configs/baseline_tcn_transformer.yaml
python scripts/train.py --config configs/baseline_full_model.yaml
```

### Minimal ablation (A/B/C/D)

```bash
python scripts/run_min_ablation.py
```

## 5) Where results are saved

For each run:

- `outputs/runs/<run_name>/logs/`
- `outputs/runs/<run_name>/checkpoints/`
- `outputs/runs/<run_name>/results/`

Evaluation artifacts:

- `outputs/runs/<run_name>/results/evaluation_test/figures/`
- `outputs/runs/<run_name>/results/evaluation_test/tables/`
- `outputs/runs/<run_name>/results/evaluation_test/paper_artifacts/`
  - `figures/`
  - `tables/`
  - `summary.md`

Global paper tables:

- `outputs/results/ablation_results.csv`
- `outputs/results/ablation_results.md`

## 6) Reproducibility notes

- Seed control is enabled (`experiment.seed`).
- Deterministic PyTorch settings are enabled in `src/utils.py`.
- Baselines and ablations reuse the same training/evaluation scripts to avoid code duplication.
