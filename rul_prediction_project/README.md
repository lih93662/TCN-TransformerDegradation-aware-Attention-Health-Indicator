# PHM2012 RUL Prediction (Paper-safe Version)

This repository provides a conservative, reproducible PHM2012 RUL pipeline with:
- TCN + Transformer backbone,
- degradation-aware attention,
- HI as **auxiliary-only** signal,
- feature-only RUL head (no HI shortcut path).

## Key anti-leakage defaults
- `model.rul_head_mode: feature_only`
- `model.use_hi_in_rul_head: false`
- `model.use_hi_auxiliary: true`
- `model.use_degradation_attention: true`
- no output flipping,
- no calibration in default evaluation.

## Train
```bash
python scripts/train.py --config configs/config.yaml
```

## Evaluate
```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/runs/<run_name>/checkpoints/best_model.pth --split test
```

Evaluation exports:
- run-level metrics JSON (normalized/raw)
- grouped per-bearing metrics
- `outputs/results/per_bearing_results.csv`

## Baseline benchmark package (main table)
```bash
python scripts/run_benchmark.py --dry-run
python scripts/run_benchmark.py
```

Exports:
- `outputs/results/main_results.csv`
- `outputs/results/main_results.md`

## Ablation package
```bash
python scripts/run_ablation.py --list
python scripts/run_ablation.py --suite legacy
python scripts/run_ablation.py --suite all
```

Exports:
- `outputs/results/ablation_results.csv`
- `outputs/results/ablation_results.md`

## Paper artifact figures
First run evaluation, then:
```bash
python scripts/generate_paper_figures.py --config configs/config.yaml --split test
```

Exports under:
- `outputs/runs/<run_name>/paper_artifacts/figures/`

Required figure set includes:
1. per-bearing predicted-vs-true RUL curves,
2. pred-vs-true scatter,
3. per-bearing HI curves,
4. attention mean heatmap,
5. training/validation loss curve,
6. per-bearing RMSE bar chart.

## Notes
- Use identical data split and preprocessing for all baseline/ablation comparisons.
- Metric consistency is enforced by shared `compute_regression_metrics`.
- See `docs/paper_package.md` for package checklist.
