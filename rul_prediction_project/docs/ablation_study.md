# Ablation Study Matrix (Publication-Ready)

This repository keeps legacy A/B/C/D ablations and adds a factorized matrix for optimization, HI pathway, attention internals, and backbone variants.

## 1) Legacy A/B/C/D (kept unchanged)

- **A (backbone)**: `use_attention=false`, `loss_name=mse`, `bias_regularization_weight=0.0`
- **B (attention)**: `use_attention=true`, `loss_name=mse`, `bias_regularization_weight=0.0`
- **C (main_loss)**: `use_attention=false`, `loss_name=mse_mae`, `bias_regularization_weight=0.05`
- **D (full)**: `use_attention=true`, `loss_name=mse_mae`, `bias_regularization_weight=0.05`

## 2) Extended suites

### Optimization disentanglement
- **OPT1 / mse_only**
- **OPT2 / mse_plus_mae**
- **OPT3 / mse_plus_bias**
- **OPT4 / mse_mae_plus_bias**

### HI pathway
- **HI1 / no_hi**
- **HI2 / hi_only_without_da**
- **HI3 / hi_plus_attention**
- **HI4 / full_model**

### Attention internals
- **ATT1 / vanilla_attention**
- **ATT2 / attention_plus_hi_bias**
- **ATT3 / attention_plus_temporal_gate**
- **ATT4 / full_degradation_attention**

### Backbone variants
- **BB1 / tcn_only**
- **BB2 / transformer_only**
- **BB3 / tcn_transformer**

## 3) Scientific question mapping

- Optimization suite isolates MAE-term and bias-regularization effects independently.
- HI suite isolates whether HI pathway itself helps and how it interacts with attention.
- Attention internal suite isolates internal components (HI bias / temporal gate / recency).
- Backbone suite supports local-global synergy claims.

## 4) Commands

### List all registered ablations
```bash
python scripts/run_ablation.py --list
```

### Run all ablations
```bash
python scripts/run_ablation.py --suite all
```

### Run only legacy A/B/C/D
```bash
python scripts/run_ablation.py --suite legacy
```

### Run one suite
```bash
python scripts/run_ablation.py --suite optimization
python scripts/run_ablation.py --suite hi
python scripts/run_ablation.py --suite attention_internal
python scripts/run_ablation.py --suite backbone
```

### Run one experiment code
```bash
python scripts/run_ablation.py --code ATT3
```

### Dry run / collect only
```bash
python scripts/run_ablation.py --suite all --dry-run
python scripts/run_ablation.py --suite all --collect-only
```

## 5) Result artifacts

Per experiment run directory (`outputs/runs/<run_name>/`):
- `logs/resolved_config.json`
- `logs/history.json`
- `logs/epoch_metrics.csv`
- `logs/raw_predictions/epoch_000_test.csv`
- `checkpoints/best_model.pth`
- `results/test_metrics.json`
- `results/run_metadata.json`

Aggregated ablation exports:
- `outputs/results/ablation_summary.csv`
- `outputs/results/ablation_summary.json`
- `outputs/results/ablation_table.csv`
- `outputs/results/ablation_table.md`

## 6) Limitations

- "Transformer-only + degradation attention" is supported, but attention receives transformer sequence features directly from raw sensor projection.
- Internal attention ablations are implemented as switches over HI bias, temporal gate, and recency bias while preserving the same module interface.
- No experimental numbers are pre-filled; run suites to generate tables.
