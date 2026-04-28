# Paper Package (SCI Q3/Q4 Minimal Version)

## 1) Final model description
- Backbone: TCN + Transformer.
- Attention: degradation-aware attention (optional HI conditioning for attention only).
- HI branch: auxiliary-only.
- RUL head: **feature-only** (`rul_head_mode=feature_only`, `use_hi_in_rul_head=false`).

## 2) Dataset split protocol
- PHM2012 run-level split with fixed seed.
- Windowing and scaling shared across all experiments.
- Recommended defaults:
  - `window_size=128`, `stride=4`, `scaler_mode=standard`.
  - train-bin balancing enabled (`hybrid + median`).

## 3) Anti-leakage design
- No HI concatenation/addition/gating/residual to final RUL prediction.
- No output flipping.
- No calibration in default evaluation.
- Checkpoint head compatibility checked strictly.

## 4) HI role explanation
HI is used for:
- auxiliary monotonic/smoothness/variance constraints,
- attention conditioning,
- interpretability curves.

HI is **not** used as a direct RUL proxy in the final prediction head.

## 5) Baseline list
- TCN only
- Transformer only
- TCN + Transformer
- Full model

Run: `python scripts/run_benchmark.py`

## 6) Ablation list
- A/B/C/D and extended registry in `scripts/ablation_registry.py`.

Run: `python scripts/run_ablation.py --suite all`

## 7) Figure/table list
### Tables
- `outputs/results/main_results.csv/.md`
- `outputs/results/ablation_results.csv/.md`
- `outputs/results/per_bearing_results.csv`

### Figures
Generated under `outputs/runs/<run_name>/paper_artifacts/figures/`:
1. predicted vs true RUL per test bearing,
2. pred vs true scatter,
3. HI per bearing,
4. attention mean heatmap,
5. training/validation loss curve,
6. per-bearing RMSE bar chart.

## 8) Limitations
- Correlation objectives improve directionality but may need seed-robust tuning.
- Attention-conditioned HI can still induce weak coupling; monitor `|corr(pred, hi)|` warnings.
- Final publishable claims require completed full benchmark/ablation runs on target hardware.
