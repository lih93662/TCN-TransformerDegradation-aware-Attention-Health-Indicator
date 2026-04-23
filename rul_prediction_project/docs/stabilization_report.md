# Main Model Stabilization Report

## Observed issues (from latest run logs)
- Early best epoch (`best_epoch=3`) and early stop at epoch 8.
- Under-dispersed predictions (`pred_true_std_ratio` around 0.5).
- Negative test R2.
- Raw PHM score overflow to `inf` in raw-scale evaluation.

## Root-cause findings from code inspection
1. **PHM overflow**: PHM score used direct `exp()` on raw-scale errors with no clipping, so large positive errors overflowed numerically.
2. **Premature stop pressure**: early stopping monitored RMSE with low patience and no minimum epoch gate.
3. **Dispersion pressure missing**: existing loss optimized error magnitude/bias but had no term to counter prediction under-dispersion.
4. **Correlation penalty interpretability was weak**: logged `corr_pen` values were hard to interpret without the paired correlation value and clear sign convention.
5. **HI/attention diagnostics too weak**: training/evaluation logs did not consistently expose HI distribution statistics to verify branch effectiveness.
6. **Eval/model config mismatch risk**: evaluate model builder did not include newly added model switches (backbone/HI/attention-internal flags).

## Changes made
- Added numerically-safe PHM score clipping to prevent overflow while preserving metric shape.
- Added optional `std_regularization_weight` loss term to reduce prediction under-dispersion.
- Added optional `correlation_regularization_weight` term to discourage mean-like regression collapse.
- Added optional `hi_supervision_weight` to weakly align HI with normalized RUL for better HI informativeness.
- Added HI rank/variance regularizers (`hi_rank_weight`, `hi_variance_weight`, `hi_variance_floor`) to reduce HI near-constant collapse on valid/test.
- Logged both `corr_penalty` and its underlying `corr_value` for interpretability.
- Added early-stopping guards:
  - `early_stopping_min_epochs`
  - `early_stopping_min_delta`
- Extended trainer logging with:
  - std penalty component
  - HI mean/std/min/max per epoch
- Extended evaluation diagnostics with:
  - HI summary JSON
  - prediction-vs-target raw scatter CSV export
- Updated evaluation model builder to consume explicit model switches (`backbone_variant`, `use_hi`, attention internal flags).
- Tuned default baseline config conservatively for stability:
  - `loss_name: mse_mae`, `use_mae_term: true`, `loss_mae_weight: 0.2`
  - `std_regularization_weight: 0.05`
  - `early_stopping_patience: 8`, `early_stopping_min_epochs: 15`, `early_stopping_min_delta: 0.0005`
  - `scheduler_patience: 5`

## Validation performed
- Config parsing and ablation dry-run registry check.
- Python compile check for updated scripts/modules.
- Synthetic smoke train/eval path was run successfully (`scripts/smoke_train_eval.py`).
- (Environment-limited) no full PHM2012 run was executed in this container.

## Remaining uncertainty
- Final metric improvement needs at least one real training/evaluation run on your PHM2012 setup.
- Optimal `std_regularization_weight` and `correlation_regularization_weight` may require light tuning around `[0.01, 0.10]`.
