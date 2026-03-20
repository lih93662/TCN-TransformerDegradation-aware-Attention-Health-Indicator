| experiment_name | attention_on | loss_type | bias_regularization_weight | rmse | mae | r2 | mean_error | pred_mean | true_mean | pred_std | true_std | best_epoch | status | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| A_backbone_only | off | mse | 0.0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | failed | Execution blocked in this environment: ModuleNotFoundError: No module named 'torch' |
| B_backbone_attention | on | mse | 0.0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not_run | Not started because launcher stops on first failure |
| C_backbone_improved_loss | off | mse_mae | 0.05 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not_run | Not started because launcher stops on first failure |
| D_full | on | mse_mae | 0.05 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not_run | Not started because launcher stops on first failure |
