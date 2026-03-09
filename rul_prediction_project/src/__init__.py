"""RUL prediction research package.

This package provides modular components for a hybrid PHM2012 prognostics model:
- Data loading and preprocessing
- Health indicator modeling
- TCN and Transformer encoders
- Degradation-aware attention
- Training, evaluation, and visualization utilities

The codebase is intentionally organized for experiment reproducibility and easy
ablation studies in academic settings.
"""

__all__ = [
    "dataset",
    "preprocess",
    "health_indicator",
    "tcn",
    "transformer_encoder",
    "degradation_attention",
    "hybrid_rul_model",
    "rul_head",
    "loss",
    "trainer",
    "evaluator",
    "visualization",
    "utils",
    "features",
    "data",
    "models",
]
