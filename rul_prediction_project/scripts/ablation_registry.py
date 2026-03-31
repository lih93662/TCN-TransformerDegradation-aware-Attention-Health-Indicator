"""Central registry for ablation experiments.

Each entry contains:
- code: short paper-friendly id
- name: descriptive short name
- suite: logical group for selective runs
- config: path relative to project root
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class AblationSpec:
    code: str
    name: str
    suite: str
    config: Path
    question: str


ABLATION_SPECS: List[AblationSpec] = [
    # Legacy A/B/C/D kept for backward-compatible paper tables.
    AblationSpec("A", "backbone", "legacy", Path("configs/config_backbone.yaml"), "Backbone baseline without DA attention"),
    AblationSpec("B", "attention", "legacy", Path("configs/config_attention.yaml"), "Gain from enabling attention"),
    AblationSpec("C", "main_loss", "legacy", Path("configs/config_loss.yaml"), "Combined optimization variant without attention"),
    AblationSpec("D", "full", "legacy", Path("configs/config_full.yaml"), "Legacy full variant"),

    # Optimization disentanglement.
    AblationSpec("OPT1", "mse_only", "optimization", Path("configs/ablations/opt_mse_only.yaml"), "Pure MSE supervision"),
    AblationSpec("OPT2", "mse_plus_mae", "optimization", Path("configs/ablations/opt_mse_plus_mae.yaml"), "Composite loss effect"),
    AblationSpec("OPT3", "mse_plus_bias", "optimization", Path("configs/ablations/opt_mse_plus_bias.yaml"), "Bias regularization effect"),
    AblationSpec("OPT4", "mse_mae_plus_bias", "optimization", Path("configs/ablations/opt_mse_mae_plus_bias.yaml"), "Composite+bias joint effect"),

    # HI pathway.
    AblationSpec("HI1", "no_hi", "hi", Path("configs/ablations/hi_no_hi.yaml"), "No HI pathway"),
    AblationSpec("HI2", "hi_only_without_da", "hi", Path("configs/ablations/hi_only_without_da.yaml"), "HI path without degradation attention"),
    AblationSpec("HI3", "hi_plus_attention", "hi", Path("configs/ablations/hi_plus_attention.yaml"), "HI with vanilla attention"),
    AblationSpec("HI4", "full_model", "hi", Path("configs/ablations/hi_full_model.yaml"), "HI with full degradation attention"),

    # Attention internals.
    AblationSpec("ATT1", "vanilla_attention", "attention_internal", Path("configs/ablations/attn_vanilla.yaml"), "Attention without HI bias, temporal gate, recency bias"),
    AblationSpec("ATT2", "attention_plus_hi_bias", "attention_internal", Path("configs/ablations/attn_plus_hi_bias.yaml"), "Add HI-conditioned bias only"),
    AblationSpec("ATT3", "attention_plus_temporal_gate", "attention_internal", Path("configs/ablations/attn_plus_temporal_gate.yaml"), "Add temporal gate without recency bias"),
    AblationSpec("ATT4", "full_degradation_attention", "attention_internal", Path("configs/ablations/attn_full_degradation.yaml"), "Full DA attention stack"),

    # Backbone variants.
    AblationSpec("BB1", "tcn_only", "backbone", Path("configs/ablations/backbone_tcn_only.yaml"), "Local-only temporal modeling"),
    AblationSpec("BB2", "transformer_only", "backbone", Path("configs/ablations/backbone_transformer_only.yaml"), "Global-only temporal modeling"),
    AblationSpec("BB3", "tcn_transformer", "backbone", Path("configs/ablations/backbone_tcn_transformer.yaml"), "Local-global hybrid"),
]


def list_suites() -> List[str]:
    return sorted({s.suite for s in ABLATION_SPECS})


def get_specs(suite: str | None = None, code: str | None = None) -> List[AblationSpec]:
    specs = ABLATION_SPECS
    if suite and suite != "all":
        specs = [s for s in specs if s.suite == suite]
    if code:
        specs = [s for s in specs if s.code == code]
    return specs


def to_rows(specs: Iterable[AblationSpec]) -> List[Dict[str, str]]:
    return [
        {
            "code": s.code,
            "name": s.name,
            "suite": s.suite,
            "config": str(s.config),
            "question": s.question,
        }
        for s in specs
    ]
