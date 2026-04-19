"""Checkpoint/model compatibility helpers for RUL head variants."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def detect_checkpoint_head_variant(model_state: Mapping[str, torch.Tensor]) -> str:
    """Infer checkpoint RUL head variant from state-dict keys."""

    keys = set(model_state.keys())
    has_old_proj = any(k.startswith("rul_head.proj.") for k in keys)
    has_new_hi = any(k.startswith("rul_head.hi_trend.") for k in keys)
    has_new_residual = any(k.startswith("rul_head.residual.") for k in keys)
    has_new = has_new_hi or has_new_residual
    if has_old_proj and not has_new:
        return "old"
    if has_new and not has_old_proj:
        return "hi_driven"
    if has_old_proj and has_new:
        return "mixed"
    return "unknown"


def resolve_rul_head_mode(model_cfg: Mapping[str, object], default: str = "hi_guided_residual") -> str:
    """Resolve head mode, supporting optional alias model_head_variant."""

    explicit_mode = str(model_cfg.get("rul_head_mode", "")).strip().lower()
    if explicit_mode:
        return explicit_mode
    variant = str(model_cfg.get("model_head_variant", "")).strip().lower()
    if variant == "old":
        return "plain"
    if variant == "hi_driven":
        return "hi_guided_residual"
    return default


def detect_model_head_variant(model) -> str:
    """Infer model head variant based on configured head mode."""

    mode = str(getattr(model, "rul_head_mode", "plain")).lower()
    if mode in {"plain", "concat_hi"}:
        return "old"
    if mode == "hi_guided_residual":
        return "hi_driven"
    return "unknown"


def assert_checkpoint_head_compatible(model, model_state: Mapping[str, torch.Tensor]) -> None:
    """Raise a clear error when checkpoint head and model head are incompatible."""

    ckpt_variant = detect_checkpoint_head_variant(model_state)
    model_variant = detect_model_head_variant(model)
    if ckpt_variant == "unknown" or model_variant == "unknown":
        return
    if ckpt_variant == model_variant:
        return

    model_mode = str(getattr(model, "rul_head_mode", "plain")).lower()
    raise RuntimeError(
        "Checkpoint/model RUL head mismatch detected. "
        f"Checkpoint head='{ckpt_variant}', current model head='{model_variant}' (rul_head_mode='{model_mode}'). "
        "This checkpoint was trained with a different RUL head. Please either: "
        "(a) set model.rul_head_mode to match the checkpoint variant (old checkpoints: 'plain' or 'concat_hi'; "
        "new checkpoints: 'hi_guided_residual'), or "
        "(b) retrain and evaluate with matching train/eval configs."
    )


def load_model_state_strict(model, payload: Dict[str, object]) -> None:
    """Validate head compatibility then strict-load model state."""

    state = payload["model_state"]
    assert_checkpoint_head_compatible(model, state)
    model.load_state_dict(state, strict=True)
