"""Hybrid RUL architecture: TCN + Transformer + optional temporal attention + HI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .degradation_attention import DegradationAwareAttention
from .health_indicator import HealthIndicatorNet
from .rul_head import FusionMLP, RULHead
from .tcn import TCNEncoder
from .transformer_encoder import TransformerTemporalEncoder


@dataclass
class ModelConfig:
    sensor_dim: int
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    transformer_embed_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ffn_dim: int = 256
    dropout: float = 0.1
    backbone_variant: str = "tcn_transformer"
    use_hi: bool = True
    hi_input_source: str = "tcn"
    use_attention: bool = True
    attention_use_hi_bias: bool = True
    attention_use_temporal_gate: bool = True
    attention_use_recency_bias: bool = True
    attention_temperature: float = 1.0
    attention_recency_strength: float = 0.5
    head_hidden_dim: int = 32
    output_activation: str = "identity"


class HybridRULModel(nn.Module):
    """End-to-end regression model for bearing RUL."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone_variant = str(cfg.backbone_variant).lower()
        if self.backbone_variant not in {"tcn_transformer", "tcn_only", "transformer_only"}:
            raise ValueError(f"Unsupported backbone_variant: {cfg.backbone_variant}")

        self.tcn = TCNEncoder(
            input_dim=cfg.sensor_dim,
            channels=cfg.tcn_channels,
            kernel_size=cfg.tcn_kernel_size,
            dilations=cfg.tcn_dilations,
            dropout=cfg.dropout,
        )
        tr_input_dim = cfg.tcn_channels if self.backbone_variant != "transformer_only" else cfg.sensor_dim
        self.transformer = TransformerTemporalEncoder(
            input_dim=tr_input_dim,
            embedding_dim=cfg.transformer_embed_dim,
            heads=cfg.transformer_heads,
            layers=cfg.transformer_layers,
            ffn_dim=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
        )
        self.use_hi = bool(cfg.use_hi)
        self.hi = (
            HealthIndicatorNet(
                sensor_dim=cfg.sensor_dim,
                tcn_channels=cfg.tcn_channels,
                hi_input_source=cfg.hi_input_source,
            )
            if self.use_hi
            else None
        )
        self.hi_feature_dim = self.hi.feature_dim if self.hi is not None else (cfg.sensor_dim * 4)
        self.use_attention = bool(cfg.use_attention)
        if self.use_attention:
            self.degradation_attention = DegradationAwareAttention(
                embed_dim=cfg.transformer_embed_dim,
                num_heads=cfg.transformer_heads,
                dropout=cfg.dropout,
                temperature=cfg.attention_temperature,
                recency_strength=cfg.attention_recency_strength,
                use_hi_bias=cfg.attention_use_hi_bias,
                use_temporal_gate=cfg.attention_use_temporal_gate,
                use_recency_bias=cfg.attention_use_recency_bias,
            )
            attention_dim = cfg.transformer_embed_dim
        else:
            self.degradation_attention = None
            attention_dim = cfg.transformer_embed_dim

        fusion_input_dim = cfg.tcn_channels + cfg.transformer_embed_dim + attention_dim + 1
        self.fusion = FusionMLP(fusion_input_dim, dropout=cfg.dropout)
        self.rul_head = RULHead(
            input_dim=64,
            hidden_dim=cfg.head_hidden_dim,
            activation=cfg.output_activation,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Stage 1/2 backbone variants
        if self.backbone_variant == "tcn_only":
            tcn_seq, tcn_pool = self.tcn(x)
            tr_seq = torch.zeros(
                (x.size(0), tcn_seq.size(1), self.cfg.transformer_embed_dim),
                device=x.device,
                dtype=x.dtype,
            )
            tr_pool = torch.zeros(
                (x.size(0), self.cfg.transformer_embed_dim),
                device=x.device,
                dtype=x.dtype,
            )
        elif self.backbone_variant == "transformer_only":
            tr_seq, tr_pool = self.transformer(x)
            tcn_seq = None
            tcn_pool = torch.zeros(
                (x.size(0), self.cfg.tcn_channels),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            tcn_seq, tcn_pool = self.tcn(x)
            tr_seq, tr_pool = self.transformer(tcn_seq)

        # HI module from raw input
        if self.use_hi and self.hi is not None:
            hi_score, hi_stats, hi_temporal = self.hi(x, tcn_seq=tcn_seq)
        else:
            hi_score = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
            hi_stats = torch.zeros((x.size(0), self.hi_feature_dim), device=x.device, dtype=x.dtype)
            hi_temporal = None

        # Stage 3: optional degradation-aware attention
        if self.use_attention and self.degradation_attention is not None:
            attn_hi = hi_score if self.cfg.attention_use_hi_bias else None
            da_feat, attn_map, temporal_attn = self.degradation_attention(tr_seq, attn_hi)
        else:
            da_feat = tr_pool
            temporal_attn = torch.full(
                (tr_seq.size(0), tr_seq.size(1)),
                1.0 / max(tr_seq.size(1), 1),
                device=tr_seq.device,
                dtype=tr_seq.dtype,
            )
            attn_map = None

        # Stage 4: Feature fusion
        fused = torch.cat([tcn_pool, tr_pool, da_feat, hi_score], dim=-1)
        fused = self.fusion(fused)

        # Stage 5: Prediction
        pred = self.rul_head(fused)

        return {
            "pred": pred,
            "hi": hi_score,
            "hi_stats": hi_stats,
            "hi_temporal": hi_temporal,
            "tcn_pool": tcn_pool,
            "transformer_pool": tr_pool,
            "degradation_feat": da_feat,
            "attn_map": attn_map,
            "temporal_attn": temporal_attn,
            "fusion_feat": fused,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
