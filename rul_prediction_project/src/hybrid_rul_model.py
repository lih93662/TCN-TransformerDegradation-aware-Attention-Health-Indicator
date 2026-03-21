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
    use_attention: bool = True
    attention_temperature: float = 1.0
    attention_recency_strength: float = 0.5
    head_hidden_dim: int = 32
    output_activation: str = "identity"


class HybridRULModel(nn.Module):
    """End-to-end regression model for bearing RUL."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tcn = TCNEncoder(
            input_dim=cfg.sensor_dim,
            channels=cfg.tcn_channels,
            kernel_size=cfg.tcn_kernel_size,
            dilations=cfg.tcn_dilations,
            dropout=cfg.dropout,
        )
        self.transformer = TransformerTemporalEncoder(
            input_dim=cfg.tcn_channels,
            embedding_dim=cfg.transformer_embed_dim,
            heads=cfg.transformer_heads,
            layers=cfg.transformer_layers,
            ffn_dim=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
        )
        self.hi = HealthIndicatorNet(sensor_dim=cfg.sensor_dim)
        self.use_attention = bool(cfg.use_attention)
        if self.use_attention:
            self.degradation_attention = DegradationAwareAttention(
                embed_dim=cfg.transformer_embed_dim,
                num_heads=cfg.transformer_heads,
                dropout=cfg.dropout,
                temperature=cfg.attention_temperature,
                recency_strength=cfg.attention_recency_strength,
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
        # Stage 1: TCN
        tcn_seq, tcn_pool = self.tcn(x)

        # Stage 2: Transformer
        tr_seq, tr_pool = self.transformer(tcn_seq)

        # HI module from raw input
        hi_score, hi_stats = self.hi(x)

        # Stage 3: optional degradation-aware attention
        if self.use_attention and self.degradation_attention is not None:
            da_feat, attn_map, temporal_attn = self.degradation_attention(tr_seq, hi_score)
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
