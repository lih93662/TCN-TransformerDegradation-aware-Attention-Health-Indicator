"""Hybrid RUL architecture: TCN + Transformer + optional attention + HI."""

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
    attention_mode: str = "degradation"
    attention_temperature: float = 1.0
    attention_use_qk_norm: bool = False
    attention_bias_scale: float = 1.0
    prediction_activation: str = "identity"


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
        self.degradation_attention = DegradationAwareAttention(
            embed_dim=cfg.transformer_embed_dim,
            num_heads=cfg.transformer_heads,
            dropout=cfg.dropout,
            temperature=cfg.attention_temperature,
            mode=cfg.attention_mode,
            use_qk_norm=cfg.attention_use_qk_norm,
            bias_scale=cfg.attention_bias_scale,
        )

        fusion_input_dim = cfg.tcn_channels + cfg.transformer_embed_dim + cfg.transformer_embed_dim + 1
        self.fusion = FusionMLP(fusion_input_dim, dropout=cfg.dropout)
        self.rul_head = RULHead(activation=cfg.prediction_activation)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tcn_seq, tcn_pool = self.tcn(x)
        tr_seq, tr_pool = self.transformer(tcn_seq)
        hi_score, hi_stats = self.hi(x)
        da_feat, attn_map = self.degradation_attention(tr_seq, hi_score)

        fused = torch.cat([tcn_pool, tr_pool, da_feat, hi_score], dim=-1)
        fused = self.fusion(fused)
        pred = self.rul_head(fused)

        return {
            "pred": pred,
            "hi": hi_score,
            "hi_stats": hi_stats,
            "tcn_pool": tcn_pool,
            "transformer_pool": tr_pool,
            "degradation_feat": da_feat,
            "attn_map": attn_map,
            "fusion_feat": fused,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
