"""Attention-based multimodal feature fusion for RUL prediction.

This module fuses heterogeneous prognostics signals:
- Time-domain handcrafted/deep features
- Frequency-domain features (e.g., STFT/CWT embeddings)
- Deep temporal features from backbone networks
- Scalar Health Indicator (HI)

The design follows a research setting where each modality may carry complementary
information under different operating stages. Attention-based fusion provides
adaptive weighting rather than fixed concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    """Configuration for multimodal attention fusion.

    Attributes:
        time_dim: Feature size of time-domain modality.
        freq_dim: Feature size of frequency-domain modality.
        deep_dim: Feature size of deep model modality.
        hi_dim: Feature size of health indicator modality (usually 1).
        hidden_dim: Internal representation dimension for each modality token.
        num_heads: Number of attention heads.
        num_layers: Number of stacked transformer-style fusion blocks.
        dropout: Dropout rate.
        output_dim: Final fused feature dimension.
    """

    time_dim: int
    freq_dim: int
    deep_dim: int
    hi_dim: int = 1
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 128


class ModalityProjector(nn.Module):
    """Project a modality vector into shared hidden space.

    Input shape:
        ``(batch, input_dim)``

    Output shape:
        ``(batch, hidden_dim)``
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityGating(nn.Module):
    """Compute per-modality reliability gates conditioned on all modalities.

    Input shape:
        ``(batch, num_modalities, hidden_dim)``

    Output shape:
        gated tensor with same shape

    The gate values are in [0, 1] and can downweight noisy modalities.
    """

    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim

        self.context_mlp = nn.Sequential(
            nn.Linear(num_modalities * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply global conditioning gates.

        Args:
            tokens: Input tensor ``(B, M, H)``.

        Returns:
            Tuple ``(gated_tokens, gates)`` where
            - gated_tokens: ``(B, M, H)``
            - gates: ``(B, M)``
        """

        if tokens.ndim != 3:
            raise ValueError("ModalityGating expects tensor shape (B,M,H)")

        b, m, h = tokens.shape
        if m != self.num_modalities or h != self.hidden_dim:
            raise ValueError(
                f"Expected (B,{self.num_modalities},{self.hidden_dim}) but got {tuple(tokens.shape)}"
            )

        flat = tokens.reshape(b, m * h)
        gates = self.context_mlp(flat)

        gated = tokens * gates.unsqueeze(-1)
        return gated, gates


class CrossModalAttentionBlock(nn.Module):
    """One transformer-style block for cross-modal interaction.

    Input shape:
        ``(batch, num_modalities, hidden_dim)``

    Output shape:
        same as input

    Block components:
        1. Multi-head self-attention across modality tokens.
        2. Residual + LayerNorm.
        3. Feedforward network.
        4. Residual + LayerNorm.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run cross-modal attention block.

        Args:
            x: Token tensor ``(B,M,H)``.

        Returns:
            Tuple ``(out, attn_weights)`` where attn weights have shape
            ``(B, M, M)`` (averaged across heads by PyTorch API).
        """

        # Attention across modalities lets each token query complementary cues.
        attn_out, attn_w = self.attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)

        # Feedforward network refines modality-mixed token representation.
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_w


class HealthConditionedAttention(nn.Module):
    """Health-indicator conditioned token reweighting.

    Input shapes:
        tokens: ``(batch, num_modalities, hidden_dim)``
        hi: ``(batch, hi_dim)``

    Output shapes:
        conditioned tokens with same shape and per-modality coefficients.

    Rationale:
        During severe degradation, certain modalities (e.g., frequency-domain)
        may become more informative. This module adapts token importance using
        HI-driven coefficients.
    """

    def __init__(self, hidden_dim: int, num_modalities: int, hi_dim: int = 1):
        super().__init__()
        self.num_modalities = num_modalities
        self.coeff_net = nn.Sequential(
            nn.Linear(hi_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Condition token weights on HI values.

        Args:
            tokens: Tensor ``(B,M,H)``.
            hi: Tensor ``(B,hi_dim)``.

        Returns:
            Tuple ``(conditioned, coeffs)`` with coeffs shape ``(B,M)``.
        """

        coeffs = self.coeff_net(hi)
        conditioned = tokens * coeffs.unsqueeze(-1)
        return conditioned, coeffs


class AttentionPooling(nn.Module):
    """Attention pooling from modality tokens to fused vector.

    Input shape:
        ``(batch, num_modalities, hidden_dim)``

    Output shape:
        pooled vector ``(batch, hidden_dim)`` and attention weights ``(batch, M)``.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool modality tokens with learnable attention weights."""

        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MultimodalFeatureFusion(nn.Module):
    """Main attention-based multimodal fusion network.

    Inputs:
        time_feat: ``(batch, time_dim)``
        freq_feat: ``(batch, freq_dim)``
        deep_feat: ``(batch, deep_dim)``
        hi_feat: ``(batch, hi_dim)``

    Output:
        Dictionary containing:
            - fused: ``(batch, output_dim)``
            - modality_tokens: ``(batch, 4, hidden_dim)``
            - modality_gates: ``(batch, 4)``
            - hi_coeffs: ``(batch, 4)``
            - pool_weights: ``(batch, 4)``
            - attention_maps: list of ``(batch, 4, 4)``
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.time_proj = ModalityProjector(config.time_dim, h, config.dropout)
        self.freq_proj = ModalityProjector(config.freq_dim, h, config.dropout)
        self.deep_proj = ModalityProjector(config.deep_dim, h, config.dropout)
        self.hi_proj = ModalityProjector(config.hi_dim, h, config.dropout)

        self.gating = ModalityGating(hidden_dim=h, num_modalities=4, dropout=config.dropout)
        self.hi_condition = HealthConditionedAttention(hidden_dim=h, num_modalities=4, hi_dim=config.hi_dim)

        self.blocks = nn.ModuleList(
            [
                CrossModalAttentionBlock(hidden_dim=h, num_heads=config.num_heads, dropout=config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        self.pool = AttentionPooling(hidden_dim=h)

        self.out_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        time_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        deep_feat: torch.Tensor,
        hi_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        """Fuse multimodal features with stacked attention.

        Args:
            time_feat: Time-domain feature tensor ``(B, D_t)``.
            freq_feat: Frequency-domain feature tensor ``(B, D_f)``.
            deep_feat: Deep feature tensor ``(B, D_d)``.
            hi_feat: Health indicator tensor ``(B, D_h)``.

        Returns:
            Output dictionary with fused features and interpretability signals.
        """

        # Project each modality into shared hidden space.
        t_tok = self.time_proj(time_feat)
        f_tok = self.freq_proj(freq_feat)
        d_tok = self.deep_proj(deep_feat)
        h_tok = self.hi_proj(hi_feat)

        tokens = torch.stack([t_tok, f_tok, d_tok, h_tok], dim=1)  # (B, 4, H)

        # Global modality reliability gating.
        tokens, gates = self.gating(tokens)

        # HI-conditioned emphasis per modality.
        tokens, hi_coeffs = self.hi_condition(tokens, hi_feat)

        attn_maps = []
        for block in self.blocks:
            tokens, attn = block(tokens)
            attn_maps.append(attn)

        pooled, pool_weights = self.pool(tokens)
        fused = self.out_head(pooled)

        return {
            "fused": fused,
            "modality_tokens": tokens,
            "modality_gates": gates,
            "hi_coeffs": hi_coeffs,
            "pool_weights": pool_weights,
            "attention_maps": attn_maps,
        }


class MultimodalRULHead(nn.Module):
    """RUL regression head on top of multimodal fused embedding.

    Input shape:
        ``(batch, fusion_dim)``

    Output shape:
        ``(batch, 1)``
    """

    def __init__(self, fusion_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class CompleteMultimodalPredictor(nn.Module):
    """End-to-end wrapper: fusion backbone + RUL head.

    Inputs:
        time_feat: ``(B, time_dim)``
        freq_feat: ``(B, freq_dim)``
        deep_feat: ``(B, deep_dim)``
        hi_feat: ``(B, hi_dim)``

    Outputs:
        Dict containing ``pred`` and all fusion diagnostics.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.fusion = MultimodalFeatureFusion(config)
        self.head = MultimodalRULHead(fusion_dim=config.output_dim)

    def forward(
        self,
        time_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        deep_feat: torch.Tensor,
        hi_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        fusion_out = self.fusion(time_feat, freq_feat, deep_feat, hi_feat)
        pred = self.head(fusion_out["fused"])
        fusion_out["pred"] = pred
        return fusion_out


def build_multimodal_inputs_from_backbone(
    tcn_pool: torch.Tensor,
    transformer_pool: torch.Tensor,
    time_stats: torch.Tensor,
    freq_stats: torch.Tensor,
    hi_score: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Construct modality tensors from existing project outputs.

    Args:
        tcn_pool: TCN pooled feature ``(B, D_tcn)``.
        transformer_pool: Transformer pooled feature ``(B, D_tr)``.
        time_stats: Time-domain statistics ``(B, D_time)``.
        freq_stats: Frequency-domain statistics ``(B, D_freq)``.
        hi_score: Health indicator ``(B, 1)``.

    Returns:
        Dictionary with canonical modality keys for fusion model.
    """

    deep = torch.cat([tcn_pool, transformer_pool], dim=-1)

    return {
        "time_feat": time_stats,
        "freq_feat": freq_stats,
        "deep_feat": deep,
        "hi_feat": hi_score,
    }


def summarize_attention_maps(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    """Average stacked attention maps for interpretability.

    Args:
        attn_maps: List of tensors each shaped ``(B, M, M)``.

    Returns:
        Mean attention map tensor of shape ``(B, M, M)``.
    """

    if not attn_maps:
        raise ValueError("attn_maps cannot be empty")

    stacked = torch.stack(attn_maps, dim=0)
    return stacked.mean(dim=0)


def compute_modality_importance(
    gates: torch.Tensor,
    hi_coeffs: torch.Tensor,
    pool_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute composite modality importance score.

    Args:
        gates: Reliability gates ``(B,M)``.
        hi_coeffs: HI-conditioned coefficients ``(B,M)``.
        pool_weights: Attention pooling weights ``(B,M)``.

    Returns:
        Composite importance tensor ``(B,M)`` normalized across modalities.
    """

    imp = gates * hi_coeffs * pool_weights
    imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-8)
    return imp


def demo_multimodal_fusion(seed: int = 42) -> Dict[str, Tuple[int, ...]]:
    """Synthetic sanity check for multimodal fusion module.

    Returns:
        Dictionary of output shapes for quick validation.
    """

    torch.manual_seed(seed)
    b = 4

    cfg = FusionConfig(time_dim=16, freq_dim=24, deep_dim=64, hi_dim=1, hidden_dim=64, output_dim=96)
    model = CompleteMultimodalPredictor(cfg)

    time_feat = torch.randn(b, cfg.time_dim)
    freq_feat = torch.randn(b, cfg.freq_dim)
    deep_feat = torch.randn(b, cfg.deep_dim)
    hi_feat = torch.sigmoid(torch.randn(b, cfg.hi_dim))

    out = model(time_feat, freq_feat, deep_feat, hi_feat)
    avg_attn = summarize_attention_maps(out["attention_maps"])
    imp = compute_modality_importance(out["modality_gates"], out["hi_coeffs"], out["pool_weights"])

    return {
        "pred": tuple(out["pred"].shape),
        "fused": tuple(out["fused"].shape),
        "tokens": tuple(out["modality_tokens"].shape),
        "avg_attn": tuple(avg_attn.shape),
        "importance": tuple(imp.shape),
    }
