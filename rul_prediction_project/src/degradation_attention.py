"""Degradation-aware temporal attention module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DegradationAwareAttention(nn.Module):
    """Lightweight temporal attention integrating health indicator and recency prior.

    Compared with the previous version, this module avoids forced L2-normalized
    queries/keys, adds a configurable score temperature, and uses a second-stage
    temporal pooling gate so the attended representation does not collapse into a
    simple mean over time.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        recency_strength: float = 0.5,
        use_hi_bias: bool = True,
        use_temporal_gate: bool = True,
        use_recency_bias: bool = True,
        conditioning_gain: float = 2.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.temperature = max(float(temperature), 1e-4)
        self.recency_strength = float(recency_strength)
        self.use_hi_bias = bool(use_hi_bias)
        self.use_temporal_gate = bool(use_temporal_gate)
        self.use_recency_bias = bool(use_recency_bias)
        self.conditioning_gain = float(conditioning_gain)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.hi_to_head_bias = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_heads),
            nn.Tanh(),
        )
        self.hi_to_temporal_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.temporal_score = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(
        self,
        x: torch.Tensor,
        hi_score: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Sequence feature (B, T, C).
            hi_score: Optional scalar HI in [0, 1], shape (B, 1).

        Returns:
            weighted_vector: (B, C)
            attn_weights: (B, H, T, T)
            temporal_weights: (B, T)
        """

        b, t, _ = x.shape
        x_norm = self.input_norm(x)
        q = self._split_heads(self.q_proj(x_norm))
        k = self._split_heads(self.k_proj(x_norm))
        v = self._split_heads(self.v_proj(x_norm))

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        life = torch.linspace(0.0, 1.0, steps=t, device=x.device, dtype=x.dtype)
        key_life = life.view(1, 1, 1, t)
        temporal_life = life.view(1, t)

        cond_hi = None
        if self.use_hi_bias and hi_score is not None:
            hi_std = torch.std(hi_score, dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            hi_centered = (hi_score - hi_score.mean(dim=0, keepdim=True)) / hi_std
            cond_hi = torch.tanh(hi_centered) * self.conditioning_gain
            head_bias = self.hi_to_head_bias(cond_hi).view(b, self.num_heads, 1, 1)
        else:
            head_bias = torch.zeros((b, self.num_heads, 1, 1), dtype=x.dtype, device=x.device)
        if self.use_recency_bias:
            degradation_bias = self.recency_strength * head_bias * key_life
        else:
            degradation_bias = torch.zeros_like(logits)

        base_scores = logits / self.temperature
        scores = (logits + degradation_bias) / self.temperature
        base_attn = torch.softmax(base_scores, dim=-1)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = self.out_proj(self._combine_heads(context))

        temporal_logits = self.temporal_score(context).squeeze(-1)
        if self.use_temporal_gate:
            if hi_score is not None:
                gate = self.hi_to_temporal_gate(hi_score)
            else:
                gate = torch.zeros((b, 1), dtype=x.dtype, device=x.device)
            if self.use_recency_bias:
                temporal_logits = temporal_logits + self.recency_strength * gate * temporal_life
            temporal_weights = torch.softmax(temporal_logits / self.temperature, dim=-1)
        else:
            temporal_weights = torch.full(
                (b, t),
                1.0 / max(t, 1),
                dtype=x.dtype,
                device=x.device,
            )
        temporal_weights = self.dropout(temporal_weights)

        weighted_vector = torch.sum(context * temporal_weights.unsqueeze(-1), dim=1)
        debug = {
            "hi_cond_mean": cond_hi.mean() if cond_hi is not None else torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "hi_cond_std": cond_hi.std(unbiased=False) if cond_hi is not None else torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "head_bias_mean": head_bias.mean(),
            "head_bias_std": head_bias.std(unbiased=False),
            "degradation_bias_mean": degradation_bias.mean(),
            "degradation_bias_std": degradation_bias.std(unbiased=False),
            "attn_delta_l1": torch.mean(torch.abs(attn - base_attn)),
        }
        return weighted_vector, attn, temporal_weights, debug
