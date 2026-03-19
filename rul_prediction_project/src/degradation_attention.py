"""Degradation-aware attention module.

Core idea:
    attention = softmax(QK^T / sqrt(d) + degradation_bias)
where degradation_bias is derived from HI and temporal progression so that later
life-cycle positions are weighted more under stronger degradation estimates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationAwareAttention(nn.Module):
    """Custom attention integrating health indicator and late-life prior."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.5,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.hi_to_bias = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_heads),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.temperature = float(temperature)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(
        self,
        x: torch.Tensor,
        hi_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Sequence feature (B, T, C)
            hi_score: Scalar HI in [0,1], shape (B,1)

        Returns:
            weighted_vector: (B, C)
            attn_weights: (B, H, T, T)
        """

        b, t, _ = x.shape
        x_norm = self.input_norm(x)
        q = self._split_heads(self.q_proj(x_norm))
        k = self._split_heads(self.k_proj(x_norm))
        v = self._split_heads(self.v_proj(x_norm))

        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Late-life bias: linearly increasing over keys.
        life = torch.linspace(0, 1, steps=t, device=x.device, dtype=x.dtype)
        life = life.view(1, 1, 1, t)

        hi_gate = self.hi_to_bias(hi_score).view(b, self.num_heads, 1, 1)
        degradation_bias = hi_gate * life

        scores = (logits + degradation_bias) / self.temperature
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.out_proj(self._combine_heads(out))

        weighted_vector = out.mean(dim=1)
        return weighted_vector, attn
