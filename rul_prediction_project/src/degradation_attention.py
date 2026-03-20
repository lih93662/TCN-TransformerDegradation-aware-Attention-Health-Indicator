"""Degradation-aware attention module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DegradationAwareAttention(nn.Module):
    """Lightweight temporal attention with ablation-friendly modes.

    Modes:
    - ``none``: bypass attention and return mean pooled features.
    - ``degradation``: multi-head temporal attention with HI-conditioned late-life bias.

    The implementation stays conservative: it keeps a small multi-head attention
    block but exposes temperature, score normalization, and bias scaling so that
    attention sharpness can be diagnosed and tuned in a controlled manner.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        mode: str = "degradation",
        use_qk_norm: bool = False,
        bias_scale: float = 1.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.mode = str(mode).lower()
        self.use_qk_norm = bool(use_qk_norm)
        self.scale = 1.0 / math.sqrt(self.head_dim)

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
        self.temperature = max(float(temperature), 1e-3)
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_scale), dtype=torch.float32))
        self.pool_gate = nn.Linear(embed_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.pool_gate):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def _uniform_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        pooled = x.mean(dim=1)
        attn = torch.full((b, 1, t, t), 1.0 / max(1, t), device=x.device, dtype=x.dtype)
        return pooled, attn

    def forward(
        self,
        x: torch.Tensor,
        hi_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "none":
            return self._uniform_attention(x)
        if self.mode != "degradation":
            raise ValueError(f"Unsupported attention mode: {self.mode}")

        b, t, _ = x.shape
        x_norm = self.input_norm(x)
        q = self._split_heads(self.q_proj(x_norm))
        k = self._split_heads(self.k_proj(x_norm))
        v = self._split_heads(self.v_proj(x_norm))

        if self.use_qk_norm:
            q = torch.nn.functional.normalize(q, p=2.0, dim=-1)
            k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        life = torch.linspace(0, 1, steps=t, device=x.device, dtype=x.dtype)
        life = life.view(1, 1, 1, t)

        hi_gate = self.hi_to_bias(hi_score).view(b, self.num_heads, 1, 1)
        degradation_bias = hi_gate * life * self.bias_scale.to(dtype=x.dtype)

        scores = (logits + degradation_bias) / self.temperature
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.out_proj(self._combine_heads(out))

        query_summary = attn.mean(dim=1).mean(dim=1)
        pool_logits = self.pool_gate(out).squeeze(-1)
        pool_scores = pool_logits + query_summary
        pool_weights = torch.softmax(pool_scores, dim=-1)
        weighted_vector = torch.sum(out * pool_weights.unsqueeze(-1), dim=1)
        return weighted_vector, attn
