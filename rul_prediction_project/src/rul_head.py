"""Regression heads used by hybrid model."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Fusion network: input -> 256 -> 128 -> 64."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OutputActivation(nn.Module):
    """Configurable final activation for the regression head."""

    def __init__(self, activation: str = "identity"):
        super().__init__()
        activation = str(activation).lower()
        if activation == "identity":
            self.activation = nn.Identity()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported output activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class RULHead(nn.Module):
    """Prediction head: 64 -> hidden -> 1, with configurable activation."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, activation: str = "identity"):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.output_activation = OutputActivation(activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.proj(x))


class HIGuidedResidualRULHead(nn.Module):
    """HI-guided head: predict RUL from HI trend plus feature residual."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        activation: str = "identity",
        residual_scale: float = 0.3,
    ):
        super().__init__()
        self.hi_trend = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = float(residual_scale)
        self.output_activation = OutputActivation(activation=activation)

    def forward(self, features: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep a direct HI passthrough so prediction is always explicitly HI-driven.
        hi_component = hi + self.hi_trend(hi)
        # Bound residual correction magnitude to prevent feature branch from dominating.
        residual_component = torch.tanh(self.residual(features)) * self.residual_scale
        pred = self.output_activation(hi_component + residual_component)
        return pred, hi_component, residual_component
