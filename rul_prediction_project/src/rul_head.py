"""Regression heads used by hybrid model."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Fusion network: 256 -> 128 -> 64."""

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


class PredictionActivation(nn.Module):
    """Configurable final activation for regression outputs."""

    def __init__(self, name: str = "identity"):
        super().__init__()
        key = str(name).lower()
        if key == "identity":
            self.activation = nn.Identity()
        elif key == "relu":
            self.activation = nn.ReLU()
        elif key == "softplus":
            self.activation = nn.Softplus()
        elif key == "sigmoid":
            self.activation = nn.Sigmoid()
        elif key == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported prediction activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class RULHead(nn.Module):
    """Prediction head: 64 -> 32 -> 1 with optional final activation."""

    def __init__(self, hidden_dim: int = 32, activation: str = "identity"):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.activation = PredictionActivation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(x))
