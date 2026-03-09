"""Temporal Convolutional Network implementation for RUL modeling."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove right padding to maintain causal temporal length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual temporal block with dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Stacked TCN encoder.

    Input shape:
        (batch, time, sensors)
    Output:
        sequence representation (batch, time, channels)
        pooled representation (batch, channels)
    """

    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = input_dim
        for d in dilations:
            blocks.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            in_ch = channels

        self.tcn = nn.Sequential(*blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (B, S, T)
        feat = self.tcn(x).transpose(1, 2)  # (B, T, C)
        pooled = feat.mean(dim=1)
        return feat, pooled
