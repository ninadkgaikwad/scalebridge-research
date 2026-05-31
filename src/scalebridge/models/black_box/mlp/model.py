from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """PyTorch MLP baseline for one-zone building thermal prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
