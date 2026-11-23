"""Macaron feed-forward module.

Gulati et al. 2020, Section 2.1. Two linear layers with Swish between them.
The Conformer block, not this module, scales the residual by 1/2.
"""

from __future__ import annotations

import torch
from torch import nn


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.linear_in = nn.Linear(d_model, hidden)
        self.linear_out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, d_model). Pre-norm, then expand, Swish, project back.
        x = self.norm(x)
        x = self.linear_in(x)
        x = torch.nn.functional.silu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x
