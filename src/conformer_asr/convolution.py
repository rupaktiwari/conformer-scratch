"""Convolution module.

Gulati et al. 2020, Section 2.3 and Figure 2.

    LayerNorm
    pointwise conv, expansion 2, GLU
    depthwise conv (kernel 31 here; the paper reports 32)
    batch norm
    Swish
    pointwise conv
    dropout
"""

from __future__ import annotations

import torch
from torch import nn


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding preserves time")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, time, d_model). pad_mask: (batch, time), True on padding.
        x = self.norm(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        x = torch.nn.functional.glu(x, dim=1)
        x = self.depthwise(x)
        x = torch.nn.functional.silu(self.batch_norm(x))
        x = self.pointwise_out(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x
