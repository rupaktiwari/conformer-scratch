"""Convolutional subsampling by 4.

Gulati et al. 2020, Section 3: two stride-2 2D convolutions on the mel
spectrogram, then a linear projection into the model dimension.
"""

from __future__ import annotations

import torch
from torch import nn


def conv2d_out_length(length: torch.Tensor, kernel: int = 3, stride: int = 2) -> torch.Tensor:
    return (length - kernel) // stride + 1


def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
    return conv2d_out_length(conv2d_out_length(lengths))


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_mels: int, d_model: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, 1, 32, n_mels))
        freq = probe.shape[-1]
        self.proj = nn.Linear(channels * freq, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, feats: torch.Tensor, feat_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # feats: (batch, time, n_mels)
        x = self.conv(feats.unsqueeze(1))
        batch, channels, time, freq = x.shape
        x = self.proj(x.transpose(1, 2).contiguous().view(batch, time, channels * freq))
        x = self.dropout(x)
        lengths = subsampled_lengths(feat_lens).clamp(min=0, max=time)
        return x, lengths
