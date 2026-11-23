"""Conformer encoder.

Gulati et al. 2020, Section 2.4. One block is a macaron pair of feed-forward
modules wrapped around relative self-attention and a convolution module:

    x = x + (1/2) * FFN(x)
    x = x + MHSA(x)
    x = x + Conv(x)
    x = x + (1/2) * FFN(x)
    x = LayerNorm(x)

Each of FFN, MHSA, and Conv applies its own pre-norm internally.
"""

from __future__ import annotations

import torch
from torch import nn

from conformer_asr.attention import RelativeMultiHeadAttention, RelativePositionalEncoding
from conformer_asr.convolution import ConvolutionModule
from conformer_asr.feedforward import FeedForwardModule
from conformer_asr.subsampling import Conv2dSubsampling


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_expansion: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.attn = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.dropout(self.ffn1(x))
        x = x + self.attn(x, pos_emb, pad_mask)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.dropout(self.ffn2(x))
        x = self.final_norm(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 144,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_channels: int = 144,
    ) -> None:
        super().__init__()
        self.subsampling = Conv2dSubsampling(n_mels, d_model, subsampling_channels, dropout)
        self.pos_encoding = RelativePositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, num_heads, ffn_expansion, conv_kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, feats: torch.Tensor, feat_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.subsampling(feats, feat_lens)
        x, pos_emb = self.pos_encoding(x)
        pad_mask = _pad_mask(lengths, x.size(1))
        for layer in self.layers:
            x = layer(x, pos_emb, pad_mask)
        return x, lengths


def _pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) >= lengths.unsqueeze(1)
