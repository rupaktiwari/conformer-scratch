"""Multi-headed self-attention with relative sinusoidal positions.

Gulati et al. 2020, Section 2.2. The relative positions follow Transformer-XL
(Dai et al. 2019): content scores, position scores, and two learned biases.

    scores = (Q + u) K^T + shifted((Q + v) P^T)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal encodings for offsets -(T-1) ... +(T-1).

    The layout matches the shift trick in Transformer-XL / ESPnet: positive
    offsets are reversed and concatenated with the strict negative offsets,
    so a later `rel_shift` lines offset 0 up on the attention diagonal.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal relative positions")
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pe: torch.Tensor | None = None
        self._extend(max_len)

    def _extend(self, length: int) -> None:
        if self.pe is not None and self.pe.size(1) >= 2 * length - 1:
            return
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        positive = torch.zeros(length, self.d_model)
        negative = torch.zeros(length, self.d_model)
        positive[:, 0::2] = torch.sin(position * div)
        negative[:, 0::2] = torch.sin(-position * div)
        if self.d_model > 1:
            positive[:, 1::2] = torch.cos(position * div)
            negative[:, 1::2] = torch.cos(-position * div)
        # Index of offset 0 is `length - 1` after this concat.
        pe = torch.cat([positive.flip(0), negative[1:]], dim=0).unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, d_model)
        time = x.size(1)
        self._extend(time)
        assert self.pe is not None
        center = self.pe.size(1) // 2
        pos_emb = self.pe[:, center - time + 1 : center + time]
        pos_emb = pos_emb.to(device=x.device, dtype=x.dtype)
        # Scale features the way Transformer embeddings are scaled. These are
        # conv-subsampled frames, and the scale keeps attention logits in the
        # range this relative-position parameterization expects.
        x = x * self.xscale
        return self.dropout(x), self.dropout(pos_emb)


def rel_shift(scores: torch.Tensor) -> torch.Tensor:
    """Turn (batch, head, query, 2 * query - 1) into (batch, head, query, query).

    Same padding-and-view trick as Transformer-XL and the ESPnet Conformer.
    `view` is required here: a reshape would read the storage in a different order.
    """
    zero_pad = scores.new_zeros((*scores.size()[:3], 1))
    padded = torch.cat([zero_pad, scores], dim=-1)
    padded = padded.view(*scores.size()[:2], scores.size(3) + 1, scores.size(2))
    shifted = padded[:, :, 1:].view_as(scores)
    return shifted[:, :, :, : scores.size(-1) // 2 + 1]


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide num_heads ({num_heads})")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.w_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(num_heads, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, time, d_model)
        # pos_emb: (1, 2 * time - 1, d_model)
        # pad_mask: (batch, time) True where the frame is padding
        x = self.norm(x)
        batch, time, _ = x.shape
        query = self._split(self.w_q(x), batch, time)
        key = self._split(self.w_k(x), batch, time)
        value = self._split(self.w_v(x), batch, time)
        pos = self._split(self.w_pos(pos_emb), pos_emb.size(0), pos_emb.size(1))

        query_u = query + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        query_v = query + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        content = torch.matmul(query_u, key.transpose(-2, -1))
        position = rel_shift(torch.matmul(query_v, pos.transpose(-2, -1)))
        position = position[:, :, :, :time]
        scores = (content + position) / math.sqrt(self.d_k)

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
            # Padded queries have no legal keys. Give them a finite row, then
            # zero their outputs below so softmax does not see an all -inf row.
            scores = scores.masked_fill(pad_mask[:, None, :, None], 0.0)

        attn = torch.softmax(scores, dim=-1)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], 0.0)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch, time, -1)
        out = self.dropout(self.w_out(context))
        if pad_mask is not None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out

    def _split(self, tensor: torch.Tensor, batch: int, time: int) -> torch.Tensor:
        return tensor.view(batch, time, self.num_heads, self.d_k).transpose(1, 2)
