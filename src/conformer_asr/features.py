"""Log-mel features and SpecAugment.

80-dimensional filterbanks match the Conformer paper's input. SpecAugment
(Park et al. 2019) is the augmentation used in the Conformer training recipe.
"""

from __future__ import annotations

import torch
from torch import nn


class LogMelExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> None:
        super().__init__()
        self.mel = torchaudio_mel(sample_rate, n_mels, n_fft, hop_length, win_length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (samples,) in [-1, 1]. Returns (time, n_mels), utterance-CMVN'd.
        spec = self.mel(waveform)
        log_mel = torch.log(spec.clamp(min=1e-10)).transpose(0, 1)
        mean = log_mel.mean(dim=0, keepdim=True)
        std = log_mel.std(dim=0, keepdim=True).clamp(min=1e-5)
        return (log_mel - mean) / std


def torchaudio_mel(sample_rate: int, n_mels: int, n_fft: int, hop_length: int, win_length: int):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
        f_min=0.0,
        f_max=sample_rate / 2,
    )


def spec_augment(
    mel: torch.Tensor,
    freq_mask_width: int,
    time_mask_width: int,
    n_freq_masks: int,
    n_time_masks: int,
) -> torch.Tensor:
    """Zero random time and frequency bands. `mel` is (time, freq)."""
    out = mel.clone()
    time, freq = out.shape
    for _ in range(n_freq_masks):
        width = int(torch.randint(0, freq_mask_width + 1, ()).item())
        if width == 0 or width >= freq:
            continue
        start = int(torch.randint(0, freq - width + 1, ()).item())
        out[:, start : start + width] = 0
    max_time = min(time_mask_width, max(time // 2, 0))
    for _ in range(n_time_masks):
        if max_time == 0:
            break
        width = int(torch.randint(0, max_time + 1, ()).item())
        if width == 0 or width >= time:
            continue
        start = int(torch.randint(0, time - width + 1, ()).item())
        out[start : start + width, :] = 0
    return out
