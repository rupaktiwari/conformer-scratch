"""LibriSpeech subset, cached as log-mel tensors.

The full `train.100` split is about 6 GB. This loader streams
`openslr/librispeech_asr` (CC-BY-4.0) and keeps a few hundred to a thousand
utterances, which is what a free T4 session can download and train on.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from conformer_asr.config import DataConfig
from conformer_asr.features import LogMelExtractor, spec_augment
from conformer_asr.subsampling import subsampled_lengths
from conformer_asr.text import normalize_text, tokenize


class MelDataset(Dataset):
    def __init__(self, path: Path, augment: dict | None = None) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.mels: list[torch.Tensor] = blob["mels"]
        self.tokens: list[torch.Tensor] = blob["tokens"]
        self.texts: list[str] = blob["texts"]
        self.lengths = [int(mel.shape[0]) for mel in self.mels]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.mels)

    def __getitem__(self, index: int) -> dict:
        mel = self.mels[index].float()
        if self.augment is not None:
            mel = spec_augment(mel, **self.augment)
        return {"mel": mel, "tokens": self.tokens[index], "text": self.texts[index]}


class BucketBatchSampler(Sampler[list[int]]):
    """Group similar lengths so padding stays small on a 16 GB GPU."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, seed: int) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        # BatchNorm in the convolution module needs more than one utterance.
        if len(self.batches) > 1 and len(self.batches[-1]) < 2:
            self.batches[-2].extend(self.batches.pop())

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in perm]
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate_batch(items: list[dict]) -> dict:
    feat_lens = torch.tensor([item["mel"].shape[0] for item in items], dtype=torch.long)
    max_frames = int(feat_lens.max())
    n_mels = items[0]["mel"].shape[1]
    feats = torch.zeros(len(items), max_frames, n_mels)
    for index, item in enumerate(items):
        frames = item["mel"].shape[0]
        feats[index, :frames] = item["mel"]
    token_lens = torch.tensor([item["tokens"].numel() for item in items], dtype=torch.long)
    tokens = torch.cat([item["tokens"] for item in items])
    texts = [item["text"] for item in items]
    return {
        "feats": feats,
        "feat_lens": feat_lens,
        "tokens": tokens,
        "token_lens": token_lens,
        "texts": texts,
    }


class LibriSpeechDataModule:
    def __init__(self, data_cfg: DataConfig, cache_dir: Path, batch_size: int, num_workers: int, seed: int):
        self.data_cfg = data_cfg
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_ds: MelDataset | None = None
        self.valid_ds: MelDataset | None = None

    @property
    def fingerprint_dir(self) -> Path:
        payload = json.dumps(self.data_cfg.__dict__, sort_keys=True)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
        return self.cache_dir / digest

    def prepare(self) -> Path:
        target = self.fingerprint_dir
        meta_path = target / "meta.json"
        if meta_path.exists():
            print(f"Using cached subset at {target}")
            return target
        target.mkdir(parents=True, exist_ok=True)
        extractor = LogMelExtractor(
            sample_rate=self.data_cfg.sample_rate,
            n_mels=self.data_cfg.n_mels,
            n_fft=self.data_cfg.n_fft,
            hop_length=self.data_cfg.hop_length,
            win_length=self.data_cfg.win_length,
        )
        extractor.eval()
        print("Streaming LibriSpeech clean. Keeping a small subset for a T4 session.")
        train_stats = _materialize_split(
            dataset=self.data_cfg.dataset,
            config=self.data_cfg.dataset_config,
            split=self.data_cfg.train_split,
            limit=self.data_cfg.max_train_utterances,
            data_cfg=self.data_cfg,
            extractor=extractor,
            destination=target / "train.pt",
        )
        valid_stats = _materialize_split(
            dataset=self.data_cfg.dataset,
            config=self.data_cfg.dataset_config,
            split=self.data_cfg.valid_split,
            limit=self.data_cfg.max_valid_utterances,
            data_cfg=self.data_cfg,
            extractor=extractor,
            destination=target / "valid.pt",
        )
        meta = {"train": train_stats, "valid": valid_stats}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(
            f"Cached {train_stats['kept']} train and {valid_stats['kept']} valid utterances in {target}"
        )
        return target

    def setup(self) -> None:
        augment = None
        if self.data_cfg.specaugment:
            augment = {
                "freq_mask_width": self.data_cfg.freq_mask_width,
                "time_mask_width": self.data_cfg.time_mask_width,
                "n_freq_masks": self.data_cfg.n_freq_masks,
                "n_time_masks": self.data_cfg.n_time_masks,
            }
        folder = self.fingerprint_dir
        self.train_ds = MelDataset(folder / "train.pt", augment=augment)
        self.valid_ds = MelDataset(folder / "valid.pt", augment=None)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        sampler = BucketBatchSampler(
            self.train_ds.lengths, self.batch_size, shuffle=True, seed=self.seed
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.valid_ds is not None
        sampler = BucketBatchSampler(
            self.valid_ds.lengths, self.batch_size, shuffle=False, seed=self.seed
        )
        return DataLoader(
            self.valid_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )


def _materialize_split(
    dataset: str,
    config: str,
    split: str,
    limit: int,
    data_cfg: DataConfig,
    extractor: LogMelExtractor,
    destination: Path,
) -> dict:
    from datasets import Audio, load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True)
    try:
        stream = stream.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"Audio cast skipped ({exc}). Decoding whatever the dataset returns.")

    mels: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    texts: list[str] = []
    seen = 0
    skipped = 0
    for example in stream:
        seen += 1
        try:
            waveform, sample_rate = _read_audio(example["audio"])
        except Exception:
            skipped += 1
            continue
        duration = waveform.shape[0] / float(sample_rate)
        if duration < data_cfg.min_duration_sec or duration > data_cfg.max_duration_sec:
            skipped += 1
            continue
        if sample_rate != data_cfg.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sample_rate, data_cfg.sample_rate)
        text = normalize_text(example["text"])
        token_ids = tokenize(text)
        if not token_ids:
            skipped += 1
            continue
        with torch.no_grad():
            mel = extractor(waveform)
        out_len = int(subsampled_lengths(torch.tensor([mel.shape[0]])).item())
        # CTC needs at least as many encoder frames as target tokens.
        if out_len < len(token_ids):
            skipped += 1
            continue
        mels.append(mel.half().cpu())
        tokens.append(torch.tensor(token_ids, dtype=torch.long))
        texts.append(text)
        if len(mels) % 50 == 0:
            print(f"  {split}: kept {len(mels)} / scanned {seen}")
        if len(mels) >= limit:
            break
    if not mels:
        raise RuntimeError(f"No utterances kept for split {split}. Check duration filters and network.")
    torch.save({"mels": mels, "tokens": tokens, "texts": texts}, destination)
    return {"kept": len(mels), "scanned": seen, "skipped": skipped}


def _read_audio(audio) -> tuple[torch.Tensor, int]:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        waveform = samples.data.squeeze().float()
        return waveform, int(samples.sample_rate)
    if isinstance(audio, dict) and audio.get("array") is not None:
        waveform = torch.tensor(np.asarray(audio["array"]), dtype=torch.float32)
        sample_rate = int(audio["sampling_rate"])
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        import soundfile as sf

        decoded, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        waveform = torch.tensor(np.asarray(decoded), dtype=torch.float32)
    else:
        raise TypeError(f"Unrecognized audio payload: {type(audio)}")
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=-1)
    return waveform, sample_rate
