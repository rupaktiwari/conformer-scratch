"""YAML config for the small T4 run."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    n_mels: int = 80
    d_model: int = 144
    num_heads: int = 4
    num_layers: int = 4
    ffn_expansion: int = 4
    conv_kernel_size: int = 31
    dropout: float = 0.1
    subsampling_channels: int = 144


@dataclass
class DataConfig:
    dataset: str = "openslr/librispeech_asr"
    dataset_config: str = "clean"
    train_split: str = "train.100"
    valid_split: str = "validation"
    max_train_utterances: int = 1000
    max_valid_utterances: int = 200
    min_duration_sec: float = 2.0
    max_duration_sec: float = 10.0
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    specaugment: bool = True
    freq_mask_width: int = 15
    time_mask_width: int = 40
    n_freq_masks: int = 2
    n_time_masks: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 8
    max_epochs: int = 25
    max_steps: int | None = None
    lr_warmup_steps: int = 200
    gradient_clip_val: float = 5.0
    precision: str = "32"
    accumulate_grad_batches: int = 1
    num_workers: int = 0
    early_stop_patience: int = 10


@dataclass
class LogConfig:
    wandb_project: str = "conformer-asr"
    hf_repo_id: str = "YOUR_HF_USERNAME/conformer-asr-tiny"
    hf_private: bool = True
    output_dir: str = "outputs"


@dataclass
class AppConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    log: LogConfig


def _fill(cls, raw: dict):
    known = {item.name for item in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw["train"]["precision"] = str(raw["train"]["precision"])
    return AppConfig(
        seed=int(raw["seed"]),
        model=_fill(ModelConfig, raw["model"]),
        data=_fill(DataConfig, raw["data"]),
        train=_fill(TrainConfig, raw["train"]),
        log=_fill(LogConfig, raw["log"]),
    )
