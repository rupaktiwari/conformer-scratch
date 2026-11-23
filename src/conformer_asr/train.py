"""Train the small Conformer CTC model.

Tokens are read from the environment so the Colab notebook can set them
without writing secrets into this repository:

    WANDB_API_KEY
    HF_TOKEN
    HF_REPO_ID          optional override of configs/small.yaml
    WANDB_PROJECT       optional
    WANDB_ENTITY        optional
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from conformer_asr.asr import ConformerASR, count_parameters, evaluate_loader
from conformer_asr.config import AppConfig, load_config
from conformer_asr.data import LibriSpeechDataModule, MelDataset, collate_batch
from conformer_asr.hf_log import HuggingFaceLogCallback


def run(
    config_path: str | Path = "configs/small.yaml",
    max_epochs: int | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    max_train_utterances: int | None = None,
    max_valid_utterances: int | None = None,
    use_wandb: bool = True,
    use_hf: bool = True,
) -> dict:
    cfg = load_config(config_path)
    if max_epochs is not None:
        cfg.train.max_epochs = max_epochs
    if max_steps is not None:
        cfg.train.max_steps = max_steps
    if batch_size is not None:
        cfg.train.batch_size = batch_size
    if max_train_utterances is not None:
        cfg.data.max_train_utterances = max_train_utterances
    if max_valid_utterances is not None:
        cfg.data.max_valid_utterances = max_valid_utterances
    if os.environ.get("HF_REPO_ID"):
        cfg.log.hf_repo_id = os.environ["HF_REPO_ID"]
    if os.environ.get("WANDB_PROJECT"):
        cfg.log.wandb_project = os.environ["WANDB_PROJECT"]

    _require_logging(cfg, use_wandb, use_hf)
    pl.seed_everything(cfg.seed, workers=True)

    output_dir = Path(cfg.log.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = LibriSpeechDataModule(
        cfg.data,
        cache_dir=Path("data/cache"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        seed=cfg.seed,
    )
    data.prepare()
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    model = ConformerASR(asdict(cfg.model), asdict(cfg.train))
    print(f"Trainable parameters: {count_parameters(model):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    before = evaluate_loader(model, val_loader)
    print(
        f"Untrained val  WER {before['wer']:.3f}  CER {before['cer']:.3f}  loss {before['loss']:.3f}"
    )

    loggers = []
    callbacks = []
    if use_wandb:
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg.log.wandb_project,
                entity=os.environ.get("WANDB_ENTITY") or None,
                name="conformer-tiny-librispeech",
                save_dir=str(output_dir),
                log_model=False,
            )
        )
    hf_callback = None
    if use_hf:
        hf_callback = HuggingFaceLogCallback(
            repo_id=cfg.log.hf_repo_id,
            token=os.environ["HF_TOKEN"],
            output_dir=output_dir,
            private=cfg.log.hf_private,
        )
        callbacks.append(hf_callback)
        model.hf_logger = hf_callback

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    callbacks.append(checkpoint)
    callbacks.append(
        pl.callbacks.EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=cfg.train.early_stop_patience,
            check_on_train_epoch_end=False,
        )
    )
    if use_wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        max_steps=cfg.train.max_steps if cfg.train.max_steps is not None else -1,
        accelerator="auto",
        devices=1,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        log_every_n_steps=10,
        num_sanity_val_steps=1,
        default_root_dir=str(output_dir),
        enable_checkpointing=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint.best_model_path
    if best_path:
        model = ConformerASR.load_from_checkpoint(best_path)
    model.to(device)
    after = evaluate_loader(model, val_loader)
    train_probe = _probe_train(data, model, device)
    print(
        f"Best ckpt val   WER {after['wer']:.3f}  CER {after['cer']:.3f}  loss {after['loss']:.3f}"
    )
    print(
        f"Train sample    WER {train_probe['wer']:.3f}  CER {train_probe['cer']:.3f}  "
        f"({train_probe['utterances']} utterances, no SpecAugment)"
    )
    print("References and hypotheses from the validation set:")
    for item in after["examples"]:
        print(f"  REF: {item['ref']}")
        print(f"  HYP: {item['hyp']}")

    if use_wandb:
        import wandb

        if wandb.run is not None:
            wandb.summary["val/wer"] = after["wer"]
            wandb.summary["val/cer"] = after["cer"]
            wandb.summary["val/wer_untrained"] = before["wer"]
            wandb.summary["val/cer_untrained"] = before["cer"]
            wandb.summary["train_sample/cer"] = train_probe["cer"]
            wandb.finish()

    return {
        "wer": after["wer"],
        "cer": after["cer"],
        "loss": after["loss"],
        "wer_untrained": before["wer"],
        "cer_untrained": before["cer"],
        "train_sample_wer": train_probe["wer"],
        "train_sample_cer": train_probe["cer"],
        "examples": after["examples"],
        "checkpoint": best_path,
        "parameters": count_parameters(model),
        "hf_repo_id": cfg.log.hf_repo_id if use_hf else None,
    }


def _probe_train(data: LibriSpeechDataModule, model: ConformerASR, device: torch.device) -> dict:
    folder = data.fingerprint_dir
    dataset = MelDataset(folder / "train.pt", augment=None)
    keep = min(64, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(keep)))
    loader = DataLoader(subset, batch_size=8, collate_fn=collate_batch)
    model.to(device)
    scores = evaluate_loader(model, loader)
    scores["utterances"] = keep
    return scores


def _require_logging(cfg: AppConfig, use_wandb: bool, use_hf: bool) -> None:
    if use_wandb:
        key = os.environ.get("WANDB_API_KEY", "")
        if not key or "TOKEN HERE" in key:
            raise SystemExit("Set WANDB_API_KEY. The Colab notebook has a TOKEN HERE cell.")
        import wandb

        wandb.login(key=key, relogin=True)
    if use_hf:
        token = os.environ.get("HF_TOKEN", "")
        repo = cfg.log.hf_repo_id
        if not token or "TOKEN HERE" in token:
            raise SystemExit("Set HF_TOKEN. The Colab notebook has a TOKEN HERE cell.")
        if "YOUR_HF_USERNAME" in repo or not repo:
            raise SystemExit("Set HF_REPO_ID to '<your-username>/conformer-asr-tiny'.")
        from huggingface_hub import HfApi

        HfApi(token=token).whoami()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small Gulati Conformer CTC ASR model")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-utterances", type=int)
    parser.add_argument("--max-valid-utterances", type=int)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    args = parser.parse_args()
    run(
        config_path=args.config,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_train_utterances=args.max_train_utterances,
        max_valid_utterances=args.max_valid_utterances,
        use_wandb=not args.no_wandb,
        use_hf=not args.no_hf,
    )


if __name__ == "__main__":
    main()
