"""Push metrics and the best weights to the Hugging Face Hub.

Each validation epoch appends one JSON line and uploads `metrics.jsonl`.
The fit-end callback uploads the best weights once. The repo is private
unless `hf_private` is set false in the config.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import pytorch_lightning as pl

from conformer_asr.text import VOCAB


class HuggingFaceLogCallback(pl.Callback):
    def __init__(self, repo_id: str, token: str, output_dir: Path, private: bool) -> None:
        self.repo_id = repo_id
        self.token = token
        self.output_dir = output_dir
        self.private = private
        self.rows: list[dict] = []
        self.best_model_path: str | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        who = api.whoami()
        print(f"Hugging Face login ok: {who.get('name') or who.get('fullname') or 'account'}")
        api.create_repo(self.repo_id, private=self.private, exist_ok=True)
        print(f"Hugging Face repo: https://huggingface.co/{self.repo_id}")

    def append_row(self, row: dict) -> None:
        """Called from the module after WER and CER for this epoch are known."""
        self.rows.append(row)
        path = self.output_dir / "metrics.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item) + "\n" for item in self.rows))
        self._upload(path, "metrics.jsonl")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        checkpoint_cb = trainer.checkpoint_callback
        if checkpoint_cb is not None and getattr(checkpoint_cb, "best_model_path", ""):
            self.best_model_path = checkpoint_cb.best_model_path
        weights = self.output_dir / "conformer_ctc.pt"
        checkpoint = self.best_model_path
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state_dict = state["state_dict"]
        else:
            state_dict = {key: value.detach().cpu() for key, value in pl_module.state_dict().items()}
        torch.save(
            {
                "state_dict": state_dict,
                "vocab": list(VOCAB),
                "hparams": dict(pl_module.hparams),
            },
            weights,
        )
        self._upload(weights, "conformer_ctc.pt")
        metrics = self.output_dir / "metrics.jsonl"
        if metrics.exists():
            self._upload(metrics, "metrics.jsonl")
        card = self.output_dir / "README.md"
        card.write_text(_model_card(self.repo_id, self.rows))
        self._upload(card, "README.md")
        print(f"Uploaded weights and metrics to https://huggingface.co/{self.repo_id}")

    def _upload(self, path: Path, repo_path: str) -> None:
        from huggingface_hub import HfApi

        try:
            HfApi(token=self.token).upload_file(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                commit_message=f"Update {repo_path}",
            )
        except Exception as exc:
            print(f"Hugging Face upload of {repo_path} failed: {exc}")


def _model_card(repo_id: str, rows: list[dict]) -> str:
    best = _best_row(rows)
    wer = _fmt(best.get("val/wer"))
    cer = _fmt(best.get("val/cer"))
    loss = _fmt(best.get("val/loss"))
    return f"""---
library_name: pytorch-lightning
tags:
- speech
- automatic-speech-recognition
- conformer
- ctc
license: cc-by-4.0
---

# {repo_id}

Small Gulati Conformer encoder with a CTC head, trained from scratch on a
subset of LibriSpeech clean (`openslr/librispeech_asr`, CC-BY-4.0).

This is a learning run sized for a free Colab T4. The checkpoint is the
validation-loss selection from that run.

| metric | value |
| --- | --- |
| val WER | {wer} |
| val CER | {cer} |
| val loss | {loss} |

Architecture: macaron feed-forward, relative-position self-attention, depthwise
convolution block (Gulati et al., Interspeech 2020). Decoder here is greedy CTC.
The paper's decoder is an RNN-Transducer.
"""


def _best_row(rows: list[dict]) -> dict:
    scored = [row for row in rows if "val/loss" in row]
    if not scored:
        return rows[-1] if rows else {}
    return min(scored, key=lambda row: row["val/loss"])


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
