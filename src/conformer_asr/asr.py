"""CTC ASR model on top of the Gulati Conformer encoder.

The paper decodes with an RNN-Transducer. CTC is a single softmax over
characters plus a blank, which fits a 16 GB T4 and still yields WER and CER.
"""

from __future__ import annotations

import torch
from torch import nn
import pytorch_lightning as pl

from conformer_asr.encoder import ConformerEncoder
from conformer_asr.metrics import corpus_errors
from conformer_asr.text import VOCAB, ctc_greedy_decode


class ConformerASR(pl.LightningModule):
    def __init__(self, model_cfg: dict, train_cfg: dict, vocab: list[str] | None = None) -> None:
        super().__init__()
        model_cfg = dict(model_cfg)
        train_cfg = dict(train_cfg)
        vocab = list(vocab or VOCAB)
        # Capture the plain dicts above so load_from_checkpoint can rebuild the net.
        self.save_hyperparameters()
        self.vocab = vocab
        self.train_cfg = train_cfg
        self.encoder = ConformerEncoder(**model_cfg)
        self.ctc_head = nn.Linear(model_cfg["d_model"], len(vocab))
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self._val_refs: list[str] = []
        self._val_hyps: list[str] = []
        self._val_loss_sum = 0.0
        self._val_count = 0
        self.examples: list[dict] = []
        self.metric_history: list[dict] = []

    def forward(
        self, feats: torch.Tensor, feat_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, out_lens = self.encoder(feats, feat_lens)
        return self.ctc_head(hidden), out_lens

    def _loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, out_lens = self(batch["feats"], batch["feat_lens"])
        log_probs = torch.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        loss = self.criterion(
            log_probs,
            batch["tokens"],
            out_lens,
            batch["token_lens"],
        )
        return loss, logits, out_lens

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._loss(batch)
        batch_size = batch["feats"].size(0)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        self.log("lr", optimizer.param_groups[0]["lr"], on_step=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits, out_lens = self._loss(batch)
        batch_size = batch["feats"].size(0)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size, sync_dist=False)
        self._val_loss_sum += float(loss.detach()) * batch_size
        self._val_count += batch_size
        hyps = greedy_hypotheses(logits, out_lens, self.vocab)
        self._val_refs.extend(batch["texts"])
        self._val_hyps.extend(hyps)

    def on_validation_epoch_end(self) -> None:
        if self.trainer is not None and self.trainer.sanity_checking:
            self._clear_val()
            return
        if not self._val_refs:
            self._clear_val()
            return
        scores = corpus_errors(self._val_refs, self._val_hyps)
        self.log("val/wer", scores["wer"], prog_bar=True, batch_size=len(self._val_refs))
        self.log("val/cer", scores["cer"], prog_bar=True, batch_size=len(self._val_refs))
        self.examples = [
            {"ref": ref, "hyp": hyp}
            for ref, hyp in list(zip(self._val_refs, self._val_hyps))[:8]
        ]
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "val/wer": scores["wer"],
            "val/cer": scores["cer"],
            "val/loss": self._val_loss_sum / max(self._val_count, 1),
        }
        self.metric_history.append(row)
        hf_logger = getattr(self, "hf_logger", None)
        if hf_logger is not None:
            hf_logger.append_row(row)
        self._log_examples()
        self._clear_val()

    def _clear_val(self) -> None:
        self._val_refs.clear()
        self._val_hyps.clear()
        self._val_loss_sum = 0.0
        self._val_count = 0

    def _log_examples(self) -> None:
        try:
            import wandb

            if wandb.run is None or not self.examples:
                return
            table = wandb.Table(
                columns=["reference", "hypothesis"],
                data=[[item["ref"], item["hyp"]] for item in self.examples],
            )
            wandb.log({"val/examples": table})
        except Exception:
            return

    def configure_optimizers(self):
        # Gulati et al. use Adam (0.9, 0.98) and the Vaswani learning-rate schedule.
        # Base lr is 1 so the lambda below is the actual learning rate.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        d_model = self.hparams.model_cfg["d_model"]
        warmup = max(int(self.train_cfg["lr_warmup_steps"]), 1)

        def lr_lambda(step: int) -> float:
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def greedy_hypotheses(logits: torch.Tensor, lengths: torch.Tensor, vocab: list[str]) -> list[str]:
    pred = logits.argmax(dim=-1).detach().cpu()
    lengths = lengths.detach().cpu()
    hyps = []
    for index in range(pred.size(0)):
        limit = int(lengths[index])
        hyps.append(ctc_greedy_decode(pred[index, :limit].tolist()))
    return hyps


@torch.no_grad()
def evaluate_loader(model: ConformerASR, loader) -> dict:
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    total_loss = 0.0
    total_items = 0
    device = model.device
    for batch in loader:
        batch = {
            "feats": batch["feats"].to(device),
            "feat_lens": batch["feat_lens"].to(device),
            "tokens": batch["tokens"].to(device),
            "token_lens": batch["token_lens"].to(device),
            "texts": batch["texts"],
        }
        loss, logits, out_lens = model._loss(batch)
        total_loss += float(loss) * batch["feats"].size(0)
        total_items += batch["feats"].size(0)
        refs.extend(batch["texts"])
        hyps.extend(greedy_hypotheses(logits, out_lens, model.vocab))
    scores = corpus_errors(refs, hyps)
    scores["loss"] = total_loss / max(total_items, 1)
    scores["examples"] = [{"ref": ref, "hyp": hyp} for ref, hyp in list(zip(refs, hyps))[:8]]
    return scores


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
