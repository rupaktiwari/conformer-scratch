"""Shape, shift, and memorization checks for the Gulati block.

Run from the repo root:

    PYTHONPATH=src python tests/test_architecture.py
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.nn.functional import ctc_loss
from torch.utils.data import DataLoader, Dataset

from conformer_asr.asr import ConformerASR, greedy_hypotheses
from conformer_asr.data import collate_batch
from conformer_asr.attention import rel_shift
from conformer_asr.encoder import ConformerEncoder
from conformer_asr.subsampling import subsampled_lengths
from conformer_asr.text import ctc_greedy_decode


def test_rel_shift_shape_and_view():
    scores = torch.arange(15, dtype=torch.float32).view(1, 1, 3, 5)
    shifted = rel_shift(scores)
    assert shifted.shape == (1, 1, 3, 3)
    assert torch.isfinite(shifted).all()


def test_subsampling_lengths_match_forward():
    encoder = ConformerEncoder(
        n_mels=80,
        d_model=32,
        num_heads=4,
        num_layers=1,
        conv_kernel_size=7,
        subsampling_channels=16,
        dropout=0.0,
    )
    encoder.eval()
    feats = torch.randn(2, 160, 80)
    lens = torch.tensor([160, 90])
    hidden, out_lens = encoder(feats, lens)
    assert hidden.shape[0] == 2
    assert hidden.shape[1] == int(subsampled_lengths(torch.tensor([160])).item())
    assert hidden.shape[2] == 32
    assert out_lens.tolist() == subsampled_lengths(lens).tolist()
    # Right-padded frames stay zero after the block mask.
    assert torch.count_nonzero(hidden[1, out_lens[1] :]) == 0


def test_ctc_backward_and_greedy_collapse():
    torch.manual_seed(0)
    model = ConformerASR(
        model_cfg=dict(
            n_mels=80,
            d_model=32,
            num_heads=4,
            num_layers=2,
            ffn_expansion=2,
            conv_kernel_size=7,
            dropout=0.0,
            subsampling_channels=16,
        ),
        train_cfg=dict(lr_warmup_steps=10),
    )
    model.train()
    feats = torch.randn(2, 120, 80)
    feat_lens = torch.tensor([120, 100])
    tokens = torch.tensor([1, 2, 3, 1, 2])
    token_lens = torch.tensor([3, 2])
    batch = {
        "feats": feats,
        "feat_lens": feat_lens,
        "tokens": tokens,
        "token_lens": token_lens,
        "texts": ["abc", "ab"],
    }
    opt_config = model.configure_optimizers()
    assert opt_config["optimizer"].param_groups[0]["lr"] > 0
    loss, logits, out_lens = model._loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.layers[0].attn.pos_bias_u.grad is not None
    hyps = greedy_hypotheses(logits, out_lens, model.vocab)
    assert len(hyps) == 2
    assert ctc_greedy_decode([0, 0, 1, 1, 0, 2, 2, 3]) == "abc"


def test_one_batch_loss_drops():
    torch.manual_seed(1)
    model = ConformerASR(
        model_cfg=dict(
            n_mels=80,
            d_model=32,
            num_heads=4,
            num_layers=2,
            ffn_expansion=2,
            conv_kernel_size=7,
            dropout=0.0,
            subsampling_channels=16,
        ),
        train_cfg=dict(lr_warmup_steps=10),
    )
    model.train()
    feats = torch.randn(2, 80, 80)
    feat_lens = torch.tensor([80, 80])
    tokens = torch.tensor([1, 2, 1, 2])
    token_lens = torch.tensor([2, 2])
    batch = {
        "feats": feats,
        "feat_lens": feat_lens,
        "tokens": tokens,
        "token_lens": token_lens,
        "texts": ["ab", "ab"],
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        loss, _, _ = model._loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.75, losses


class _ToyMel(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "mel": torch.randn(48, 80),
            "tokens": torch.tensor([1, 2, 3], dtype=torch.long),
            "text": "abc",
        }


def test_lightning_loop_logs_error_rates():
    torch.manual_seed(0)
    model = ConformerASR(
        model_cfg=dict(
            n_mels=80,
            d_model=32,
            num_heads=4,
            num_layers=1,
            ffn_expansion=2,
            conv_kernel_size=7,
            dropout=0.0,
            subsampling_channels=16,
        ),
        train_cfg=dict(lr_warmup_steps=4),
    )
    loader = DataLoader(_ToyMel(), batch_size=2, collate_fn=collate_batch)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader, loader)
    assert model.metric_history, "validation did not record WER/CER"
    assert "val/wer" in model.metric_history[-1]
    assert "val/cer" in model.metric_history[-1]


def test_direct_ctc_reference_is_finite():
    # Guards the log-prob layout expected by nn.CTCLoss: (time, batch, vocab).
    logits = torch.randn(2, 20, 5, requires_grad=True)
    log_probs = logits.log_softmax(-1).transpose(0, 1)
    loss = ctc_loss(
        log_probs,
        torch.tensor([1, 2, 1, 2]),
        torch.tensor([20, 20]),
        torch.tensor([2, 2]),
        blank=0,
        zero_infinity=True,
    )
    loss.backward()
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_rel_shift_shape_and_view()
    test_subsampling_lengths_match_forward()
    test_ctc_backward_and_greedy_collapse()
    test_direct_ctc_reference_is_finite()
    test_one_batch_loss_drops()
    test_lightning_loop_logs_error_rates()
    print("architecture checks passed")
