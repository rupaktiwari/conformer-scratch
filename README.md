# Gulati Conformer, small CTC ASR

A from-scratch PyTorch implementation of the **Conformer encoder** from
[Gulati et al., Interspeech 2020](https://arxiv.org/abs/2005.08100), trained as
a character CTC speech recognizer on a **subset of LibriSpeech**. The default
config is sized for a **free Colab T4 (16 GB)**.

The paper's decoder is an RNN-Transducer. This repo keeps their encoder and
uses CTC so one free session can finish and report **WER** and **CER**.

## The block

Each Conformer block is a macaron pair of feed-forward modules around relative
self-attention and a convolution module. Pre-norm sits inside each module.
`src/conformer_asr/encoder.py` is the equation:

```text
x = x + 1/2 FFN(x)
x = x + MHSA(x)     relative sinusoidal positions (Transformer-XL)
x = x + Conv(x)     pointwise + GLU, depthwise conv, batch norm, Swish
x = x + 1/2 FFN(x)
x = LayerNorm(x)
```

| Paper piece | Code |
| --- | --- |
| Feed-forward, Swish, expansion 4 | `feedforward.py` |
| Relative multi-head self-attention | `attention.py` |
| Convolution module | `convolution.py` |
| Block and encoder stack | `encoder.py` |
| Stride-2 conv subsampling (4x) | `subsampling.py` |
| CTC head, greedy decode, WER/CER | `asr.py`, `metrics.py` |

Input is an 80-dim log-mel filterbank (25 ms window, 10 ms hop), per-utterance
normalized. SpecAugment runs on the training set only.

The depthwise kernel is **31**. The paper reports 32; an odd kernel keeps
same-padding symmetric.

## What a T4 run actually is

`configs/small.yaml`:

- 4 blocks, d_model 144, 4 heads, about 2.6 million parameters
- 1000 train / 200 validation utterances from
  [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
  (`clean`), duration 2–10 seconds
- batch size 8, fp32, Adam with the Vaswani schedule, 25 epochs
- streaming download stops after the subset is full (on the order of 0.5–1 GB,
  not the full 100 hours)

LibriSpeech is CC-BY-4.0. This code is MIT.

On a couple of hours of speech, validation WER stays high. The check that the
system works is: training loss falls, and validation CER moves down from the
untrained baseline the trainer prints first (that baseline is near 1.0 because
a random CTC model emits blanks). A sample of the training set is scored at the
end as well, without SpecAugment, so you can see whether the net memorized
anything.

## Colab

1. Push this repo to GitHub (public, unless the notebook's clone URL includes credentials).
2. Open `notebooks/train_conformer_colab.ipynb` in Colab with a **T4** runtime.
3. Fill in the **TOKEN HERE** cell: WandB key, Hugging Face token, GitHub repo URL, and `HF_REPO_ID`.
4. Run all cells. The notebook clones the repo, installs it, trains, prints WER and CER, logs to WandB, and uploads `conformer_ctc.pt` plus `metrics.jsonl` to a private Hugging Face model repo.

## Layout

```text
src/conformer_asr/    encoder, CTC model, data, training entry point
configs/small.yaml    T4 defaults
notebooks/            Colab notebook
tests/test_architecture.py
```

Local shape check, no dataset and no GPU:

```bash
pip install torch torchaudio pytorch-lightning jiwer pyyaml
PYTHONPATH=src python tests/test_architecture.py
```

`torch` is not pinned in `pyproject.toml` so a Colab install does not replace
the CUDA build that the runtime already has.
