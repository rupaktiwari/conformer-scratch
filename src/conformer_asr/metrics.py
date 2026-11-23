"""Corpus WER and CER.

jiwer scores the full validation list once per epoch, which is the usual
corpus-level word error rate rather than an average of per-utterance rates.
"""

from __future__ import annotations

import jiwer

from conformer_asr.text import normalize_text


def corpus_errors(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    refs = [normalize_text(text) for text in references]
    hyps = [normalize_text(text) for text in hypotheses]
    # jiwer rejects an empty reference string. Keep a placeholder space so a
    # blank hypothesis is still counted as a full error.
    refs = [text if text else " " for text in refs]
    return {
        "wer": float(jiwer.wer(refs, hyps)),
        "cer": float(jiwer.cer(refs, hyps)),
    }
