"""Character targets for CTC.

LibriSpeech transcripts are read speech in English. A fixed character inventory
is enough for a small subset and keeps the softmax tiny (29 classes).
"""

from __future__ import annotations

BLANK = "<blank>"
# Index 0 is the CTC blank. Letters, space, and apostrophe are the rest.
VOCAB: list[str] = [BLANK] + [chr(code) for code in range(ord("a"), ord("z") + 1)] + [" ", "'"]
CHAR2ID: dict[str, int] = {char: index for index, char in enumerate(VOCAB)}
ALLOWED = set(CHAR2ID) - {BLANK}


def normalize_text(text: str) -> str:
    """Lowercase LibriSpeech text and keep letters, spaces, and apostrophes."""
    text = text.lower().replace("-", " ")
    text = "".join(char for char in text if char in ALLOWED)
    return " ".join(text.split())


def tokenize(text: str) -> list[int]:
    normalized = normalize_text(text)
    return [CHAR2ID[char] for char in normalized]


def ids_to_text(token_ids: list[int]) -> str:
    chars = [VOCAB[index] for index in token_ids if 0 < index < len(VOCAB)]
    return "".join(chars).strip()


def ctc_greedy_decode(token_ids: list[int]) -> str:
    """Collapse CTC repeats and drop the blank."""
    collapsed: list[int] = []
    previous = None
    for index in token_ids:
        if index != previous and index != 0:
            collapsed.append(index)
        previous = index
    return ids_to_text(collapsed)
