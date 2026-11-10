"""Normalisation of Amharic homophone characters (Ge'ez script).

Amharic spelling keeps several letters that once had different sounds but are pronounced identically
today. Writers use them inconsistently, so the SAME word appears under several spellings, which inflates
the vocabulary and splits statistics. Normalisation folds every variant onto one canonical letter of its
family. Each family is a "row" of the Ethiopic block with 7 vowel orders (ä u i a e ə o) at fixed offsets:

    family  canonical        variants folded into it
    h       ሀ  U+1200        ሐ U+1210 , ኀ U+1280
    s       ሰ  U+1230        ሠ U+1220
    a       አ  U+12A0        ዐ U+12D0
    ts      ጸ  U+1338        ፀ U+1340

All 7 vowel orders move together (ሑ -> ሁ, ዓ -> አ ... via the same offset). The 8th slot of a row is a
different, labialised sound and is NOT folded. Ethiopic punctuation (። ፣ ፤ ...) is never touched, except
that the optional "wordspace" mode turns the old word separator ፡ (U+1361) into an ordinary space.

Folding is lossy by design: a model trained on normalised text cannot produce the dropped spellings.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

ORDERS = 7  # the first 7 slots of a row are the vowel orders; the 8th is labialised

FAMILIES: dict[str, dict] = {
    "h": {"canonical": 0x1200, "variants": (0x1210, 0x1280)},
    "s": {"canonical": 0x1230, "variants": (0x1220,)},
    "a": {"canonical": 0x12A0, "variants": (0x12D0,)},
    "ts": {"canonical": 0x1338, "variants": (0x1340,)},
}
WORDSPACE = "፡"  # ፡


def build_table(families: tuple[str, ...] = tuple(FAMILIES)) -> dict[int, int]:
    """str.translate table {variant codepoint: canonical codepoint} for the chosen families."""
    table: dict[int, int] = {}
    for name in families:
        fam = FAMILIES[name]
        for variant_base in fam["variants"]:
            for o in range(ORDERS):
                table[variant_base + o] = fam["canonical"] + o
    return table


_FULL_TABLE = build_table()


def normalize_homophones(text: str, families: tuple[str, ...] = tuple(FAMILIES)) -> str:
    """Fold homophone letters onto their canonical form. Same length as the input."""
    table = _FULL_TABLE if families == tuple(FAMILIES) else build_table(families)
    return text.translate(table)


def normalize_amharic(text: str, mode: str | None = "homophones") -> str:
    """mode: None/"none" (identity) | "homophones" | "homophones+wordspace"."""
    if mode in (None, "none", ""):
        return text
    if mode not in ("homophones", "homophones+wordspace"):
        raise ValueError(f"unknown normalization mode {mode!r}")
    text = unicodedata.normalize("NFC", text)
    text = normalize_homophones(text)
    if mode == "homophones+wordspace":
        text = text.replace(WORDSPACE, " ")
    return text


def homophone_counts(text: str) -> Counter:
    """How often each folded-away variant letter occurs in `text` (for reporting)."""
    table = _FULL_TABLE
    return Counter(ch for ch in text if ord(ch) in table)


def changed_positions(before: str, after: str) -> list[int]:
    """Indices where two equal-length strings differ (used to highlight changes in the demo)."""
    assert len(before) == len(after)
    return [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
