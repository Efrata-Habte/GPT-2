"""Tokenizers: a character-level tokenizer and a minimal byte-level BPE tokenizer.

Both expose the same interface so the rest of the code does not care which is used:
    vocab_size: int
    encode(text: str) -> list[int]
    decode(ids: list[int]) -> str
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


# --------------------------------------------------------------------------------------
# Character-level tokenizer
# --------------------------------------------------------------------------------------
class CharTokenizer:
    """One token per unique character in the training text.

    Tiny vocab (65 for Tiny Shakespeare) but long sequences: 1 token == 1 character.
    Characters never seen at fit time raise KeyError in `encode` (no <unk> token).
    """

    def __init__(self, chars: list[str]) -> None:
        self.chars: list[str] = list(chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build the vocabulary from the sorted set of characters in `text`."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "char", "chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])


# --------------------------------------------------------------------------------------
# Byte-level BPE tokenizer
# --------------------------------------------------------------------------------------
# Pre-tokenization: split text into "chunks" (words with their leading space, punctuation
# runs, whitespace runs). Merges never cross chunk boundaries, which (a) keeps tokens
# meaningful and (b) lets us train on a {chunk: frequency} table instead of the whole text.
# This is a simplified version of GPT-2's regex (which uses \p{L}/\p{N} from the `regex`
# package; we use stdlib `re` and \w instead). The alternatives cover EVERY character, so
# "".join(chunks) == text always holds (checked in tests).
_PRETOKENIZE = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+""")


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` by `new_id` (left to right)."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Minimal byte-level Byte-Pair-Encoding tokenizer.

    Base vocabulary = the 256 possible bytes, so ANY string can be encoded (no <unk>).
    Training repeatedly merges the most frequent adjacent pair of tokens into a new token.
    `self.merges` maps (left_id, right_id) -> new_id, in the order merges were learned;
    the new id doubles as the merge's *rank* (smaller = learned earlier = applied first).
    """

    def __init__(self, merges: dict[tuple[int, int], int] | None = None) -> None:
        self.merges: dict[tuple[int, int], int] = dict(merges or {})
        self._build_vocab()
        self._cache: dict[str, list[int]] = {}

    def _build_vocab(self) -> None:
        """id -> bytes. Merged tokens are the concatenation of their two parts."""
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():  # dict preserves learning order
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---- training -------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text`."""
        assert vocab_size >= 256, "vocab_size must be >= 256 (the byte alphabet)"
        # {chunk: count}, then each unique chunk as a list of byte ids.
        chunk_freq = Counter(_PRETOKENIZE.findall(text))
        words: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in chunk_freq}

        self.merges = {}
        for new_id in range(256, vocab_size):
            # Count adjacent pairs, weighting each chunk by how often it occurs.
            pair_counts: Counter[tuple[int, int]] = Counter()
            for w, ids in words.items():
                f = chunk_freq[w]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += f
            if not pair_counts:
                break
            # Most frequent pair; ties broken by the pair itself so training is deterministic.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < 2:  # nothing repeats any more -> further merges are useless
                break
            self.merges[best] = new_id
            for w in words:
                words[w] = _merge_pair(words[w], best, new_id)
            if verbose and (new_id - 255) % 100 == 0:
                print(f"merge {new_id - 255}: {best} -> {new_id} ({pair_counts[best]} occurrences)")
        self._build_vocab()
        self._cache = {}

    def truncated(self, vocab_size: int) -> "BPETokenizer":
        """A tokenizer using only the first `vocab_size - 256` merges.

        Because merges are learned greedily in order, training once with a big vocab and
        truncating gives exactly what training with the smaller vocab would have given.
        """
        keep = max(0, vocab_size - 256)
        return BPETokenizer(dict(list(self.merges.items())[:keep]))

    # ---- encode / decode ------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Among adjacent pairs that have a learned merge, apply the EARLIEST-learned one.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge_pair(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _PRETOKENIZE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        # Join bytes first, decode UTF-8 once (a multi-byte char may span several tokens).
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---- persistence ----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {"type": "bpe", "merges": [[a, b] for (a, b) in self.merges]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        merges_list = json.loads(Path(path).read_text(encoding="utf-8"))["merges"]
        return cls({(a, b): 256 + i for i, (a, b) in enumerate(merges_list)})


def load_tokenizer(path: str | Path) -> CharTokenizer | BPETokenizer:
    """Load whichever tokenizer type was saved at `path`."""
    kind = json.loads(Path(path).read_text(encoding="utf-8"))["type"]
    return CharTokenizer.load(path) if kind == "char" else BPETokenizer.load(path)
