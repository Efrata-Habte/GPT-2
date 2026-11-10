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
    """Minimal Byte-Pair-Encoding tokenizer with two base alphabets (`level`):

    level="byte"  base vocabulary = the 256 possible bytes of UTF-8 (like GPT-2). ANY string can be
                  encoded, but a non-ASCII character starts out as 2-4 separate tokens, so early
                  merges are spent re-assembling characters (very visible for 3-byte Ge'ez).
    level="char"  base vocabulary = the characters of the alphabet given at training time. Sequences
                  start 1 token per character, but a character never seen in training raises KeyError
                  (there is no <unk>).

    Training repeatedly merges the most frequent adjacent pair of tokens into a new token.
    `self.merges` maps (left_id, right_id) -> new_id, in the order merges were learned; the new id
    doubles as the merge's *rank* (smaller = learned earlier = applied first).
    """

    def __init__(self, merges: dict[tuple[int, int], int] | None = None, level: str = "byte",
                 alphabet: list[str] | None = None) -> None:
        assert level in ("byte", "char"), level
        self.level = level
        self.alphabet: list[str] | None = (list(alphabet) if alphabet is not None else []) if level == "char" else None
        assert level == "byte" or self.alphabet or not merges, "char-level BPE with merges needs an alphabet"
        self.merges: dict[tuple[int, int], int] = dict(merges or {})
        self._build_vocab()
        self._cache: dict[str, list[int]] = {}

    # ---- vocabulary -----------------------------------------------------------------
    @property
    def base_size(self) -> int:
        """Number of base symbols: 256 bytes, or the size of the character alphabet."""
        return 256 if self.level == "byte" else len(self.alphabet or [])

    def _build_vocab(self) -> None:
        """id -> bytes. Merged tokens are the concatenation of their two parts."""
        if self.level == "byte":
            self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        else:
            self.char2id = {c: i for i, c in enumerate(self.alphabet or [])}
            self.vocab = {i: c.encode("utf-8") for c, i in self.char2id.items()}
        for (a, b), new_id in self.merges.items():  # dict preserves learning order
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return self.base_size + len(self.merges)

    def _base_ids(self, chunk: str) -> list[int]:
        """The chunk as a list of base-symbol ids (bytes, or character indices)."""
        if self.level == "byte":
            return list(chunk.encode("utf-8"))
        try:
            return [self.char2id[c] for c in chunk]
        except KeyError as e:
            raise KeyError(f"character {e.args[0]!r} is not in this tokenizer's alphabet") from None

    # ---- training -------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False, alphabet: str | list[str] | None = None) -> None:
        """Learn `vocab_size - base_size` merges from `text`.

        alphabet: (char level only) characters to include in the base vocabulary in addition to those in
        `text`, e.g. the characters of the validation split, so that encoding it never fails.
        """
        if self.level == "char":
            self.alphabet = sorted(set(text) | set(alphabet or ""))
        base = self.base_size
        assert vocab_size >= base, f"vocab_size must be >= {base} (the base alphabet)"
        self._build_vocab()  # base vocabulary only, so _base_ids works while training
        # {chunk: count}, then each unique chunk as a list of base ids.
        chunk_freq = Counter(_PRETOKENIZE.findall(text))
        words: dict[str, list[int]] = {w: self._base_ids(w) for w in chunk_freq}

        # Pair statistics are maintained INCREMENTALLY: after a merge only the chunks that contain the
        # merged pair change, so only those are re-counted (same result as recounting everything each
        # time, but orders of magnitude faster on large vocabularies / big corpora).
        pair_counts: Counter[tuple[int, int]] = Counter()      # pair -> total occurrences (weighted by chunk frequency)
        where: dict[tuple[int, int], set[str]] = {}            # pair -> chunks that contain it
        for w, ids in words.items():
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += chunk_freq[w]
                where.setdefault(pair, set()).add(w)

        self.merges = {}
        for new_id in range(base, vocab_size):
            if not pair_counts:
                break
            # Most frequent pair; ties broken by the pair itself so training is deterministic.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            best_count = pair_counts[best]
            if best_count < 2:  # nothing repeats any more -> further merges are useless
                break
            self.merges[best] = new_id
            for w in list(where.get(best, ())):
                f, old = chunk_freq[w], words[w]
                for pair in zip(old, old[1:]):                  # retract this chunk's old pairs ...
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(w)
                words[w] = _merge_pair(old, best, new_id)
                for pair in zip(words[w], words[w][1:]):        # ... and add its new ones
                    pair_counts[pair] += f
                    where.setdefault(pair, set()).add(w)
            where.pop(best, None)
            if verbose and (new_id - base + 1) % 100 == 0:
                print(f"merge {new_id - base + 1}: {best} -> {new_id} ({best_count} occurrences)")
        self._build_vocab()
        self._cache = {}

    def truncated(self, vocab_size: int) -> "BPETokenizer":
        """A tokenizer using only the first `vocab_size - base_size` merges.

        Because merges are learned greedily in order, training once with a big vocab and
        truncating gives exactly what training with the smaller vocab would have given.
        """
        keep = max(0, vocab_size - self.base_size)
        return BPETokenizer(dict(list(self.merges.items())[:keep]), self.level, self.alphabet)

    # ---- encode / decode ------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = self._base_ids(chunk)
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
    def to_dict(self) -> dict:
        d = {"type": "bpe" if self.level == "byte" else "bpe_char", "merges": [[a, b] for (a, b) in self.merges]}
        if self.level == "char":
            d["alphabet"] = self.alphabet
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        level = "byte" if d["type"] == "bpe" else "char"
        base = 256 if level == "byte" else len(d["alphabet"])
        merges = {(a, b): base + i for i, (a, b) in enumerate(d["merges"])}
        return cls(merges, level, d.get("alphabet"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


Tokenizer = CharTokenizer | BPETokenizer


def tokenizer_from_dict(d: dict) -> Tokenizer:
    """Rebuild a tokenizer from its dict form (type: "char" | "bpe" | "bpe_char")."""
    return CharTokenizer(d["chars"]) if d["type"] == "char" else BPETokenizer.from_dict(d)


def tokenizer_to_dict(tok: Tokenizer) -> dict:
    return {"type": "char", "chars": tok.chars} if isinstance(tok, CharTokenizer) else tok.to_dict()


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load whichever tokenizer type was saved at `path`."""
    return tokenizer_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def token_labels(tok: Tokenizer, ids: list[int], visible: bool = True) -> list[str]:
    """Human-readable strings for tokens, for axes/tables. Partial UTF-8 pieces (byte-level BPE) show as <E1 88>;
    with visible=True a space is drawn as '·' and a newline as '\\n'."""
    out = []
    for i in ids:
        if hasattr(tok, "vocab"):
            b = tok.vocab[i]
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                out.append("<" + b.hex(" ").upper() + ">")
                continue
        else:
            s = tok.decode([i])
        if visible:
            s = s.replace("\n", "\\n").replace(" ", "·")
        out.append(s)
    return out
