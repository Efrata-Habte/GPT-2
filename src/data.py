"""Dataset preparation (text -> token ids -> .bin) and random batch sampling.

On disk, a prepared dataset directory contains:
    train.bin, val.bin   flat arrays of token ids (uint16), the first 90% / last 10% of the text
    tokenizer.json       the fitted tokenizer (char vocab, char-level BPE or byte-level BPE merges)
    meta.json            vocab size, token and character counts (chars are needed for bits-per-char)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer


TOKENIZERS = ("char", "bpe", "bpe_byte", "bpe_char")   # "bpe" is an alias of "bpe_byte" (kept for old configs)


def default_data_dir(text_path: str | Path, tokenizer: str, bpe_vocab_size: int, normalize: str | None = None) -> Path:
    """<raw-file-dir>/<file-stem>_<tokenizer>[<vocab>][_norm], next to the raw text file."""
    p = Path(text_path)
    suffix = {"char": "char", "bpe": f"bpe{bpe_vocab_size}", "bpe_byte": f"bpe{bpe_vocab_size}",
              "bpe_char": f"cbpe{bpe_vocab_size}"}[tokenizer]
    if normalize and normalize != "none":
        suffix += "_norm"
    return p.parent / f"{p.stem}_{suffix}"


def data_dir_from_config(dc: dict) -> Path:
    """Prepared-dataset directory for a config's `data:` section."""
    if dc.get("dir"):
        return Path(dc["dir"])
    return default_data_dir(dc["path"], dc["tokenizer"], dc.get("bpe_vocab_size", 512), dc.get("normalize"))


def normalize_text(text: str, mode: str | None) -> str:
    """Optional Amharic homophone normalisation (data/amharic/normalize.py); identity when mode is None."""
    if mode in (None, "none", ""):
        return text
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from data.amharic.normalize import normalize_amharic
    return normalize_amharic(text, mode)


def prepare_dataset(
    text: str, out_dir: str | Path, tokenizer: str = "char", bpe_vocab_size: int = 512, val_frac: float = 0.1,
    normalize: str | None = None, bpe_train_chars: int | None = None,
) -> dict:
    """Split text 90/10 (contiguous), fit a tokenizer on the TRAIN part only, write .bin files.

    tokenizer: "char" | "bpe_char" (BPE over characters) | "bpe"/"bpe_byte" (BPE over UTF-8 bytes).
    normalize: optional text normalisation applied BEFORE splitting/tokenizing (see data/amharic/normalize.py).
    bpe_train_chars: learn the BPE merges on only the first N characters of the training split (much faster on big
        corpora; the whole corpus is still encoded with the result). None = use all of it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = normalize_text(text, normalize)
    split = int((1.0 - val_frac) * len(text))
    train_text, val_text = text[:split], text[split:]

    if tokenizer == "char":
        tok = CharTokenizer.from_text(train_text)
        # A rare char may appear only in val; give the vocab all chars so encode never fails.
        tok = CharTokenizer.from_text(text) if set(val_text) - set(tok.chars) else tok
    elif tokenizer in ("bpe", "bpe_byte"):
        tok = BPETokenizer(level="byte")
        tok.train(train_text[:bpe_train_chars], bpe_vocab_size)  # byte-level => val text can never contain unknown symbols
    elif tokenizer == "bpe_char":
        tok = BPETokenizer(level="char")
        tok.train(train_text[:bpe_train_chars], bpe_vocab_size, alphabet=set(text))  # alphabet always from the WHOLE text
    else:
        raise ValueError(f"unknown tokenizer {tokenizer!r}; choose from {TOKENIZERS}")

    assert tok.vocab_size < 2**16, "uint16 storage needs vocab_size < 65536"
    train_ids = np.array(tok.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tok.encode(val_text), dtype=np.uint16)
    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")
    tok.save(out / "tokenizer.json")
    meta = {
        "tokenizer": tokenizer,
        "normalize": normalize or "none",
        "bpe_train_chars": bpe_train_chars,
        "vocab_size": tok.vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def load_data(data_dir: str | Path) -> dict[str, np.ndarray]:
    """Memory-map train/val token arrays (no RAM copy)."""
    d = Path(data_dir)
    return {s: np.memmap(d / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}


def load_data_tokenizer(data_dir: str | Path):
    return load_tokenizer(Path(data_dir) / "tokenizer.json")


def get_batch(
    data: np.ndarray, batch_size: int, block_size: int, generator: torch.Generator, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows. y is x shifted left by one: y[:, t] = x[:, t+1] (next token).

    Returns x, y of shape (B, T), int64.
    """
    assert len(data) > block_size + 1, "dataset shorter than block_size + 1"
    # Valid start i needs i + block_size + 1 <= len(data) (y reads one token past x).
    ix = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])          # (B, T)
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])  # (B, T)
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
