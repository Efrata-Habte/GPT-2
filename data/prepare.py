"""Download Tiny Shakespeare (or use your own text) and write train/val .bin files.

Usage:
    python data/prepare.py                                   # char-level  -> data/input_char/
    python data/prepare.py --tokenizer bpe --bpe_vocab_size 1000   # BPE -> data/input_bpe1000/
    python data/prepare.py --input my_corpus.txt             # another corpus (not downloaded)
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import default_data_dir, prepare_dataset  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path(__file__).parent / "input.txt"))
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--bpe_vocab_size", type=int, default=512)
    ap.add_argument("--out_dir", default=None, help="default: next to the input, e.g. data/input_char")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        if inp.name != "input.txt":
            raise SystemExit(f"{inp} not found")
        inp.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} -> {inp}")
        urllib.request.urlretrieve(URL, inp)
    text = inp.read_text(encoding="utf-8")
    print(f"{inp}: {len(text):,} characters, {len(set(text))} unique")

    out = Path(args.out_dir) if args.out_dir else default_data_dir(inp, args.tokenizer, args.bpe_vocab_size)
    meta = prepare_dataset(text, out, args.tokenizer, args.bpe_vocab_size)
    print(f"wrote {out}/: {meta}")


if __name__ == "__main__":
    main()
