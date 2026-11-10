"""Download Tiny Shakespeare (Phase 1). Tokenizing + writing train/val .bin is added in Phase 4.

Usage:
    python data/prepare.py                       # downloads to data/input.txt
    python data/prepare.py --out data/input.txt
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "input.txt"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        print(f"Downloading {URL} -> {out}")
        urllib.request.urlretrieve(URL, out)
    text = out.read_text(encoding="utf-8")
    print(f"{out}: {len(text):,} characters, {len(set(text))} unique characters")


if __name__ == "__main__":
    main()
