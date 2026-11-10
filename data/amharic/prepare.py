"""Build an Amharic (Ge'ez script) training corpus: data/amharic/amharic.txt

Source: MasakhaNEWS, Amharic news articles (Adelani et al., 2023), https://github.com/masakhane-io/masakhane-news
        Check the dataset licence (I believe CC BY-NC 4.0) before redistributing; data/ is git-ignored, so the
        text is never committed. You can also pass your own Amharic text via --input and skip the download.

Each article becomes "headline\\ntext"; articles are separated by a blank line, de-duplicated and SHUFFLED
(fixed seed) because the source is grouped by news category and the 90/10 split used later is contiguous.

Usage (from the repo root):
    python -I data/amharic/prepare.py                    # download + build
    python -I data/amharic/prepare.py --input my.txt     # use your own text file instead
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/masakhane-io/masakhane-news/main/data/amh/{}.tsv"
ETHIOPIC = re.compile(r"[ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯]")   # Ge'ez blocks


def read_articles(raw_dir: Path, download: bool) -> list[str]:
    arts: list[str] = []
    csv.field_size_limit(sys.maxsize)
    for split in ("train", "dev", "test"):
        p = raw_dir / f"{split}.tsv"
        if not p.exists():
            if not download:
                raise SystemExit(f"{p} missing (re-run with downloads enabled)")
            raw_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL.format(split), p)
        with open(p, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                head, body = (row.get("headline") or "").strip(), (row.get("text") or "").strip()
                text = re.sub(r"\s+", " ", f"{head}\n{body}".replace("\n", " \n ")).replace(" \n ", "\n").strip()
                if text:
                    arts.append(text)
    return arts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/amharic_download")
    ap.add_argument("--input", default=None, help="your own Amharic .txt file (skips the download)")
    ap.add_argument("--out", default="data/amharic/amharic.txt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.input:
        text = Path(a.input).read_text(encoding="utf-8")
    else:
        arts = list(dict.fromkeys(read_articles(Path(a.raw_dir), download=True)))   # de-duplicate, keep order
        random.Random(a.seed).shuffle(arts)
        text = "\n\n".join(arts) + "\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n_eth = len(ETHIOPIC.findall(text))
    print(f"wrote {out}: {len(text):,} chars ({len(text.encode('utf-8')):,} UTF-8 bytes), "
          f"{len(set(text))} unique chars, {100 * n_eth / len(text):.1f}% Ethiopic")


if __name__ == "__main__":
    main()
