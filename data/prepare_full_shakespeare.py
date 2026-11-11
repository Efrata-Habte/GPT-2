"""Build the "full data" English corpus: the Complete Works of Shakespeare, in Tiny Shakespeare's format.

Outputs
    data/shakespeare_full.txt          training corpus for the full-data model (its own 90/10 split happens in training)
    data/test_sets/english_test.txt    shared held-out test text: Shakespeare passages that are NOT in Tiny Shakespeare
                                       and NOT in shakespeare_full.txt, so both the small- and the full-data model
                                       can be scored on it fairly.

Cleaning, so the two corpora look alike and differ only in AMOUNT:
    * Gutenberg header/footer and table of contents removed
    * speaker lines `ROMEO.` -> `ROMEO:`; stage directions (`[_Exit._]`, ` Enter Romeo.`) and ACT/SCENE headings dropped
    * curly quotes -> straight, em dash -> '--', accents folded (é -> e, æ -> ae)
    * only Tiny Shakespeare's 65 characters are kept

Source: Project Gutenberg eBook #100 (public domain). Usage (from the repo root):
    python -I data/prepare_full_shakespeare.py
"""
from __future__ import annotations

import argparse
import random
import re
import unicodedata
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
FOLD = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "...", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE"}
SPEAKER = re.compile(r"^([A-Z][A-Z0-9 ,'&\-]*[A-Z])\.$")          # ROMEO.  FIRST CITIZEN.  LADY CAPULET.
STAGE = re.compile(r"^\s*(\[_|_\]|(Enter|Exit|Exeunt|Re-enter|Flourish|Alarum)\b)")
HEADING = re.compile(r"^\s*(ACT|SCENE|Scene|PROLOGUE|EPILOGUE|Dramatis Person|DRAMATIS PERSON)\b")


def download(url: str, dest: Path) -> None:
    """Download to `dest`. Wikimedia (and some other hosts) answer 403 to Python's default user agent, so send a
    descriptive one; write to a temporary file first so an interrupted download never leaves a broken file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gpt2-from-scratch-data-prep/1.0 (educational project; Python urllib)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def strip_gutenberg(raw: str) -> str:
    start = raw.find("*** START")
    end = raw.find("*** END")
    body = raw[raw.find("\n", start) + 1:end] if start >= 0 and end > start else raw
    # skip the table of contents: the real text starts at the second, un-indented "THE SONNETS"
    hits = [m.start() for m in re.finditer(r"^THE SONNETS\s*$", body, flags=re.M)]
    return body[hits[1]:] if len(hits) > 1 else body


def to_tiny_format(text: str, alphabet: set[str]) -> str:
    out = []
    for line in text.splitlines():
        if STAGE.match(line) or HEADING.match(line):
            continue
        line = line.strip()
        m = SPEAKER.match(line)
        if m:
            line = m.group(1) + ":"
        for a, b in FOLD.items():
            line = line.replace(a, b)
        line = "".join(c for c in unicodedata.normalize("NFKD", line) if not unicodedata.combining(c))
        line = "".join(c for c in line if c in alphabet)
        out.append(line.rstrip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def make_blocks(text: str, size: int = 2000) -> list[str]:
    """Group paragraphs (blank-line separated) into ~size-character blocks."""
    blocks, cur = [], []
    for para in text.split("\n\n"):
        cur.append(para)
        if sum(len(p) for p in cur) >= size:
            blocks.append("\n\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n\n".join(cur))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/shakespeare_download/pg100.txt")
    ap.add_argument("--tiny", default="data/input.txt")
    ap.add_argument("--out", default="data/shakespeare_full.txt")
    ap.add_argument("--test_out", default="data/test_sets/english_test.txt")
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw_p = Path(a.raw)
    if not raw_p.exists():
        print(f"downloading {URL}")
        download(URL, raw_p)
    tiny = Path(a.tiny).read_text(encoding="utf-8")
    alphabet = set(tiny)
    text = to_tiny_format(strip_gutenberg(raw_p.read_text(encoding="utf-8-sig")), alphabet)

    # Lines of Tiny Shakespeare (long enough to be distinctive) tell us which blocks it already contains.
    # Compare letters only (case/punctuation differ between editions), so the overlap check is not fooled by commas.
    def key(line: str) -> str:
        return re.sub(r"[^a-z]", "", line.lower())

    tiny_lines = {key(ln) for ln in tiny.splitlines() if len(key(ln)) >= 15}
    blocks = make_blocks(text)

    def overlap(block: str) -> float:
        lines = [key(ln) for ln in block.splitlines() if len(key(ln)) >= 15]
        return sum(ln in tiny_lines for ln in lines) / max(1, len(lines))

    ov = [overlap(b) for b in blocks]
    candidates = [i for i, o in enumerate(ov) if o == 0.0]          # nothing in common with Tiny Shakespeare
    rng = random.Random(a.seed)
    test_idx = set(rng.sample(candidates, k=min(len(candidates), round(a.test_frac * len(blocks)))))
    train_blocks = [b for i, b in enumerate(blocks) if i not in test_idx]
    test_blocks = [blocks[i] for i in sorted(test_idx)]

    Path(a.out).write_text("\n\n".join(train_blocks) + "\n", encoding="utf-8")
    Path(a.test_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.test_out).write_text("\n\n".join(test_blocks) + "\n", encoding="utf-8")
    full_chars = sum(map(len, train_blocks))
    in_tiny = sum(len(b) for b, o in zip(blocks, ov) if o > 0.5)
    print(f"cleaned complete works: {len(text):,} chars, {len(set(text))} distinct (Tiny Shakespeare: {len(tiny):,} chars, {len(alphabet)} distinct)")
    print(f"blocks overlapping Tiny Shakespeare (>50% of lines): {in_tiny:,} chars ({100 * in_tiny / len(text):.0f}% of the works)")
    print(f"wrote {a.out}: {full_chars:,} chars ({full_chars / len(tiny):.1f}x Tiny Shakespeare)")
    print(f"wrote {a.test_out}: {sum(map(len, test_blocks)):,} chars in {len(test_blocks)} blocks with no Tiny Shakespeare lines")


if __name__ == "__main__":
    main()
