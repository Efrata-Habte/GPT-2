"""Phase 1 experiment: vocab size vs. sequence length: char-level vs. my BPE (over characters "cbpe" and over bytes "bbpe").

BPE is trained on the first 90% of the text only and all statistics are measured on the
held-out last 10%, so the numbers reflect generalisation, not memorisation.

Usage:  python experiments/compare_tokenizers.py --data data/input.txt
Output: results/tokenizer_comparison.md and results/tokenizer_comparison.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tokenizer import BPETokenizer, CharTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--vocab_sizes", type=int, nargs="+", default=[300, 500, 1000, 2000])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    split = int(0.9 * len(text))
    train_text, val_text = text[:split], text[split:]

    rows = []
    char_tok = CharTokenizer.from_text(train_text)
    n_char = len(val_text)
    rows.append(("char", char_tok.vocab_size, n_char, 1.0))

    for level, label in (("char", "cbpe"), ("byte", "bbpe")):
        t0 = time.time()
        big = BPETokenizer(level=level)
        big.train(train_text, max(args.vocab_sizes), alphabet=set(text) if level == "char" else None)
        print(f"trained {level}-level BPE up to vocab {big.vocab_size} in {time.time() - t0:.1f}s")
        for v in args.vocab_sizes:
            if v < big.base_size:
                continue
            tok = big.truncated(v)
            ids = tok.encode(val_text)
            assert tok.decode(ids) == val_text  # round trip on held-out text
            rows.append((f"{label}-{v}", tok.vocab_size, len(ids), n_char / len(ids)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "| tokenizer | vocab size | tokens in val text | chars / token | seq-length vs char |",
        "|---|---|---|---|---|",
    ]
    for name, v, n, cpt in rows:
        lines.append(f"| {name} | {v} | {n:,} | {cpt:.2f} | {n / n_char:.2f}x |")
    (out / "tokenizer_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix, color, name in (("cbpe", "#0072B2", "char-level BPE"), ("bbpe", "#D55E00", "byte-level BPE")):
        pts = [r for r in rows if r[0].startswith(prefix)]
        ax.plot([r[1] for r in pts], [r[3] for r in pts], "o-", color=color, label=name)
    ax.plot([rows[0][1]], [rows[0][3]], "s", color="#009E73", ms=8, label="char-level (no merges)")
    ax.set_xscale("log")
    ax.set_xlabel("vocab size")
    ax.set_ylabel("avg characters per token (higher = shorter sequences)")
    ax.set_title("Vocab size vs. compression (held-out text)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "tokenizer_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
