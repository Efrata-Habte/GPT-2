"""Experiment 7 (bonus): char-level vs. my BPE tokenizer.

Losses are per TOKEN, and BPE tokens cover more characters, so raw losses are not comparable. We compare
bits per character instead:   bpc = loss_per_token * (tokens / characters) / ln 2   (computed by train.py on the val split).

Confound to keep in mind: with the same block_size (in tokens) a BPE model sees more CHARACTERS of context and
processes more characters per step. Besides `bpe` (same block_size in tokens) we therefore also run `bpe-ctx-matched`
whose block_size is reduced so the context covers about as many characters as the char model's.

    python experiments/exp7_tokenizer.py --bpe_vocab 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (STANDARD_HEADERS, add_sweep_args, enter_repo_root, load_base, md_table, plot_overlay, run_variants,
                                standard_result_rows, write_report)

from src.data import data_dir_from_config, load_meta, prepare_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    add_sweep_args(ap)
    ap.add_argument("--bpe_vocab", type=int, default=512)
    a = ap.parse_args()
    enter_repo_root()
    base = load_base(a)

    # prepare the BPE dataset up front to learn its characters-per-token ratio (needed for context matching)
    bpe_data = {**base["data"], "tokenizer": "bpe", "bpe_vocab_size": a.bpe_vocab}
    ddir = data_dir_from_config(bpe_data)
    if not (ddir / "train.bin").exists():
        prepare_dataset(Path(base["data"]["path"]).read_text(encoding="utf-8"), ddir, "bpe", a.bpe_vocab)
    meta = load_meta(ddir)
    cpt = meta["train_chars"] / meta["train_tokens"]
    matched_block = max(8, round(base["model"]["block_size"] / cpt))

    common_bpe = {"data.tokenizer": "bpe", "data.bpe_vocab_size": a.bpe_vocab}
    variants = [
        {"name": "char", "overrides": {"data.tokenizer": "char"}},
        {"name": f"bpe{a.bpe_vocab}", "overrides": common_bpe},
        {"name": f"bpe{a.bpe_vocab}-ctx-matched", "overrides": {**common_bpe, "model.block_size": matched_block}},
    ]
    results = run_variants(base, variants, a.runs_dir or f"{a.out}/runs/exp7_tokenizer", a.force)
    out = Path(a.out)
    plot_overlay(results, out / "exp7_tokenizer.png", "Char-level vs BPE (curves are per-token losses; compare bits/char in the table)")
    write_report(out / "exp7_tokenizer.md", "Experiment 7: char-level vs BPE", [
        f"Base config `{a.base}`. BPE vocab {a.bpe_vocab} gives {cpt:.2f} characters per token on the training split, so "
        f"`bpe{a.bpe_vocab}-ctx-matched` uses block_size {matched_block} (about {matched_block * cpt:.0f} characters of context, "
        f"vs {base['model']['block_size']} for char).",
        "**Compare the `val bits/char` column**, not the loss columns (those are per token and not comparable across tokenizers).",
        md_table(STANDARD_HEADERS, standard_result_rows(results)),
        "![tok](exp7_tokenizer.png)",
    ])


if __name__ == "__main__":
    main()
