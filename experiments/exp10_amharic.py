"""Experiment 10 (bonus): tokenization effects of Ge'ez (Amharic) vs. English.

Both corpora are cut to the same number of characters (--max_chars) so that differences come from the
script/language, not from corpus size. BPE is trained on the first 90% and every statistic is measured on
the held-out last 10%.

What is measured, and why it is interesting for Ge'ez
  * character inventory   Ge'ez is a syllabary (one character = consonant+vowel), so the char-level vocab is
                          several times larger than English, which inflates the embedding table / output layer.
  * UTF-8 bytes per char  Ge'ez characters take 3 bytes. My BPE is BYTE-level (base vocab = 256 bytes), so a
                          single Amharic character starts as 3 tokens and the first merges are spent just
                          re-assembling characters.
  * fragment share        fraction of encoded tokens that are not even valid UTF-8 on their own (half a character).
  * own-token share       fraction of character occurrences whose character is one token.
  * chars/token, fertility (tokens per whitespace word): compression at each BPE vocab size.

    python data/amharic/prepare.py
    python experiments/exp10_amharic.py
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import COLORS, enter_repo_root, md_table, plt, write_report

from src.model import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.utils import load_config


def char_coverage(text: str, q: float) -> int:
    """Smallest number of distinct characters that account for a fraction q of all character occurrences."""
    counts = sorted(Counter(text).values(), reverse=True)
    total, run = sum(counts), 0
    for k, c in enumerate(counts, 1):
        run += c
        if run >= q * total:
            return k
    return len(counts)


def show_tokens(tok: BPETokenizer, ids: list[int]) -> str:
    """Readable token view: whole characters as text, partial UTF-8 fragments as <hex bytes>, '|' between tokens."""
    parts = []
    for i in ids:
        b = tok.vocab[i]
        try:
            parts.append(b.decode("utf-8").replace("\n", "\\n"))
        except UnicodeDecodeError:
            parts.append("<" + b.hex(" ").upper() + ">")
    return " | ".join(parts)


def fragment_stats(tok: BPETokenizer, ids: list[int], text: str) -> dict[str, float]:
    """fragment_share: share of tokens that are invalid UTF-8 on their own.
    own_token_share: share of character occurrences whose character exists as a single token."""
    frag = 0
    for i in ids:
        try:
            tok.vocab[i].decode("utf-8")
        except UnicodeDecodeError:
            frag += 1
    have = {b for b in tok.vocab.values()}
    counts = Counter(text)
    own = sum(n for c, n in counts.items() if c.encode("utf-8") in have)
    return {"fragment_share": frag / max(1, len(ids)), "own_token_share": own / max(1, len(text))}


def params_for_vocab(model_cfg: dict, vocab_size: int) -> tuple[int, float]:
    """(total params, embedding share) of the tied-weight GPT with this vocab size (V * n_embd is the only V-dependent term)."""
    p1 = GPT(GPTConfig(vocab_size=1, **model_cfg)).num_params()
    total = p1 + (vocab_size - 1) * model_cfg["n_embd"]
    return total, vocab_size * model_cfg["n_embd"] / total


def analyse(name: str, text: str, vocab_sizes: list[int], model_cfg: dict) -> dict:
    split = int(0.9 * len(text))
    train, val = text[:split], text[split:]
    words = max(1, len(val.split()))
    info = {
        "name": name, "chars": len(text), "bytes": len(text.encode("utf-8")), "unique": len(set(text)),
        "cov95": char_coverage(train, 0.95), "cov99": char_coverage(train, 0.99),
        "char_vocab": len(set(train)), "words": words, "val_chars": len(val),
    }
    info["bytes_per_char"] = info["bytes"] / info["chars"]
    info["char_fertility"] = len(val) / words
    t0 = time.time()
    big = BPETokenizer()
    big.train(train, max(vocab_sizes))
    info["bpe_train_s"] = time.time() - t0
    rows = []
    for V in vocab_sizes:
        tok = big.truncated(V)
        ids = tok.encode(val)
        assert tok.decode(ids) == val, "round trip failed"
        fs = fragment_stats(tok, ids, val)
        rows.append({"V": tok.vocab_size, "tokens": len(ids), "chars_per_token": len(val) / len(ids),
                     "bytes_per_token": len(val.encode("utf-8")) / len(ids), "fertility": len(ids) / words, **fs})
    info["bpe"] = rows
    sample = val[:36].replace("\n", " ")
    info["sample"] = sample
    info["sample_views"] = {V: show_tokens(big.truncated(V), big.truncated(V).encode(sample)) for V in (300, vocab_sizes[len(vocab_sizes) // 2], vocab_sizes[-1])}
    info["params_char"] = params_for_vocab(model_cfg, info["char_vocab"])
    return info


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 and cannot print Ge'ez
    ap = argparse.ArgumentParser()
    ap.add_argument("--amharic", default="data/amharic/amharic.txt")
    ap.add_argument("--english", default="data/input.txt")
    ap.add_argument("--max_chars", type=int, default=1_000_000)
    ap.add_argument("--vocab_sizes", type=int, nargs="+", default=[300, 500, 1000, 2000, 4000])
    ap.add_argument("--model_config", default="configs/small.yaml", help="for the parameter-count comparison")
    ap.add_argument("--out", default="results/amharic")
    a = ap.parse_args()
    enter_repo_root()
    mcfg = load_config(a.model_config)["model"]

    corpora = []
    for name, path in (("English (Shakespeare)", a.english), ("Amharic (Ge'ez)", a.amharic)):
        text = Path(path).read_text(encoding="utf-8")[: a.max_chars]
        print(f"{name}: {len(text):,} chars")
        corpora.append(analyse(name, text, a.vocab_sizes, mcfg))
        print(f"   BPE trained to {a.vocab_sizes[-1]} in {corpora[-1]['bpe_train_s']:.0f}s")
    en, am = corpora

    # ------------------ tables ------------------
    facts = md_table(["", en["name"], am["name"]], [
        ["characters (analysed)", f"{en['chars']:,}", f"{am['chars']:,}"],
        ["distinct characters", en["unique"], am["unique"]],
        ["UTF-8 bytes per character", f"{en['bytes_per_char']:.2f}", f"{am['bytes_per_char']:.2f}"],
        ["distinct chars to cover 95% / 99% of the text", f"{en['cov95']} / {en['cov99']}", f"{am['cov95']} / {am['cov99']}"],
        ["char-level vocab size (train split)", en["char_vocab"], am["char_vocab"]],
        ["characters per whitespace word", f"{en['char_fertility']:.2f}", f"{am['char_fertility']:.2f}"],
        [f"params of `{Path(a.model_config).stem}` model with char vocab", f"{en['params_char'][0]:,}", f"{am['params_char'][0]:,}"],
        ["... share of params in the (tied) embedding table", f"{100 * en['params_char'][1]:.1f}%", f"{100 * am['params_char'][1]:.1f}%"],
    ])
    hdr = ["BPE vocab", "tokens (val)", "chars/token", "bytes/token", "tokens/word", "fragment tokens", "chars with own token"]

    def bpe_rows(c):
        return [[r["V"], f"{r['tokens']:,}", f"{r['chars_per_token']:.2f}", f"{r['bytes_per_token']:.2f}", f"{r['fertility']:.2f}",
                 f"{100 * r['fragment_share']:.1f}%", f"{100 * r['own_token_share']:.1f}%"] for r in c["bpe"]]

    views = "\n\n".join(
        f"**{c['name']}** sample: `{c['sample']}`\n" + "\n".join(f"- vocab {V}: `{v}`" for V, v in c["sample_views"].items())
        for c in corpora)

    # ------------------ figure ------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for c, col, ls in ((en, COLORS[0], "--"), (am, COLORS[1], "-")):
        Vs = [r["V"] for r in c["bpe"]]
        axes[0][0].plot(Vs, [r["chars_per_token"] for r in c["bpe"]], "o" + ls, color=col, label=c["name"])
        axes[0][1].plot(Vs, [r["fertility"] for r in c["bpe"]], "o" + ls, color=col, label=c["name"])
        axes[0][1].plot([c["char_vocab"]], [c["char_fertility"]], "s", color=col, ms=9, mfc="none")
        axes[1][0].plot(Vs, [100 * r["fragment_share"] for r in c["bpe"]], "o" + ls, color=col, label=f"{c['name']}: fragment tokens")
        axes[1][0].plot(Vs, [100 * r["own_token_share"] for r in c["bpe"]], "^" + ls, color=col, alpha=0.5, label=f"{c['name']}: chars with own token")
    axes[0][0].set(xscale="log", xlabel="BPE vocab size", ylabel="characters per token", title="Compression")
    axes[0][1].set(xscale="log", yscale="log", xlabel="vocab size (squares = char-level)", ylabel="tokens per word", title="Fertility")
    axes[1][0].set(xscale="log", xlabel="BPE vocab size", ylabel="% of tokens / % of characters", title="Byte-level BPE: partial characters")
    for c, col in ((en, COLORS[0]), (am, COLORS[1])):
        freq = sorted(Counter(Path(a.amharic if c is am else a.english).read_text(encoding="utf-8")[: a.max_chars]).values(), reverse=True)
        tot = sum(freq)
        axes[1][1].loglog(range(1, len(freq) + 1), [f / tot for f in freq], color=col, label=c["name"])
    axes[1][1].set(xlabel="character rank", ylabel="relative frequency", title="Character frequency (Zipf-like)")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for ax in (axes[0][0], axes[1][0]):          # readable ticks on the log axes: just the vocab sizes we ran
        ax.set_xticks(a.vocab_sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes.flat:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle(f"Ge'ez (Amharic) vs. English tokenization ({a.max_chars:,} characters each)")
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp10_amharic.png", dpi=150)

    write_report(out / "exp10_amharic.md", "Experiment 10 (bonus): Amharic (Ge'ez) vs English tokenization", [
        f"Both corpora truncated to {a.max_chars:,} characters; BPE (byte-level, my implementation) trained on the first 90%, "
        "all statistics on the last 10%. Amharic source: MasakhaNEWS (see `data/amharic/prepare.py`).",
        "## Script and character inventory\n\n" + facts,
        f"## BPE on {en['name']}\n\n" + md_table(hdr, bpe_rows(en)),
        f"## BPE on {am['name']}\n\n" + md_table(hdr, bpe_rows(am)),
        "*fragment tokens* = share of tokens that are not valid UTF-8 on their own (a piece of a 3-byte character). "
        "*chars with own token* = share of character occurrences whose character is a single token.",
        "![amharic](exp10_amharic.png)",
        "## What the tokenizer actually produces\n\n`|` separates tokens; `<E1 88>` is a token that is only part of a character.\n\n" + views,
    ])


if __name__ == "__main__":
    main()
