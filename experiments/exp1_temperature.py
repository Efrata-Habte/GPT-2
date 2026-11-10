"""Experiment 1: effect of sampling temperature (and greedy decoding) on diversity and repetition.

Same prompt and same random seed for every setting, N samples each. Metrics are computed on the
generated WORDS (text level, so they don't depend on the tokenizer):
    distinct-n    unique word n-grams / total word n-grams (higher = more diverse), n = 1, 2
    repetition-4  share of word 4-grams that already occurred earlier in the same sample
    valid words   share of generated words that appear in the training corpus (char-level models
                  produce non-words when the distribution is too flat)

    python experiments/exp1_temperature.py --ckpt checkpoints/small/ckpt_best.pt --prompt "ROMEO:"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import COLORS, default_device, enter_repo_root, load_bundle, md_table, plt, write_report

import torch

from src.generate import encode_prompt, generate

WORD = re.compile(r"[A-Za-z']+")


def words_of(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def ngrams(ws: list[str], n: int) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def distinct_n(ws: list[str], n: int) -> float:
    g = ngrams(ws, n)
    return len(set(g)) / len(g) if g else 1.0


def repetition_rate(ws: list[str], n: int = 4) -> float:
    """Fraction of n-gram occurrences that repeat an earlier n-gram of the same text."""
    seen, rep, g = set(), 0, ngrams(ws, n)
    for x in g:
        rep += x in seen
        seen.add(x)
    return rep / len(g) if g else 0.0


def valid_word_rate(ws: list[str], vocab: set[str]) -> float:
    return sum(w in vocab for w in ws) / len(ws) if ws else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/small/ckpt_best.pt")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.2, 0.5, 0.8, 1.0, 1.5])
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--corpus", default="data/input.txt", help="text used to define 'valid words'")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    enter_repo_root()

    dev = default_device()
    model, ck, tok, _ = load_bundle(a.ckpt, dev)
    vocab = set(words_of(Path(a.corpus).read_text(encoding="utf-8")))
    prompt = encode_prompt(tok, a.prompt, dev).repeat(a.n_samples, 1)
    p_len = prompt.size(1)

    settings = [("greedy", dict(greedy=True))] + [(f"T={t}", dict(temperature=t)) for t in a.temps]
    rows, samples, stats = [], [], {}
    for name, kw in settings:
        g = torch.Generator().manual_seed(a.seed)   # identical randomness for every setting
        out = generate(model, prompt, a.max_new_tokens, generator=g, **kw)
        texts = [tok.decode(r[p_len:]) for r in out.tolist()]
        per = [words_of(t) for t in texts]
        m = {
            "distinct-1": sum(distinct_n(w, 1) for w in per) / len(per),
            "distinct-2": sum(distinct_n(w, 2) for w in per) / len(per),
            "repetition-4": sum(repetition_rate(w, 4) for w in per) / len(per),
            "valid words": sum(valid_word_rate(w, vocab) for w in per) / len(per),
        }
        stats[name] = m
        rows.append([name] + [f"{m[k]:.3f}" for k in m])
        samples.append((name, texts[0]))
        print(name, {k: round(v, 3) for k, v in m.items()})

    # ---- figure ----
    temps = [t for t in a.temps]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    panels = [("diversity (distinct-n)", ["distinct-1", "distinct-2"]), ("repetition-4 (lower = less looping)", ["repetition-4"]),
              ("valid-word rate", ["valid words"])]
    for ax, (title, keys) in zip(axes, panels):
        for i, k in enumerate(keys):
            c = COLORS[i]
            ax.plot(temps, [stats[f"T={t}"][k] for t in temps], "o-", color=c, label=k)
            ax.axhline(stats["greedy"][k], color=c, ls="--", lw=1, alpha=0.7, label=f"{k} (greedy)")
        ax.set_xlabel("temperature")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Sampling temperature vs. text quality (prompt {a.prompt!r}, {a.n_samples} samples x {a.max_new_tokens} tokens)")
    fig.tight_layout()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp1_temperature.png", dpi=150)

    sections = [
        f"Checkpoint `{a.ckpt}`, prompt `{a.prompt!r}`, {a.n_samples} samples x {a.max_new_tokens} new tokens per setting, "
        f"seed {a.seed} (same seed for every setting). Metrics are averaged over samples. Greedy gives one deterministic "
        "text, so all its samples are identical.",
        md_table(["setting", "distinct-1", "distinct-2", "repetition-4", "valid words"], rows),
        "![temperature](exp1_temperature.png)",
        "## Sample 1 for each setting (first 400 characters)\n\n" +
        "\n\n".join(f"**{n}**\n```\n{a.prompt}{t[:400]}\n```" for n, t in samples),
    ]
    write_report(out_dir / "exp1_temperature.md", "Experiment 1: temperature", sections)


if __name__ == "__main__":
    main()
