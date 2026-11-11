"""Phase 9: Amharic (Ge'ez script) tokenization study. Everything is written to results/amharic/.

Part 1  Tokenizer analysis, English vs Amharic (and homophone-normalised Amharic) for three tokenizers
        char-level | char-level BPE | byte-level BPE   at several vocab sizes. Both corpora are cut to the same
        number of characters; BPE is trained on the first 90%, statistics are measured on the last 10%.
        -> vocabulary needed to cover the text, fertility (tokens per word), tokens per character,
           byte-level inflation, fragment tokens
Part 2  Normalisation effect on vocabulary (distinct characters / word types / unseen words).
Part 3  Examples of learned merges (which syllables / morphemes BPE discovers).
Part 4  Trained models (from checkpoints/): train/val loss in BITS PER CHARACTER, English vs Amharic.
Part 5  Samples from the Amharic model at several temperatures.

    python -I data/amharic/prepare.py
    python experiments/exp11_amharic_study.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import COLORS, enter_repo_root, first_existing, md_table, plt, read_log, series, write_report
from experiments.exp1_temperature import words_of

import torch

from src.data import normalize_text
from src.generate import encode_prompt, generate, load_model, tokenizer_for_checkpoint
from src.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer, token_labels

LANGS = ["English", "Amharic", "Amharic (normalized)"]


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------
def types_to_cover(counts: Counter, q: float) -> int:
    """Smallest number of distinct token types that account for a fraction q of all token occurrences."""
    total, run = sum(counts.values()), 0
    for k, c in enumerate(sorted(counts.values(), reverse=True), 1):
        run += c
        if run >= q * total:
            return k
    return len(counts)


def fragment_share(tok, ids: list[int]) -> float:
    """Share of tokens that are not valid UTF-8 on their own (only possible for byte-level BPE)."""
    if not hasattr(tok, "vocab") or getattr(tok, "level", "") != "byte":
        return 0.0
    bad = 0
    for i in ids:
        try:
            tok.vocab[i].decode("utf-8")
        except UnicodeDecodeError:
            bad += 1
    return bad / max(1, len(ids))


def evaluate_tokenizer(language: str, name: str, requested_v: int, tok, val: str) -> dict:
    ids = tok.encode(val)
    assert tok.decode(ids) == val, f"round trip failed for {language}/{name}"
    counts = Counter(ids)
    n_words = max(1, len(val.split()))
    return {
        "language": language, "tokenizer": name, "requested_vocab": requested_v, "vocab_size": tok.vocab_size,
        "tokens": len(ids), "tokens_per_char": len(ids) / len(val), "chars_per_token": len(val) / len(ids),
        "bytes_per_token": len(val.encode("utf-8")) / len(ids), "fertility": len(ids) / n_words,
        "types_for_95": types_to_cover(counts, 0.95), "types_for_99": types_to_cover(counts, 0.99),
        "used_share": len(counts) / tok.vocab_size, "fragment_share": fragment_share(tok, ids),
    }


def bpc_from_loss(loss_per_token: float, tokens: int, chars: int) -> float:
    """nats/token -> bits/character:  loss * (tokens / chars) / ln 2."""
    return loss_per_token * tokens / chars / math.log(2)


def load_saved_analysis(out: Path, tok_dir: Path, slug: dict):
    """Rebuild (rows, tokenizers, stats) from tokenizer_stats.csv, stats.json and the saved tokenizers."""
    int_cols = {"requested_vocab", "vocab_size", "tokens", "types_for_95", "types_for_99"}
    rows = []
    for r in csv.DictReader(open(out / "tokenizer_stats.csv", encoding="utf-8", newline="")):
        rows.append({k: (v if k in ("language", "tokenizer") else int(v) if k in int_cols else float(v)) for k, v in r.items()})
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    bigs = {lang: {name: load_tokenizer(tok_dir / f"{slug[lang]}_{name}.json") for name in ("char", "bpe_char", "bpe_byte")} for lang in LANGS}
    return rows, bigs, stats


# ---------------------------------------------------------------------------------------------
# Part 1-3
# ---------------------------------------------------------------------------------------------
def analyse_language(language: str, text: str, vocab_sizes: list[int]) -> tuple[list[dict], dict, dict]:
    split = int(0.9 * len(text))
    train, val = text[:split], text[split:]
    rows, bigs = [], {}
    char_tok = CharTokenizer.from_text(text)           # alphabet of the whole text, as in prepare_dataset
    bigs["char"] = char_tok
    rows.append(evaluate_tokenizer(language, "char", len(char_tok.chars), char_tok, val))
    for level, name in (("char", "bpe_char"), ("byte", "bpe_byte")):
        big = BPETokenizer(level=level)
        big.train(train, max(vocab_sizes), alphabet=set(text) if level == "char" else None)
        bigs[name] = big
        for v in vocab_sizes:
            if v < big.base_size:
                continue   # vocab smaller than the base alphabet is not a valid BPE vocabulary
            rows.append(evaluate_tokenizer(language, name, v, big.truncated(v), val))
    stats = {
        "chars": len(text), "bytes": len(text.encode("utf-8")), "unique_chars": len(set(text)),
        "bytes_per_char": len(text.encode("utf-8")) / len(text),
    }
    tw, vw = words_of(train), words_of(val)
    seen = set(tw)
    stats.update({"train_word_types": len(seen), "val_unseen_word_rate": sum(w not in seen for w in vw) / max(1, len(vw)),
                  "words_per_100_chars": 100 * len(vw) / len(val)})
    return rows, bigs, stats


def merge_examples(big: BPETokenizer, train: str, top_k: int = 40, first_k: int = 30) -> dict:
    """Early merges and the most frequent multi-character tokens of the encoded training text."""
    first = []
    for (a, b), new_id in list(big.merges.items())[:first_k]:
        first.append((token_labels(big, [a], visible=True)[0], token_labels(big, [b], visible=True)[0],
                      token_labels(big, [new_id], visible=True)[0]))
    counts = Counter(big.encode(train[:400_000]))
    freq = []
    for i, n in counts.most_common():
        s = token_labels(big, [i])[0]
        if len(s.replace("·", "")) >= 2 and not s.startswith("<"):
            freq.append((s, n))
        if len(freq) == top_k:
            break
    return {"first_merges": first, "frequent_tokens": freq}


# ---------------------------------------------------------------------------------------------
# Part 4-5
# ---------------------------------------------------------------------------------------------
def run_curves(run_dir: Path) -> dict | None:
    log, meta_p = run_dir / "log.csv", run_dir / "meta.json"
    if not log.exists() or not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    rows = read_log(log)
    f_tr = meta["train_tokens"] / meta["train_chars"] / math.log(2)
    f_va = meta["val_tokens"] / meta["val_chars"] / math.log(2)
    xs, tr = series(rows, "eval_train_loss")
    _, va = series(rows, "eval_val_loss")
    out = {"steps": xs, "train_bpc": [v * f_tr for v in tr], "val_bpc": [v * f_va for v in va], "meta": meta}
    summ = run_dir / "summary.json"
    out["summary"] = json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amharic", default="data/amharic/amharic.txt")
    ap.add_argument("--english", default="data/input.txt")
    ap.add_argument("--max_chars", type=int, default=1_000_000)
    ap.add_argument("--vocab_sizes", type=int, nargs="+", default=[500, 1000, 4000])
    ap.add_argument("--normalize", default="homophones")
    t4 = Path("checkpoints/small/summary.json").exists() and Path("checkpoints/amharic/summary.json").exists()
    ap.add_argument("--runs", nargs="*", default=(["English=checkpoints/small", "Amharic=checkpoints/amharic"] if t4 else
                                                 ["English (CPU demo)=checkpoints/demo_english", "Amharic (CPU demo)=checkpoints/demo_amharic"]),
                    help="label=run_dir of trained models for Part 4 (default: the T4 models if trained, else the CPU demo models)")
    ap.add_argument("--sample_run", default=first_existing("checkpoints/amharic/ckpt_best.pt", "checkpoints/demo_amharic/ckpt_best.pt").rsplit("/", 1)[0],
                    help="Amharic model used for Part 5 samples")
    ap.add_argument("--sample_prompt", default="የኢትዮጵያ ")   # የኢትዮጵያ
    ap.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.3])
    ap.add_argument("--sample_tokens", type=int, default=300)
    ap.add_argument("--out", default="results/amharic")
    ap.add_argument("--save_tokenizers", default="assets/tokenizers", help="also save the trained tokenizers here (used by the web demo)")
    ap.add_argument("--reuse_analysis", action="store_true", help="skip re-training the BPE tokenizers: reload the saved tables/tokenizers (use after training new models)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    enter_repo_root()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    V = a.vocab_sizes

    # ----------------------------- Part 1 + 2 + 3 -----------------------------
    en_text = Path(a.english).read_text(encoding="utf-8")[: a.max_chars]
    am_text = Path(a.amharic).read_text(encoding="utf-8")[: a.max_chars]
    corpora = {"English": en_text, "Amharic": am_text, "Amharic (normalized)": normalize_text(am_text, a.normalize)}
    all_rows, bigs, stats = [], {}, {}
    slug = {"English": "english", "Amharic": "amharic", "Amharic (normalized)": "amharic_norm"}
    if a.reuse_analysis:
        all_rows, bigs, stats = load_saved_analysis(out, Path(a.save_tokenizers), slug)
        print("reusing saved tokenizer analysis (tokenizer_stats.csv, stats.json, assets/tokenizers)")
    else:
        for lang, text in corpora.items():
            print(f"analysing {lang} ({len(text):,} chars) ...")
            rows, bigs[lang], stats[lang] = analyse_language(lang, text, V)
            all_rows += rows
        if a.save_tokenizers:
            tdir = Path(a.save_tokenizers)
            tdir.mkdir(parents=True, exist_ok=True)
            for lang in LANGS:
                for name, tok in bigs[lang].items():
                    tok.save(tdir / f"{slug[lang]}_{name}.json")
            print(f"saved tokenizers to {tdir}")
        with open(out / "tokenizer_stats.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)

    def pick(lang, tok, v=None):
        return next(r for r in all_rows if r["language"] == lang and r["tokenizer"] == tok and (v is None or r["requested_vocab"] == v))

    # ---- table 1: corpus facts
    facts = md_table(["", *LANGS], [
        ["characters analysed", *[f"{stats[l]['chars']:,}" for l in LANGS]],
        ["distinct characters (= char-level vocab)", *[stats[l]["unique_chars"] for l in LANGS]],
        ["UTF-8 bytes per character", *[f"{stats[l]['bytes_per_char']:.2f}" for l in LANGS]],
        ["words per 100 characters", *[f"{stats[l]['words_per_100_chars']:.1f}" for l in LANGS]],
        ["distinct word types (train)", *[f"{stats[l]['train_word_types']:,}" for l in LANGS]],
        ["share of val words unseen in train", *[f"{100 * stats[l]['val_unseen_word_rate']:.1f}%" for l in LANGS]],
    ])

    # ---- table 2: tokenizer comparison at each vocab size
    hdr = ["language", "tokenizer", "vocab", "tokens/word", "tokens/char", "bytes/token", "types for 95% / 99% of text", "fragment tokens"]

    def row(r):
        return [r["language"], r["tokenizer"], r["vocab_size"], f"{r['fertility']:.2f}", f"{r['tokens_per_char']:.3f}",
                f"{r['bytes_per_token']:.2f}", f"{r['types_for_95']} / {r['types_for_99']}", f"{100 * r['fragment_share']:.1f}%"]

    table = md_table(hdr, [row(r) for r in sorted(all_rows, key=lambda r: (LANGS.index(r["language"]), r["tokenizer"], r["vocab_size"]))])

    # ---- table 3: byte-level inflation
    infl_rows = []
    for lang in LANGS:
        base = stats[lang]["bytes_per_char"]
        cells = [lang, f"{base:.2f}x (no merges)"]
        for v in V:
            try:
                cells.append(f"{pick(lang, 'bpe_byte', v)['tokens'] / pick(lang, 'char')['tokens']:.2f}x / "
                             f"{pick(lang, 'bpe_byte', v)['tokens'] / pick(lang, 'bpe_char', v)['tokens']:.2f}x")
            except StopIteration:
                cells.append("-")
        infl_rows.append(cells)
    inflation = md_table(["language", "raw bytes vs chars", *[f"vocab {v}: byte-BPE tokens vs char-level / vs char-BPE" for v in V]], infl_rows)

    # ---- normalization effect
    nrm = {k: stats[k] for k in ("Amharic", "Amharic (normalized)")}
    norm_rows = [
        ["distinct characters", nrm["Amharic"]["unique_chars"], nrm["Amharic (normalized)"]["unique_chars"]],
        ["distinct word types (train)", f"{nrm['Amharic']['train_word_types']:,}", f"{nrm['Amharic (normalized)']['train_word_types']:,}"],
        ["val words unseen in train", f"{100 * nrm['Amharic']['val_unseen_word_rate']:.2f}%", f"{100 * nrm['Amharic (normalized)']['val_unseen_word_rate']:.2f}%"],
    ]
    for tk in ("bpe_char", "bpe_byte"):
        v = 1000 if 1000 in V else V[len(V) // 2]
        norm_rows.append([f"{tk} @ {v}: tokens per word", f"{pick('Amharic', tk, v)['fertility']:.3f}", f"{pick('Amharic (normalized)', tk, v)['fertility']:.3f}"])
    try:
        from data.amharic.normalize import homophone_counts
        hc = homophone_counts(am_text)
        share = 100 * sum(hc.values()) / len(am_text)
        norm_note = (f"{sum(hc.values()):,} of {len(am_text):,} characters ({share:.2f}%) are folded-away homophone letters; "
                     f"most common: " + ", ".join(f"{c} ({n:,})" for c, n in hc.most_common(6)) + ".")
    except Exception as e:  # pragma: no cover
        norm_note = f"(homophone counts unavailable: {e})"
    normalization = md_table(["", "raw", "normalized"], norm_rows)

    # ---- merges
    big_am = bigs["Amharic"]
    merge_sections = []
    for name, v in (("bpe_char", max(V)), ("bpe_byte", max(V))):
        ex = merge_examples(big_am[name].truncated(v), am_text[: int(0.9 * len(am_text))])
        first = ", ".join(f"{a}+{b}->{c}" for a, b, c in ex["first_merges"][:25])
        freq = ", ".join(f"{s} ({n:,})" for s, n in ex["frequent_tokens"][:30])
        merge_sections.append(f"### {name} (Amharic, vocab {v})\n\n**First merges** (`left+right->new`, `·` = space):\n\n{first}\n\n"
                              f"**Most frequent multi-character tokens in the training text** (count):\n\n{freq}")
    en_ex = merge_examples(bigs["English"]["bpe_char"].truncated(max(V)), en_text[: int(0.9 * len(en_text))])
    merge_sections.append("### bpe_char (English, vocab %d) for comparison\n\n**Most frequent multi-character tokens:**\n\n" % max(V)
                          + ", ".join(f"{s} ({n:,})" for s, n in en_ex["frequent_tokens"][:30]))

    # ----------------------------- figures -----------------------------
    tok_colors = {"char": COLORS[2], "bpe_char": COLORS[0], "bpe_byte": COLORS[1]}
    lang_style = {"English": "--", "Amharic": "-", "Amharic (normalized)": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for lang in LANGS:
        for tk in ("bpe_char", "bpe_byte"):
            rs = sorted((r for r in all_rows if r["language"] == lang and r["tokenizer"] == tk), key=lambda r: r["vocab_size"])
            axes[0].plot([r["vocab_size"] for r in rs], [r["fertility"] for r in rs], "o" + lang_style[lang], color=tok_colors[tk], label=f"{lang}, {tk}")
            axes[1].plot([r["vocab_size"] for r in rs], [r["tokens_per_char"] for r in rs], "o" + lang_style[lang], color=tok_colors[tk])
        c = pick(lang, "char")
        axes[0].plot([c["vocab_size"]], [c["fertility"]], "s", color=tok_colors["char"], ms=8, mfc="none" if lang != "English" else tok_colors["char"])
        axes[1].plot([c["vocab_size"]], [c["tokens_per_char"]], "s", color=tok_colors["char"], ms=8, mfc="none" if lang != "English" else tok_colors["char"])
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ticks = sorted({r["vocab_size"] for r in all_rows if r["tokenizer"] != "char"} | {pick("English", "char")["vocab_size"], pick("Amharic", "char")["vocab_size"]})
    for ax, yl in zip(axes, ("tokens per word (fertility)", "tokens per character")):
        ax.set(xscale="log", xlabel="vocabulary size (squares = char-level: filled English, hollow Amharic)", ylabel=yl)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="s", ls="", color=tok_colors["char"], label="char-level"))
    axes[0].legend(handles=handles, fontsize=6)
    fig.suptitle("Fertility vs. vocabulary size: English vs Amharic, three tokenizers")
    fig.tight_layout()
    fig.savefig(out / "fertility_vs_vocab.png", dpi=150)
    plt.close(fig)

    v0 = 1000 if 1000 in V else V[len(V) // 2]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    for i, lang in enumerate(LANGS):
        vals = [pick(lang, tk, None if tk == "char" else v0)["fertility"] for tk in ("char", "bpe_char", "bpe_byte")]
        ax.bar([j + (i - 1) * width for j in range(3)], vals, width, color=COLORS[i], label=lang)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["char-level", f"char-BPE ({v0})", f"byte-BPE ({v0})"])
    ax.set_ylabel("tokens per word")
    ax.set_title("Tokens per word, English vs Amharic")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tokens_per_word.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for lang in ("English", "Amharic"):
        xs = [min(V) // 2] + V
        ys = [stats[lang]["bytes_per_char"]]
        for v in V:
            try:
                ys.append(pick(lang, "bpe_byte", v)["tokens"] / pick(lang, "char")["tokens"])
            except StopIteration:
                ys.append(float("nan"))
        ax.plot([256] + V, ys, "o-", color=COLORS[0] if lang == "English" else COLORS[1], label=lang)
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("byte-BPE vocabulary size (256 = no merges)")
    ax.set_ylabel("tokens relative to char-level (x)")
    ax.set_title("Sequence-length inflation of byte-level BPE")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "byte_inflation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (title, vals) in zip(axes, [
        ("distinct characters", [nrm[k]["unique_chars"] for k in nrm]),
        ("word types (train)", [nrm[k]["train_word_types"] for k in nrm]),
        ("val words unseen in train (%)", [100 * nrm[k]["val_unseen_word_rate"] for k in nrm]),
    ]):
        ax.bar(["raw", "normalized"], vals, color=[COLORS[1], COLORS[2]])
        ax.set_title(title, fontsize=10)
        for j, vv in enumerate(vals):
            ax.text(j, vv, f"{vv:,.1f}" if isinstance(vv, float) else f"{vv:,}", ha="center", va="bottom", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Effect of homophone normalization on the Amharic vocabulary")
    fig.tight_layout()
    fig.savefig(out / "normalization.png", dpi=150)
    plt.close(fig)

    sections = [
        f"Corpora cut to {a.max_chars:,} characters each; BPE trained on the first 90%, all statistics measured on the last 10%. "
        f"Amharic = MasakhaNEWS; normalization mode `{a.normalize}`.",
        "## Corpus facts\n\n" + facts,
        "## Tokenizer comparison\n\n" + table + "\n\n*fragment tokens*: share of tokens that are only part of a UTF-8 character (byte-level BPE). "
        "*types for 95% / 99%*: how many distinct tokens cover that share of all token occurrences.",
        "![fertility](fertility_vs_vocab.png)\n\n![per word](tokens_per_word.png)",
        "## Byte-level sequence-length inflation\n\nTokens of byte-BPE relative to char-level and to char-BPE of the same vocab size.\n\n" + inflation + "\n\n![inflation](byte_inflation.png)",
        "## Effect of homophone normalization\n\n" + norm_note + "\n\n" + normalization + "\n\n![normalization](normalization.png)",
        "## Learned merges\n\n" + "\n\n".join(merge_sections),
    ]

    # ----------------------------- Part 4: trained models -----------------------------
    runs = {}
    for item in a.runs:
        label, _, d = item.partition("=")
        c = run_curves(Path(d))
        if c:
            runs[label] = c
    if runs:
        rows = []
        for label, c in runs.items():
            s = c["summary"]
            lang = "English" if label.startswith("English") else "Amharic"
            bpc = c["val_bpc"][-1]
            bits_per_byte = bpc / stats[lang]["bytes_per_char"]                     # bytes of UTF-8 text
            bits_per_word = bpc * 100 / stats[lang]["words_per_100_chars"]           # letter-words, see words_of()
            rows.append([label, f"{s['params']:,}" if s else "-", f"{c['meta']['vocab_size']}", f"{c['train_bpc'][-1]:.3f}", f"{bpc:.3f}",
                         f"{bits_per_byte:.3f}", f"{bits_per_word:.1f}", f"{c['steps'][-1]}"])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, (label, c) in enumerate(runs.items()):
            ax.plot(c["steps"], c["val_bpc"], "-", color=COLORS[i], lw=2, label=f"{label} val")
            ax.plot(c["steps"], c["train_bpc"], "--", color=COLORS[i], lw=1, alpha=0.7, label=f"{label} train")
        ax.set_xlabel("step")
        ax.set_ylabel("bits per character")
        ax.set_title("Training curves in bits per character (lower is better)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "bpc_comparison.png", dpi=150)
        plt.close(fig)
        with open(out / "bpc_curves.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "step", "train_bpc", "val_bpc"])
            for label, c in runs.items():
                for st, tr, va in zip(c["steps"], c["train_bpc"], c["val_bpc"]):
                    w.writerow([label, st, f"{tr:.5f}", f"{va:.5f}"])
        sections += ["## Trained models: bits per character\n\nbits/char = loss per token x tokens/chars / ln 2, so models with different "
                     "tokenizers or scripts are comparable. Same architecture, steps and seed for every model.",
                     md_table(["model", "params", "vocab", "train bits/char", "val bits/char", "val bits/byte", "val bits/word", "steps"], rows),
                     "**Reading the three units.** Bits per *character* is what the assignment asks for, but characters are not equally informative: "
                     "a Ge'ez letter is a consonant+vowel syllable (2.58 UTF-8 bytes on average here, against 1.00 for English), so it naturally costs more "
                     "bits than a Latin letter. Bits per *byte* normalises for the encoding and bits per *word* for the amount of text; "
                     "use the unit that matches the claim you want to make.",
                     "![bpc](bpc_comparison.png)"]
    else:
        sections.append("## Trained models\n\nNo trained runs found; train with `configs/demo_amharic.yaml` (or `configs/amharic.yaml` on a GPU).")

    # ----------------------------- Part 5: samples -----------------------------
    sample_dir = Path(a.sample_run)
    ckpt = sample_dir / "ckpt_best.pt"
    samples = {}
    if ckpt.exists():
        model, ck = load_model(ckpt, "cpu")
        tok = tokenizer_for_checkpoint(ck, ckpt)
        idx = encode_prompt(tok, a.sample_prompt, "cpu")
        settings = [("greedy", dict(greedy=True))] + [(f"T={t}", dict(temperature=t)) for t in a.temps]
        for name, kw in settings:
            g = torch.Generator().manual_seed(1337)
            o = generate(model, idx, a.sample_tokens, generator=g, **kw)
            samples[name] = tok.decode(o[0].tolist())
        (out / "samples.json").write_text(json.dumps({"model": str(sample_dir), "prompt": a.sample_prompt, "samples": samples}, ensure_ascii=False, indent=1), encoding="utf-8")
        sections.append("## Samples from the Amharic model\n\nModel `%s`, prompt `%s`, seed 1337.\n\n" % (sample_dir, a.sample_prompt)
                        + "\n\n".join(f"**{n}**\n```\n{t}\n```" for n, t in samples.items()))
    else:
        sections.append(f"## Samples\n\n`{ckpt}` not found; train the Amharic model first.")

    write_report(out / "tokenizer_analysis.md", "Phase 9: Amharic (Ge'ez) tokenization study", sections)
    (out / "stats.json").write_text(json.dumps({k: v for k, v in stats.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
