"""Experiment 12: small vs full training data (same architecture, same steps, same seed; only the data differs).

English  small = Tiny Shakespeare (1.0M training chars)   full = Complete Works (4.3M chars, data/prepare_full_shakespeare.py)
Amharic  small = MasakhaNEWS (3.4M chars)                 full = Wikipedia + news (9.9M chars, data/amharic/prepare_wikipedia.py)

Each model's own validation loss is measured on DIFFERENT text, so it cannot be compared. Here both models of a pair
are scored on the SAME held-out test text that neither saw in training (data/test_sets/), in bits per character.
Characters that one of the models has never seen are removed from the test text for both (reported), so the two
models are scored on identical strings.

    python experiments/exp12_data_scale.py      # uses checkpoints/{small,small_full,amharic,amharic_full} if trained, else the CPU demo models
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import COLORS, default_device, enter_repo_root, first_existing, md_table, plt, write_report
from experiments.exp1_temperature import distinct_n, valid_word_rate, words_of

import torch
import torch.nn.functional as F

from src.generate import encode_prompt, generate, load_model, tokenizer_for_checkpoint
from src.tokenizer import CharTokenizer


@torch.no_grad()
def test_bits_per_char(model, tok, text: str, device: str = "cpu", batch: int = 32) -> dict:
    """Bits per character of `text` under the model: non-overlapping windows of block_size tokens, every token after
    the first of each window is predicted, total nats / ln 2 / characters. Works for any tokenizer."""
    model.eval()
    ids = tok.encode(text)
    T = model.cfg.block_size
    windows = [ids[i:i + T + 1] for i in range(0, len(ids) - 1, T)]
    windows = [w for w in windows if len(w) >= 2]
    total_nats, n_pred = 0.0, 0
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        L = max(len(w) for w in chunk)
        x = torch.zeros(len(chunk), L - 1, dtype=torch.long)
        y = torch.full((len(chunk), L - 1), -100, dtype=torch.long)       # -100 = padding, ignored by cross_entropy
        for j, w in enumerate(chunk):
            x[j, :len(w) - 1] = torch.tensor(w[:-1])
            y[j, :len(w) - 1] = torch.tensor(w[1:])
        logits, _ = model(x.to(device))
        total_nats += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1), ignore_index=-100, reduction="sum").item()
        n_pred += int((y != -100).sum())
    return {"bits_per_char": total_nats / math.log(2) / len(text), "nats_per_token": total_nats / n_pred, "tokens": len(ids), "chars": len(text)}


def shared_text(text: str, toks: list) -> tuple[str, float]:
    """Drop characters that any char-level tokenizer in `toks` cannot encode, so all models see the same string."""
    allowed = None
    for t in toks:
        alpha = set(t.chars) if isinstance(t, CharTokenizer) else (set(t.alphabet) if getattr(t, "level", "") == "char" else None)
        if alpha is not None:
            allowed = alpha if allowed is None else allowed & alpha
    if allowed is None:
        return text, 0.0
    kept = "".join(c for c in text if c in allowed)
    return kept, 1 - len(kept) / max(1, len(text))


def train_chars(ckpt: Path) -> int | None:
    meta = ckpt.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8"))["train_chars"] if meta.exists() else None


def sample_quality(model, tok, prompt: str, dictionary: set[str], n: int = 4, tokens: int = 400, seed: int = 1337) -> dict:
    idx = encode_prompt(tok, prompt, "cpu").repeat(n, 1)
    out = generate(model, idx, tokens, temperature=0.8, generator=torch.Generator().manual_seed(seed))
    texts = [tok.decode(r[idx.size(1):]) for r in out.tolist()]
    ws = [words_of(t) for t in texts]
    return {"valid_words": sum(valid_word_rate(w, dictionary) for w in ws) / n, "distinct_2": sum(distinct_n(w, 2) for w in ws) / n,
            "sample": prompt + texts[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    # defaults: the T4 models if they exist, otherwise the CPU demo models
    ap.add_argument("--english_small", default=first_existing("checkpoints/small/summary.json", "checkpoints/demo_english/ckpt_best.pt").replace("summary.json", "ckpt_best.pt"))
    ap.add_argument("--english_full", default=first_existing("checkpoints/small_full/summary.json", "checkpoints/demo_english_full/ckpt_best.pt").replace("summary.json", "ckpt_best.pt"))
    ap.add_argument("--amharic_small", default=first_existing("checkpoints/amharic/summary.json", "checkpoints/demo_amharic/ckpt_best.pt").replace("summary.json", "ckpt_best.pt"))
    ap.add_argument("--amharic_full", default=first_existing("checkpoints/amharic_full/summary.json", "checkpoints/demo_amharic_full/ckpt_best.pt").replace("summary.json", "ckpt_best.pt"))
    ap.add_argument("--max_chars", type=int, default=250_000, help="per test set (speed)")
    ap.add_argument("--out", default="results/data_scale")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    enter_repo_root()
    dev = default_device()

    setups = [
        ("English", {"small data: Tiny Shakespeare": a.english_small, "full data: Complete Works": a.english_full},
         {"unseen Shakespeare": "data/test_sets/english_test.txt"}, "ROMEO:\n", ["data/shakespeare_full.txt", "data/input.txt"]),
        ("Amharic", {"small data: news": a.amharic_small, "full data: Wikipedia + news": a.amharic_full},
         {"news (held out)": "data/test_sets/amharic_news_test.txt", "Wikipedia (held out)": "data/test_sets/amharic_wiki_test.txt"},
         "የኢትዮጵያ ", ["data/amharic/amharic_full.txt"]),
    ]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report, results = [], {}
    for lang, models, tests, prompt, dict_files in setups:
        loaded = {}
        for name, path in models.items():
            p = Path(path)
            if p.exists():
                m, ck = load_model(p, dev)
                loaded[name] = (m, tokenizer_for_checkpoint(ck, p), p)
            else:
                print(f"[skip] {lang} / {name}: {p} not found")
        if len(loaded) < 2:
            report.append(f"## {lang}\n\nNeed both checkpoints ({', '.join(models.values())}); skipped.")
            continue
        dictionary = set()
        for f in dict_files:
            if Path(f).exists():
                dictionary |= set(words_of(Path(f).read_text(encoding="utf-8")))
        rows, lang_res = [], {"models": {}, "tests": list(tests), "prompt": prompt}
        for test_name, test_path in tests.items():
            raw = Path(test_path).read_text(encoding="utf-8")[: a.max_chars]
            text, dropped = shared_text(raw, [t for _, t, _ in loaded.values()])
            for name, (m, t, p) in loaded.items():
                r = test_bits_per_char(m, t, text, dev)
                lang_res["models"].setdefault(name, {"checkpoint": str(p), "train_chars": train_chars(p), "params": m.num_params(),
                                                      "vocab": t.vocab_size, "tests": {}})["tests"][test_name] = {**r, "dropped_share": dropped}
                print(f"{lang:8s} {name:32s} {test_name:22s} {r['bits_per_char']:.3f} bits/char")
        for name, (m, t, p) in loaded.items():
            lang_res["models"][name].update(sample_quality(m, t, prompt, dictionary))
        names = list(lang_res["models"])
        for name in names:
            d = lang_res["models"][name]
            rows.append([name, f"{d['train_chars']:,}" if d["train_chars"] else "?", f"{d['params']:,}"]
                        + [f"{d['tests'][tn]['bits_per_char']:.3f}" for tn in tests] + [f"{100 * d['valid_words']:.1f}%", f"{d['distinct_2']:.2f}"])
        first, last = lang_res["models"][names[0]], lang_res["models"][names[-1]]
        def change(tn):
            d = 100 * (1 - last["tests"][tn]["bits_per_char"] / first["tests"][tn]["bits_per_char"])
            return f"{tn}: {abs(d):.1f}% {'better' if d > 0 else 'worse'}"

        deltas = ", ".join(change(tn) for tn in tests)
        drops = ", ".join(f"{tn}: {100 * first['tests'][tn]['dropped_share']:.2f}%" for tn in tests)
        report.append(f"## {lang}\n\n" + md_table(["model", "training chars", "params"] + [f"test bits/char: {tn}" for tn in tests]
                                                    + ["valid words (T=0.8)", "distinct-2"], rows)
                      + f"\n\nFull-data model vs small-data model on the shared test text: {deltas}. "
                      f"Characters removed from the test text because one model never saw them: {drops}.\n\n"
                      + "\n\n".join(f"**{n}** (prompt `{prompt.strip()}`, T=0.8, seed 1337)\n```\n{lang_res['models'][n]['sample'][:500]}\n```" for n in names))
        results[lang] = lang_res

    # figure: grouped bars of test bits/char
    panels = [(lang, r) for lang, r in results.items()]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
        for ax, (lang, r) in zip(axes[0], panels):
            names, tests = list(r["models"]), r["tests"]
            w = 0.8 / len(names)
            for i, n in enumerate(names):
                vals = [r["models"][n]["tests"][t]["bits_per_char"] for t in tests]
                bars = ax.bar([j + (i - (len(names) - 1) / 2) * w for j in range(len(tests))], vals, w, color=COLORS[i], label=n)
                ax.bar_label(bars, fmt="%.3f", fontsize=8)
            ax.set_xticks(range(len(tests)))
            ax.set_xticklabels(tests)
            ax.set_ylabel("test bits per character (lower is better)")
            ax.set_title(lang)
            ax.set_ylim(0, max(r["models"][n]["tests"][t]["bits_per_char"] for n in names for t in tests) * 1.18)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("Small vs full training data: same model, same steps, scored on the same unseen text")
        fig.tight_layout()
        fig.savefig(out / "data_scale.png", dpi=150)

    (out / "data_scale.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(out / "data_scale.md", "Experiment 12: small vs full training data", [
        "Same architecture, same number of steps and the same seed; only the training data differs. Both models of a pair are "
        "scored on the same held-out text that neither saw during training (`data/test_sets/`).",
        *report, "![data scale](data_scale.png)",
        "Caveat: one seed per model; with a fixed number of steps the full-data model also sees each character fewer times "
        "(less memorisation), which is part of the effect being measured.",
    ])


if __name__ == "__main__":
    main()
