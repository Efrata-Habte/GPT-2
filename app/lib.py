"""Logic behind the Streamlit demo (no Streamlit imports here, so it is unit-testable).

Uses ONLY this repository's own code: src.model (GPT), src.tokenizer, src.generate. Nothing from
Hugging Face / pretrained models.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate import generate, generate_stream, last_token_logits, load_model, next_token_probs, tokenizer_for_checkpoint  # noqa: E402
from src.model import GPT  # noqa: E402
from src.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer, token_labels  # noqa: E402

MODELS = {"English": "english", "Amharic": "amharic"}      # sidebar label -> checkpoint key
DATA_VARIANTS = {"Small dataset": "", "Full dataset": "_full"}   # appended to the checkpoint key: english / english_full
DATA_DESCRIPTIONS = {
    "english": "Tiny Shakespeare (excerpts from the plays)",
    "english_full": "Complete Works of Shakespeare",
    "amharic": "MasakhaNEWS Amharic news",
    "amharic_full": "Amharic Wikipedia + MasakhaNEWS",
}
TOKENIZER_NAMES = ["char", "bpe_char", "bpe_byte"]
TOKENIZER_TITLES = {"char": "Character-level", "bpe_char": "Character-level BPE", "bpe_byte": "Byte-level BPE"}
ETHIOPIC_CSS_FONTS = "'Noto Sans Ethiopic','Nyala','Ebrima','Abyssinica SIL'"


# ----------------------------------------------------------------------------------------------
# Locating and loading checkpoints
# ----------------------------------------------------------------------------------------------
def find_checkpoint(key: str) -> Path | None:
    """Where the model called `key` (english, english_full, amharic, amharic_full) is loaded from.

    If GPT_DEMO_CKPT_DIR is set, ONLY <that dir>/<key>/model.pt is used (deployments, tests). Otherwise:
    app/checkpoints/<key>/model.pt (exported, weights only), then the training output checkpoints/demo_<key>|<key>.
    """
    override = os.environ.get("GPT_DEMO_CKPT_DIR")
    if override:
        p = Path(override) / key / "model.pt"
        return p if p.exists() else None
    if (ROOT / "app" / "checkpoints" / key / "model.pt").exists():
        return ROOT / "app" / "checkpoints" / key / "model.pt"
    for name in (f"demo_{key}", key):
        p = ROOT / "checkpoints" / name / "ckpt_best.pt"
        if p.exists():
            return p
    return None


def load_demo_model(path: str | Path):
    """-> (model in eval mode, tokenizer, info dict). Works for full training checkpoints and exported ones."""
    model, ck = load_model(path, "cpu")
    tok = tokenizer_for_checkpoint(ck, path)
    info = {
        "path": str(path), "model_config": ck["model_config"], "params": model.num_params(), "step": ck.get("step"),
        "best_val_loss": ck.get("best_val_loss"), "tokenizer_type": getattr(tok, "level", "char") if not isinstance(tok, CharTokenizer) else "char",
        "vocab_size": tok.vocab_size, "meta": ck.get("meta") or {}, "data": (ck.get("config") or {}).get("data") or ck.get("data") or {},
    }
    data_meta = info["meta"].get("meta")
    if data_meta is None and (Path(path).parent / "meta.json").exists():       # un-exported training checkpoint
        data_meta = json.loads((Path(path).parent / "meta.json").read_text(encoding="utf-8"))
    info["train_chars"] = (data_meta or {}).get("train_chars")
    return model, tok, info


def export_checkpoint(src_ckpt: str | Path, out_dir: str | Path, meta: dict | None = None) -> Path:
    """Slim a training checkpoint for deployment: weights + architecture + tokenizer, no optimizer state."""
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    src_dir = Path(src_ckpt).parent
    meta = dict(meta or {})
    for name in ("meta.json", "summary.json"):           # training-data size and final metrics travel with the model
        if (src_dir / name).exists():
            meta[name.split(".")[0]] = json.loads((src_dir / name).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": ck["model"], "model_config": ck["model_config"], "tokenizer": ck["tokenizer"], "step": ck["step"],
        "best_val_loss": ck["best_val_loss"], "data": ck["config"]["data"], "meta": meta,
    }
    torch.save(payload, out / "model.pt")
    return out / "model.pt"


# ----------------------------------------------------------------------------------------------
# Generation helpers
# ----------------------------------------------------------------------------------------------
class IncrementalDecoder:
    """Turn a growing token-id list into text deltas without ever showing half a UTF-8 character.

    Byte-level BPE can emit tokens that are only part of a character (e.g. <E1 88> of a Ge'ez letter);
    decoding them alone gives the replacement char U+FFFD. We decode the whole sequence and hold back a
    trailing U+FFFD until the character is complete.
    """

    def __init__(self, tok) -> None:
        self.tok, self.ids, self.emitted = tok, [], 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        stable = text.rstrip("�") if text.endswith("�") else text
        delta, self.emitted = stable[self.emitted:], max(self.emitted, len(stable))
        return delta

    def flush(self) -> str:
        text = self.tok.decode(self.ids)
        delta, self.emitted = text[self.emitted:], len(text)
        return delta


def decoding_kwargs(method: str, temperature: float, top_k: int, top_p: float) -> dict:
    """Map the UI's decoding choice to generate()/sample_next() arguments."""
    if method == "greedy":
        return {"greedy": True}
    if method == "temperature":
        return {"temperature": temperature}
    if method == "top-k":
        return {"temperature": temperature, "top_k": int(top_k)}
    if method == "top-p":
        return {"temperature": temperature, "top_p": float(top_p)}
    raise ValueError(method)


def encode_prompt_safe(tok, prompt: str) -> tuple[list[int], list[str]]:
    """Encode, dropping characters the tokenizer has never seen. Returns (ids, dropped characters)."""
    allowed = _alphabet(tok)
    dropped = sorted({c for c in prompt if allowed is not None and c not in allowed})
    clean = "".join(c for c in prompt if allowed is None or c in allowed)
    ids = tok.encode(clean) if clean else [0]       # empty prompt: start from token 0
    return ids, dropped


def _alphabet(tok) -> set[str] | None:
    if isinstance(tok, CharTokenizer):
        return set(tok.chars)
    if getattr(tok, "level", "") == "char":
        return set(tok.alphabet or [])
    return None  # byte-level: everything is encodable


def stream_text(model: GPT, tok, prompt: str, max_new_tokens: int, method: str = "temperature", temperature: float = 0.8,
                top_k: int = 40, top_p: float = 0.9, seed: int = 1337, use_cache: bool = True):
    """Yield text pieces as tokens are generated (suitable for st.write_stream). The prompt is not repeated."""
    ids, _ = encode_prompt_safe(tok, prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed))
    dec = IncrementalDecoder(tok)
    for nid in generate_stream(model, idx, max_new_tokens, generator=gen, use_cache=use_cache,
                               **decoding_kwargs(method, temperature, top_k, top_p)):
        piece = dec.push(int(nid.item()))
        if piece:
            yield piece
    tail = dec.flush()
    if tail:
        yield tail


def generate_text(model: GPT, tok, prompt: str, max_new_tokens: int, seed: int = 1337, **decode) -> str:
    """Non-streaming generation: the continuation only (for side-by-side comparison)."""
    ids, _ = encode_prompt_safe(tok, prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    out = generate(model, idx, max_new_tokens, generator=torch.Generator().manual_seed(int(seed)), **decode)
    return tok.decode(out[0, len(ids):].tolist())


def top_next_tokens(model: GPT, tok, ids: list[int], k: int = 10, temperature: float = 1.0, top_k: int | None = None,
                    top_p: float | None = None) -> list[dict]:
    """The k most probable next tokens: [{id, label, text, prob}], using the same filtering as the sampler."""
    logits = last_token_logits(model, torch.tensor([ids], dtype=torch.long))          # (V,)
    probs = next_token_probs(logits[None], temperature, top_k, top_p)[0]              # (V,)
    p, i = probs.topk(min(k, probs.numel()))
    labels = token_labels(tok, i.tolist())
    return [{"id": int(t), "label": lab, "text": tok.decode([int(t)]), "prob": float(pr)} for t, lab, pr in zip(i, labels, p)]


@torch.no_grad()
def attention_for_text(model: GPT, tok, text: str) -> tuple[list[str], torch.Tensor, list[str]]:
    """(token labels, attention (n_layer, n_head, T, T), dropped chars). The text is cropped to block_size tokens."""
    ids, dropped = encode_prompt_safe(tok, text)
    ids = ids[: model.cfg.block_size]
    model.eval()
    _, _, attns = model(torch.tensor([ids], dtype=torch.long), return_attn=True)
    return token_labels(tok, ids), torch.stack([a[0] for a in attns]).cpu(), dropped


def distinct_n(text: str, n: int) -> float:
    """Unique character n-grams / total character n-grams (works for any script; used for quick comparisons)."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def param_breakdown_rows(model: GPT) -> list[tuple[str, int]]:
    """Component totals (blocks aggregated) for the About tab."""
    bd = model.param_breakdown()
    rows = [(k, v) for k, v in bd.items() if not k.startswith("block ")]
    for part in ("attention", "mlp", "layernorms"):
        rows.append((f"all {len(model.blocks)} blocks: {part}", sum(v for k, v in bd.items() if k.startswith("block ") and k.endswith(part))))
    rows.append(("TOTAL", sum(bd.values())))
    return rows


# ----------------------------------------------------------------------------------------------
# Tokenizer playground
# ----------------------------------------------------------------------------------------------
TOKEN_COLORS = ["rgba(0,114,178,0.28)", "rgba(213,94,0,0.28)", "rgba(0,158,115,0.28)", "rgba(204,121,167,0.30)", "rgba(230,159,0,0.30)"]


def load_playground_tokenizers(set_name: str, vocab_size: int, tok_dir: str | Path | None = None) -> dict:
    """{"char": tok, "bpe_char": tok, "bpe_byte": tok} for a tokenizer set ("english" | "amharic" | "amharic_norm").

    Only the largest BPE vocabularies are stored; smaller ones are exact prefixes of the merge lists (truncated()).
    BPE tokenizers whose base alphabet is larger than `vocab_size` are skipped.
    """
    d = Path(tok_dir) if tok_dir else ROOT / "assets" / "tokenizers"
    out = {}
    for name in TOKENIZER_NAMES:
        p = d / f"{set_name}_{name}.json"
        if not p.exists():
            continue
        tok = load_tokenizer(p)
        if name == "char":
            out[name] = tok
        elif vocab_size >= tok.base_size:
            out[name] = tok.truncated(vocab_size)
    return out


def tokenize_for_display(tok, text: str) -> dict:
    """ids, per-token strings and summary statistics. Characters outside a char-level alphabet are skipped (reported)."""
    ids, dropped = encode_prompt_safe(tok, text) if text else ([], [])
    if not text:
        ids = []
    shown = "".join(c for c in text if c not in dropped)
    labels = token_labels(tok, ids, visible=False)
    n_words = max(1, len(shown.split()))
    frag = 0
    if hasattr(tok, "vocab") and getattr(tok, "level", "") == "byte":
        for i in ids:
            try:
                tok.vocab[i].decode("utf-8")
            except UnicodeDecodeError:
                frag += 1
    return {
        "ids": ids, "labels": labels, "dropped": dropped, "n_tokens": len(ids), "n_chars": len(shown),
        "n_bytes": len(shown.encode("utf-8")), "chars_per_token": len(shown) / len(ids) if ids else 0.0,
        "bytes_per_token": len(shown.encode("utf-8")) / len(ids) if ids else 0.0, "tokens_per_word": len(ids) / n_words if ids else 0.0,
        "fragments": frag, "vocab_size": tok.vocab_size,
    }


def tokens_to_html(labels: list[str], font_stack: str = ETHIOPIC_CSS_FONTS) -> str:
    """Colored token boundaries: each token is a span with its own background; whitespace is made visible."""
    spans = []
    for i, lab in enumerate(labels):
        shown = html.escape(lab).replace("\n", "↵").replace(" ", "&nbsp;") or "&nbsp;"
        spans.append(f'<span style="background:{TOKEN_COLORS[i % len(TOKEN_COLORS)]};padding:2px 1px;border-radius:3px;'
                     f'margin-right:1px;white-space:pre-wrap;">{shown}</span>')
    return f'<div style="font-family:{font_stack},sans-serif;font-size:1.15rem;line-height:2.1;word-break:break-all;">{"".join(spans)}</div>'


def normalization_diff_html(raw: str, norm: str, font_stack: str = ETHIOPIC_CSS_FONTS) -> str:
    """The text with every character changed by normalisation highlighted (equal-length strings)."""
    parts = []
    for a, b in zip(raw, norm):
        if a == b:
            parts.append(html.escape(a))
        else:
            parts.append(f'<span style="background:rgba(213,94,0,0.35);border-radius:3px;" title="{html.escape(a)} → {html.escape(b)}">{html.escape(b)}</span>')
    return f'<div style="font-family:{font_stack},sans-serif;font-size:1.15rem;line-height:2;white-space:pre-wrap;">{"".join(parts)}</div>'


def apply_normalization(text: str, mode: str = "homophones") -> tuple[str, int]:
    """(normalised text, number of characters changed). Uses data/amharic/normalize.py."""
    from data.amharic.normalize import changed_positions, normalize_amharic
    out = normalize_amharic(text, mode)
    return out, len(changed_positions(text, out)) if len(out) == len(text) else sum(a != b for a, b in zip(text, out))


# ----------------------------------------------------------------------------------------------
# Results / study data
# ----------------------------------------------------------------------------------------------
def read_csv_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_markdown_images(md: str, base_dir: str | Path) -> list[tuple[str, str]]:
    """Split markdown into ("md", text) and ("img", path) parts so images can be shown with st.image."""
    parts, last = [], 0
    for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)\)", md):
        if md[last:m.start()].strip():
            parts.append(("md", md[last:m.start()]))
        img = Path(base_dir) / m.group(1)
        parts.append(("img", str(img)) if img.exists() else ("md", f"*(missing image {m.group(1)})*"))
        last = m.end()
    if md[last:].strip():
        parts.append(("md", md[last:]))
    return parts


def list_result_reports(results_dir: str | Path | None = None) -> list[Path]:
    """All markdown reports under results/ (excluding per-run folders), sorted by name."""
    base = Path(results_dir) if results_dir else ROOT / "results"
    return sorted(p for p in base.rglob("*.md") if "runs" not in p.relative_to(base).parts)


def find_training_runs(root: str | Path | None = None) -> dict[str, Path]:
    """{name: run dir} for every folder containing a log.csv (checkpoints/* and results/runs/**)."""
    r = Path(root) if root else ROOT
    runs = {}
    for base in (r / "checkpoints", r / "results" / "runs"):
        if base.exists():
            for log in sorted(base.rglob("log.csv")):
                runs[str(log.parent.relative_to(r)).replace("\\", "/")] = log.parent
    return runs


def run_curves(run_dir: str | Path) -> dict[str, list[tuple[int, float]]]:
    """{"train (eval)": [(step, loss)], "val (eval)": [...], "train (batch)": [...]} from a run's log.csv."""
    rows = read_csv_rows(Path(run_dir) / "log.csv")

    def col(name):
        return [(int(r["step"]), float(r[name])) for r in rows if r.get(name)]

    return {"train (batch avg)": col("train_loss"), "train (eval)": col("eval_train_loss"), "val (eval)": col("eval_val_loss")}
