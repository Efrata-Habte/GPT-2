"""Experiment 2: prompt length / context. How does the model's loss depend on how much context it has seen,
and how do generations change with a short vs. long prompt?

Part A  per-position loss: average next-token loss at each position 0..block_size-1 of held-out windows.
        (Position 0 has no context at all; later positions have more.)
Part B  loss on the next K tokens after an L-token prompt, read off the curve above.
Part C  generations from the same passage cut to a short / medium / long prompt.

    python experiments/exp2_context.py --ckpt checkpoints/small/ckpt_best.pt

Caveat: with absolute position embeddings, "more context" and "later position" are the same thing; the
model was trained on windows that start at position 0, so this curve is exactly the training distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import COLORS, default_device, enter_repo_root, load_bundle, md_table, plt, write_report

import torch
import torch.nn.functional as F

from src.data import get_batch, load_data
from src.generate import generate


@torch.no_grad()
def per_position_loss(model, data, n_windows: int, device: str, seed: int = 0, batch: int = 64) -> torch.Tensor:
    """Mean cross-entropy at each position over `n_windows` random windows. Returns (block_size,)."""
    model.eval()
    T, V = model.config.block_size, model.config.vocab_size
    gen = torch.Generator().manual_seed(seed)
    total, n = torch.zeros(T), 0
    while n < n_windows:
        b = min(batch, n_windows - n)
        x, y = get_batch(data, b, T, gen, device)                                   # (b, T)
        logits, _ = model(x)                                                         # (b, T, V)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").view(b, T)
        total += loss.sum(0).cpu()
        n += b
    return total / n


def smooth(v: list[float], k: int = 5) -> list[float]:
    h = k // 2
    return [sum(v[max(0, i - h):i + h + 1]) / len(v[max(0, i - h):i + h + 1]) for i in range(len(v))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/small/ckpt_best.pt")
    ap.add_argument("--n_windows", type=int, default=500)
    ap.add_argument("--prompt_lens", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--k", type=int, default=32, help="Part B: number of tokens scored after the prompt")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--passage_start", type=int, default=2000, help="token offset in the val split for Part C")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    enter_repo_root()

    dev = default_device()
    model, ck, tok, data_dir = load_bundle(a.ckpt, dev)
    val = load_data(data_dir)["val"]
    T = model.config.block_size

    # ---- Part A ----
    curve = per_position_loss(model, val, a.n_windows, dev).tolist()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(T), curve, color=COLORS[0], alpha=0.35, lw=1)
    ax.plot(range(T), smooth(curve), color=COLORS[0], lw=2, label="loss at position t (5-pt average)")
    for i, L in enumerate(a.prompt_lens):
        if L < T:
            ax.axvline(L, color=COLORS[1 + i % 3], ls="--", lw=1, label=f"prompt length {L}")
    ax.set_xscale("symlog", linthresh=8)
    ax.set_xlabel("position in the context window (tokens of context before it)")
    ax.set_ylabel("held-out cross-entropy (nats)")
    ax.set_title("Loss vs. position in the context window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp2_loss_by_position.png", dpi=150)

    edges = [(0, 1), (1, 8), (8, 32), (32, 64), (64, 128), (128, T)]
    bucket_rows = [[f"{lo}-{hi - 1}", f"{sum(curve[lo:hi]) / (hi - lo):.4f}"] for lo, hi in edges if lo < T]

    # ---- Part B ----
    Ls = sorted({1, 2, 4, 8, 16, 32, 64, 128, *a.prompt_lens})
    b_rows = [[L, f"{sum(curve[L - 1:L - 1 + a.k]) / len(curve[L - 1:L - 1 + a.k]):.4f}"] for L in Ls if L - 1 + a.k <= T]

    # ---- Part C ----
    Lmax = max(a.prompt_lens)
    passage = [int(t) for t in val[a.passage_start:a.passage_start + Lmax]]
    gen_sections = []
    for L in a.prompt_lens:
        ids = torch.tensor([passage[Lmax - L:]], device=dev)          # the LAST L tokens of the passage
        prompt_text = tok.decode(ids[0].tolist())
        blocks = []
        for name, kw in (("greedy", dict(greedy=True)), ("T=0.8", dict(temperature=0.8))):
            g = torch.Generator().manual_seed(a.seed)
            o = generate(model, ids, a.max_new_tokens, generator=g, **kw)
            blocks.append(f"*{name}:*\n```\n{tok.decode(o[0, L:].tolist())[:500]}\n```")
        gen_sections.append(f"### prompt length {L} tokens\nPrompt:\n```\n{prompt_text}\n```\n" + "\n".join(blocks))

    write_report(out / "exp2_context.md", "Experiment 2: prompt length and context", [
        f"Checkpoint `{a.ckpt}`, {a.n_windows} held-out windows of {T} tokens.",
        "## A. Loss by position in the window\n![loss by position](exp2_loss_by_position.png)\n\n" + md_table(["positions", "mean loss"], bucket_rows),
        f"## B. Loss on the {a.k} tokens after an L-token prompt\n\n" + md_table(["prompt length L", f"mean loss on next {a.k} tokens"], b_rows),
        "## C. Generations from the same passage with different prompt lengths\n\n" + "\n\n".join(gen_sections),
    ])


if __name__ == "__main__":
    main()
