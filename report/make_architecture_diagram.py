"""Draw report/architecture.png: data flow through the GPT with tensor shapes.

    python report/make_architecture_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_IO, C_EMB, C_NORM, C_ATTN, C_MLP, C_ADD, C_HEAD = "#dde9f5", "#e6f2e1", "#f3ecd9", "#f6dccb", "#e5dcf0", "#eeeeee", "#d9efef"


def box(ax, cx, cy, w, h, text, color, size=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#555555", lw=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, text=None, dx=0.12, size=7.5, color="#333333", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=1.0, color=color))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2, text, fontsize=size, color="#b03030", va="center", ha="left", family="monospace")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 11.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.3, 22.8)
    ax.axis("off")
    x = 3.2   # main column

    box(ax, x, 22.2, 4.2, 0.8, "token ids  idx", C_IO, bold=True)
    arrow(ax, (x, 21.8), (x, 21.0), "(B, T)")
    box(ax, x - 1.7, 20.5, 2.6, 1.1, "token embedding\nnn.Embedding(V, C)", C_EMB)
    box(ax, x + 1.8, 20.5, 3.0, 1.1, "position embedding\nlearned (block_size, C)\nor sinusoidal", C_EMB)
    arrow(ax, (x - 1.7, 19.95), (x - 0.4, 19.3))
    arrow(ax, (x + 1.8, 19.95), (x + 0.4, 19.3))
    box(ax, x, 19.0, 4.2, 0.6, "add  +  dropout", C_ADD)
    arrow(ax, (x, 18.7), (x, 17.9), "(B, T, C)")

    # ---- transformer block (pre-norm) ----
    ax.add_patch(FancyBboxPatch((0.5, 7.0), 5.4, 10.9, boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec="#888888", lw=1.4, ls="--"))
    ax.text(0.7, 17.6, "Block  (x N = n_layer)", fontsize=10, fontweight="bold", va="center")
    box(ax, x, 16.9, 3.6, 0.6, "LayerNorm  ln1", C_NORM)
    arrow(ax, (x, 16.6), (x, 15.85))
    box(ax, x, 15.4, 3.6, 0.9, "Causal self-attention\n(multi-head, detail ->)", C_ATTN, bold=True)
    arrow(ax, (x, 14.95), (x, 14.3), "(B, T, C)")
    box(ax, x, 13.95, 3.6, 0.6, "+  residual  (x + attn)", C_ADD)
    arrow(ax, (x, 13.65), (x, 12.9))
    box(ax, x, 12.5, 3.6, 0.6, "LayerNorm  ln2", C_NORM)
    arrow(ax, (x, 12.2), (x, 11.45))
    box(ax, x, 10.85, 3.6, 1.2, "MLP (position-wise)\nLinear(C, 4C) -> GELU\n-> Linear(4C, C) -> dropout", C_MLP, bold=False)
    arrow(ax, (x, 10.25), (x, 9.6), "(B, T, C)")
    box(ax, x, 9.25, 3.6, 0.6, "+  residual  (x + mlp)", C_ADD)
    ax.text(x, 8.2, "pre-norm (GPT-2):  x = x + Attn(LN(x));  x = x + MLP(LN(x))\npost-norm (Vaswani):  x = LN(x + Attn(x));  x = LN(x + MLP(x))\n(selected by config: norm_position)",
            ha="center", va="center", fontsize=7.3, color="#444444")
    # residual skip arrows
    ax.add_patch(FancyArrowPatch((1.3, 17.0), (1.3, 14.0), connectionstyle="arc3,rad=0", arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#777777"))
    ax.add_patch(FancyArrowPatch((1.3, 14.0), (x - 1.8, 13.95), arrowstyle="-", lw=0.9, color="#777777"))
    ax.add_patch(FancyArrowPatch((1.3, 14.0), (1.3, 9.3), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#777777"))
    ax.add_patch(FancyArrowPatch((1.3, 9.3), (x - 1.8, 9.25), arrowstyle="-", lw=0.9, color="#777777"))

    arrow(ax, (x, 7.0), (x, 6.3), "(B, T, C)")
    box(ax, x, 5.95, 4.2, 0.6, "final LayerNorm  ln_f", C_NORM)
    arrow(ax, (x, 5.65), (x, 4.9), "(B, T, C)")
    box(ax, x, 4.45, 4.2, 0.9, "LM head  Linear(C, V, bias=False)\nweight tied with token embedding", C_HEAD)
    arrow(ax, (x, 4.0), (x, 3.25), "(B, T, V)")
    box(ax, x, 2.9, 4.2, 0.7, "logits  ->  softmax / cross-entropy(targets)", C_IO, bold=True)

    # ---- attention detail (right) ----
    ax.add_patch(FancyBboxPatch((7.0, 8.3), 5.7, 14.3, boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec="#888888", lw=1.4, ls="--"))
    ax.text(7.2, 22.3, "Causal self-attention (detail)", fontsize=10, fontweight="bold", va="center")
    ax.annotate("", xy=(6.9, 15.4), xytext=(x + 1.8, 15.4), arrowprops=dict(arrowstyle="-|>", color="#b0552a", lw=1.2, ls="--"))
    cx = 9.85
    steps = [
        (21.5, "x", "(B, T, C)", C_IO),
        (20.2, "c_attn = Linear(C, 3C);  split into Q, K, V", "each (B, T, C)", C_ATTN),
        (18.9, "view + transpose: C -> n_head x head_size", "Q, K, V: (B, nh, T, hs)", C_ATTN),
        (17.6, "scores = Q K^T / sqrt(hs)", "(B, nh, T, T)", C_ATTN),
        (16.3, "causal mask: masked_fill(j > i, -inf)", "(B, nh, T, T)", C_ATTN),
        (15.0, "softmax over keys  (+ attention dropout)", "attn: (B, nh, T, T)", C_ATTN),
        (13.7, "weighted sum:  attn @ V", "(B, nh, T, hs)", C_ATTN),
        (12.4, "transpose + view: concatenate heads", "(B, T, C)", C_ATTN),
        (11.1, "output projection  W_O = Linear(C, C)  + dropout", "(B, T, C)", C_ATTN),
    ]
    for i, (y, label, shape, col) in enumerate(steps):
        box(ax, cx, y, 5.2, 0.7, label, col, size=8)
        if i < len(steps) - 1:
            nxt = steps[i + 1][0]
            arrow(ax, (cx, y - 0.37), (cx, nxt + 0.37))
            ax.text(cx + 0.2, (y + nxt) / 2, shape, fontsize=7.2, color="#b03030", va="center", ha="left", family="monospace")
    ax.text(cx + 0.2, 10.35, "(B, T, C)", fontsize=7.2, color="#b03030", va="center", ha="left", family="monospace")
    ax.text(9.85, 9.4, "n_head = 1 recovers single-head attention;\nhead_size = C / n_head;  scores use 1/sqrt(head_size)", ha="center", fontsize=7.3, color="#444444")

    ax.text(6.5, 0.7, "B = batch, T = sequence length (<= block_size), C = n_embd, V = vocab size, nh = n_head, hs = head size",
            ha="center", fontsize=8.5, style="italic")
    ax.set_title("GPT-2-style model: data flow and tensor shapes", fontsize=13, fontweight="bold")
    out = Path(__file__).with_name("architecture.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
