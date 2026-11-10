"""Small concept figures for the report: report/figures/{positional_encoding,causal_mask}.png

    python report/make_concept_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.model import sinusoidal_encoding

OUT = Path(__file__).parent / "figures"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # sinusoidal positional encoding: rows = positions, columns = embedding dimensions
    pe = sinusoidal_encoding(64, 128)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    im = axes[0].imshow(pe.numpy(), cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    axes[0].set(xlabel="embedding dimension  (low dims oscillate fast, high dims slowly)", ylabel="position", title="Sinusoidal positional encoding (Vaswani et al.)")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    for d in (0, 10, 40, 100):
        axes[1].plot(pe[:, d].numpy(), label=f"dim {d}")
    axes[1].set(xlabel="position", ylabel="value", title="Four dimensions: different wavelengths")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "positional_encoding.png", dpi=150)
    plt.close(fig)

    # causal mask and its effect on attention weights
    T = 10
    mask = torch.tril(torch.ones(T, T))
    torch.manual_seed(0)
    scores = torch.randn(T, T)
    unmasked = torch.softmax(scores, -1)
    masked = torch.softmax(scores.masked_fill(mask == 0, float("-inf")), -1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, m, title in zip(axes, (mask, unmasked, masked), ("lower-triangular mask", "softmax(scores): unmasked (encoder-style)", "softmax(scores + mask): causal (GPT)")):
        ax.imshow(m.numpy(), cmap="viridis")
        ax.set(title=title, xlabel="key position j", ylabel="query position i")
        ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "causal_mask.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "positional_encoding.png", "and", OUT / "causal_mask.png")


if __name__ == "__main__":
    main()
