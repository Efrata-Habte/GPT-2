"""Experiment 9 (interpretability): attention heatmaps and automatic head labelling.

1. Heatmaps for every layer x head on one sentence  -> results/exp9_attention_overview.png
2. Per-head statistics over many held-out windows (more robust than one sentence):
       prev-token   mean attention on the token directly before the query
       first-token  mean attention on position 0 (an "attention sink")
       punct mass   attention mass on punctuation/newline tokens, compared with what uniform attention would give
       entropy      attention entropy / log(#visible tokens): ~0 = sharply focused, ~1 = spread over everything
3. Heads labelled by simple threshold rules (heuristics!), and the best head of each kind drawn with token labels
   -> results/exp9_attention_selected.png

    python experiments/exp9_attention.py --ckpt checkpoints/small/ckpt_best.pt --sentence "ROMEO:\\nO, where art thou?"
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import default_device, enter_repo_root, load_bundle, md_table, plt, write_report

import torch

from src.data import get_batch, load_data


def head_statistics(attn: torch.Tensor, is_punct: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-head attention statistics for ONE layer.

    attn: (B, nh, T, T) causal attention weights (rows sum to 1);  is_punct: (B, T) bool.
    Returns tensors of shape (nh,), averaged over batch and query positions i >= 1.
    """
    B, nh, T, _ = attn.shape
    qpos = torch.arange(1, T)                                                   # skip i=0 (can only see itself)
    prev = attn[:, :, qpos, qpos - 1].mean(dim=(0, 2))                          # attention on token i-1
    first = attn[:, :, qpos, 0].mean(dim=(0, 2))                                # attention on token 0
    punct_mass = torch.einsum("bhij,bj->bhi", attn, is_punct.float())[:, :, 1:].mean(dim=(0, 2))
    # What uniform attention over visible tokens (j <= i) would put on punctuation, for the same windows:
    cum = is_punct.float().cumsum(1) / torch.arange(1, T + 1)                   # (B, T) share of punct among tokens <= i
    uniform_punct = cum[:, 1:].mean()
    ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1)                         # (B, nh, T) entropy of each row
    norm_ent = (ent[:, :, 1:] / torch.log(torch.arange(2, T + 1).float())).mean(dim=(0, 2))
    return {"prev_token": prev, "first_token": first, "punct_mass": punct_mass,
            "punct_uniform": uniform_punct.expand(nh), "entropy": norm_ent}


def label_head(s: dict[str, float]) -> str:
    """Heuristic rules, checked in this order. Thresholds are judgement calls, not ground truth."""
    if s["prev_token"] > 0.5:
        return "previous-token"
    if s["first_token"] > 0.5:
        return "first-token sink"
    if s["punct_mass"] > 0.3 and s["punct_mass"] > 2.0 * s["punct_uniform"]:
        return "punctuation"
    if s["entropy"] > 0.75:
        return "broad"
    return "mixed / local"


def token_labels(tok, ids: list[int]) -> list[str]:
    out = []
    for i in ids:
        s = tok.decode([i])
        out.append(s.replace("\n", "\\n").replace(" ", "·") if s.strip() or s == " " else repr(s)[1:-1])
    return out


def punct_mask(tok, ids: torch.Tensor) -> torch.Tensor:
    """True for tokens that contain no letter/digit/space, i.e. punctuation and newlines."""
    table = {}
    for i in set(ids.flatten().tolist()):
        s = tok.decode([i])
        table[i] = len(s) > 0 and not any(c.isalnum() or c == " " for c in s)
    return torch.tensor([[table[i] for i in row] for row in ids.tolist()])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/small/ckpt_best.pt")
    ap.add_argument("--sentence", default="ROMEO:\nO, where art thou? Speak, speak!")
    ap.add_argument("--n_windows", type=int, default=32)
    ap.add_argument("--stat_len", type=int, default=128, help="window length for the statistics")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    enter_repo_root()
    dev = default_device()
    model, ck, tok, data_dir = load_bundle(a.ckpt, dev)
    nl, nh = model.config.n_layer, model.config.n_head
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- statistics over held-out windows ----
    T = min(a.stat_len, model.config.block_size)
    x, _ = get_batch(load_data(data_dir)["val"], a.n_windows, T, torch.Generator().manual_seed(0), dev)
    _, _, attns = model(x, return_attention=True)
    punct = punct_mask(tok, x.cpu()).to(dev)
    stats = [{k: v.cpu() for k, v in head_statistics(att, punct).items()} for att in attns]
    labels = [[label_head({k: float(v[h]) for k, v in stats[l].items()}) for h in range(nh)] for l in range(nl)]

    # ---- the sentence ----
    ids = tok.encode(a.sentence)[: model.config.block_size]
    sx = torch.tensor([ids], device=dev)
    _, _, s_attns = model(sx, return_attention=True)
    s_attns = [t[0].cpu() for t in s_attns]                                      # each (nh, T, T)
    toks = token_labels(tok, ids)

    fig, axes = plt.subplots(nl, nh, figsize=(2.0 * nh, 2.0 * nl), squeeze=False)
    for l in range(nl):
        for h in range(nh):
            ax = axes[l][h]
            ax.imshow(s_attns[l][h], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{l}H{h}\n{labels[l][h]}", fontsize=6)
    fig.suptitle("Attention weights (rows = query token, columns = key token), all layers x heads", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "exp9_attention_overview.png", dpi=150)
    plt.close(fig)

    # ---- best head of each kind, with token labels ----
    picks = []
    for kind, key in (("previous-token", "prev_token"), ("punctuation", "punct_mass"), ("first-token sink", "first_token"), ("broad", "entropy")):
        l, h = max(((l, h) for l in range(nl) for h in range(nh)), key=lambda lh: float(stats[lh[0]][key][lh[1]]))
        picks.append((kind, key, l, h))
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.4), squeeze=False)
    for ax, (kind, key, l, h) in zip(axes[0], picks):
        ax.imshow(s_attns[l][h], cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(toks)))
        ax.set_yticks(range(len(toks)))
        ax.set_xticklabels(toks, fontsize=6, rotation=90)
        ax.set_yticklabels(toks, fontsize=6)
        weak = "" if labels[l][h] == kind else "  [weak: below label threshold]"   # best of its kind, not necessarily a real one
        ax.set_title(f"{kind} head: L{l}H{h}\n({key} = {float(stats[l][key][h]):.2f}){weak}", fontsize=9)
        ax.set_xlabel("key (attended-to) token")
        ax.set_ylabel("query token")
    fig.tight_layout()
    fig.savefig(out / "exp9_attention_selected.png", dpi=150)
    plt.close(fig)

    rows = []
    for l in range(nl):
        for h in range(nh):
            s = {k: float(v[h]) for k, v in stats[l].items()}
            rows.append([f"L{l}H{h}", f"{s['prev_token']:.2f}", f"{s['first_token']:.2f}", f"{s['punct_mass']:.2f} (uniform {s['punct_uniform']:.2f})",
                         f"{s['entropy']:.2f}", labels[l][h]])
    counts = {}
    for row in labels:
        for lab in row:
            counts[lab] = counts.get(lab, 0) + 1
    write_report(out / "exp9_attention.md", "Experiment 9: attention heatmaps", [
        f"Checkpoint `{a.ckpt}`. Statistics over {a.n_windows} held-out windows of {T} tokens; heatmaps for the sentence `{a.sentence!r}`.",
        "![overview](exp9_attention_overview.png)\n\n![selected](exp9_attention_selected.png)",
        "Head labels come from threshold rules (previous-token: >0.5 of attention on token i-1; first-token: >0.5 on position 0; "
        "punctuation: >0.3 of mass on punctuation/newline and >2x uniform; broad: normalised entropy >0.75). "
        "They are heuristics: check the heatmaps before claiming a head 'does' something.",
        "Heads per label: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        md_table(["head", "prev-token", "first-token", "punct mass", "entropy", "label"], rows),
    ])


if __name__ == "__main__":
    main()
