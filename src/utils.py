"""Small helpers: seeding and config loading. (Plotting helpers arrive in Phase 4.)"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply command-line overrides like `train.max_iters=200` or `model.n_layer=4` in place."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        value = yaml.safe_load(raw)  # parses ints, floats, bools, strings
        if isinstance(value, str):
            try:
                value = float(value)  # YAML 1.1 reads "1e-3" / "1e+30" as strings; they are floats
            except ValueError:
                pass
        node[leaf] = value
    return cfg


def plot_loss_curves(csv_path: str | Path, png_path: str | Path, title: str = "") -> None:
    """Plot train/val loss (from the training CSV log) and the learning-rate schedule."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    def col(name):
        pts = [(int(r["step"]), float(r[name])) for r in rows if r.get(name)]
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    xs, ys = col("train_loss")
    ax.plot(xs, ys, color="tab:blue", alpha=0.25, lw=1, label="train (running batch avg)")
    xs, ys = col("eval_train_loss")
    ax.plot(xs, ys, "o-", color="tab:blue", ms=3, label="train (eval)")
    xs, ys = col("eval_val_loss")
    ax.plot(xs, ys, "o-", color="tab:orange", ms=3, label="val (eval)")
    if ys:
        best = min(range(len(ys)), key=ys.__getitem__)
        ax.axvline(xs[best], color="gray", ls="--", lw=1)
        ax.annotate(f"best val {ys[best]:.3f} @ {xs[best]}", (xs[best], ys[best]), xytext=(5, 12),
                    textcoords="offset points", fontsize=8)
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title or "Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    xs, ys = col("lr")
    ax2.plot(xs, ys, color="tab:green")
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


ETHIOPIC_FONTS = ["Noto Sans Ethiopic", "Nyala", "Ebrima", "Abyssinica SIL", "Noto Serif Ethiopic"]


def setup_ethiopic_font() -> str | None:
    """Make matplotlib able to draw Ge'ez (Ethiopic) text.

    Registers the bundled Noto Sans Ethiopic (assets/fonts/, SIL OFL) and puts every available Ethiopic-capable
    font in front of DejaVu Sans in the font fallback list; matplotlib then picks a font per glyph, so English and
    Ge'ez labels can share one plot. Returns the preferred Ethiopic font family found, or None.
    """
    from matplotlib import font_manager as fm
    import matplotlib

    bundled = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansEthiopic-Regular.ttf"
    if bundled.exists():
        fm.fontManager.addfont(str(bundled))
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [n for n in ETHIOPIC_FONTS if n in available]
    matplotlib.rcParams["font.family"] = chosen + ["DejaVu Sans"]
    return chosen[0] if chosen else None
