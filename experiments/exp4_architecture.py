"""Experiment 4: architecture sweep. Vary ONE hyper-parameter, keep everything else (data, steps, seed) fixed.

    python experiments/exp4_architecture.py --param n_layer --values 2 4 6
    python experiments/exp4_architecture.py --param n_head  --values 1 2 4 8
    python experiments/exp4_architecture.py --param n_embd  --values 128 256 384

Reports parameters, best/final validation loss, training time. Finished runs are cached, so you can run the three
sweeps one after another (or resume after a disconnect). Note n_embd must be divisible by n_head.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (COLORS, STANDARD_HEADERS, add_sweep_args, enter_repo_root, load_base, md_table, plot_overlay,
                                plt, run_variants, standard_result_rows, write_report)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_sweep_args(ap)
    ap.add_argument("--param", choices=["n_layer", "n_head", "n_embd"], default="n_layer")
    ap.add_argument("--values", type=int, nargs="+", default=None)
    a = ap.parse_args()
    enter_repo_root()
    values = a.values or {"n_layer": [2, 4, 6], "n_head": [1, 2, 4, 8], "n_embd": [128, 256, 384]}[a.param]

    base = load_base(a)
    variants = [{"name": f"{a.param}={v}", "overrides": {f"model.{a.param}": v}} for v in values]
    runs_dir = a.runs_dir or f"{a.out}/runs/exp4_{a.param}"
    results = run_variants(base, variants, runs_dir, a.force)

    stem = f"exp4_{a.param}"
    out = Path(a.out)
    plot_overlay(results, out / f"{stem}_curves.png", f"Architecture sweep: {a.param}")
    ok = [r for r in results if r["status"] != "diverged"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([r["params"] / 1e6 for r in ok], [r["best_val_loss"] for r in ok], "o-", color=COLORS[0])
    for r in ok:
        axes[0].annotate(r["name"].split("=")[1], (r["params"] / 1e6, r["best_val_loss"]), xytext=(4, 6), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("parameters (millions)")
    axes[0].set_ylabel("best validation loss")
    axes[1].bar([r["name"] for r in ok], [r["train_seconds"] for r in ok], color=COLORS[1])
    axes[1].set_ylabel("training time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Architecture sweep over {a.param}")
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=150)

    write_report(out / f"{stem}.md", f"Experiment 4: architecture sweep over {a.param}", [
        f"Base config `{a.base}` (overrides {a.set or 'none'}); only `model.{a.param}` varies. Same seed, steps and data for every run.",
        md_table(STANDARD_HEADERS, standard_result_rows(results)),
        f"![sweep]({stem}.png)\n\n![curves]({stem}_curves.png)",
    ])


if __name__ == "__main__":
    main()
