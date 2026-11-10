"""Experiment 8 (bonus): scaling plot, parameter count vs. best validation loss.

Trains a ladder of model sizes (same data, steps, seed), optionally merges finished runs from other experiments
(--extra_runs dir ...), and fits a power law  loss ~ a * N^slope  in log-log space.

    python experiments/exp8_scaling.py
    python experiments/exp8_scaling.py --extra_runs results/runs/exp4_n_layer results/runs/exp4_n_embd

Caveat: with a fixed (short) training budget, the largest models are under-trained, so the curve bends; with ~5
points the fitted slope is a rough description, not a scaling law.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (COLORS, STANDARD_HEADERS, add_sweep_args, enter_repo_root, load_base, md_table, plt, run_variants,
                                standard_result_rows, write_report)

LADDER = [(2, 2, 64), (2, 4, 128), (4, 4, 128), (4, 4, 256), (6, 6, 384)]   # (n_layer, n_head, n_embd)


def fit_power_law(params: list[float], losses: list[float]) -> tuple[float, float]:
    """Least-squares line through (log N, log L). Returns (slope, intercept): L ~ exp(intercept) * N^slope."""
    xs, ys = [math.log(p) for p in params], [math.log(l) for l in losses]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def main() -> None:
    ap = argparse.ArgumentParser()
    add_sweep_args(ap, "configs/exp8_scaling.yaml")
    ap.add_argument("--extra_runs", nargs="*", default=[], help="folders of finished runs (containing summary.json) to include")
    ap.add_argument("--no_train", action="store_true", help="only aggregate --extra_runs, train nothing")
    a = ap.parse_args()
    enter_repo_root()
    base = load_base(a)

    results = []
    if not a.no_train:
        variants = [{"name": f"L{l}-H{h}-E{e}", "overrides": {"model.n_layer": l, "model.n_head": h, "model.n_embd": e}} for l, h, e in LADDER]
        results = run_variants(base, variants, a.runs_dir or f"{a.out}/runs/exp8_scaling", a.force)
    for folder in a.extra_runs:
        for s in sorted(Path(folder).glob("*/summary.json")):
            results.append({"name": s.parent.name, "status": "cached", "dir": s.parent, **json.loads(s.read_text())})
    ok = sorted((r for r in results if r["status"] != "diverged"), key=lambda r: r["params"])
    if len(ok) < 3:
        raise SystemExit("need at least 3 finished runs for a scaling fit")

    N, L = [r["params"] for r in ok], [r["best_val_loss"] for r in ok]
    slope, intercept = fit_power_law(N, L)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(N, L, "o", color=COLORS[0], ms=7)
    xs = [min(N) * (max(N) / min(N)) ** (i / 50) for i in range(51)]
    ax.plot(xs, [math.exp(intercept) * x ** slope for x in xs], "--", color=COLORS[1], label=f"fit: loss ~ N^{slope:.3f}")
    for r in ok:
        ax.annotate(r["name"], (r["params"], r["best_val_loss"]), xytext=(5, 6), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("best validation loss (log scale)")
    ax.set_title("Scaling: parameters vs. validation loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp8_scaling.png", dpi=150)
    write_report(out / "exp8_scaling.md", "Experiment 8: scaling", [
        f"Base config `{a.base}`; fixed steps/seed/data, model size varies. Power-law fit in log-log space: "
        f"loss ~ N^{slope:.3f} (n = {len(ok)} points).",
        md_table(STANDARD_HEADERS, standard_result_rows(ok)),
        "![scaling](exp8_scaling.png)",
        "Caveat: short fixed training budget (bigger models are under-trained) and few points; read the slope as descriptive only.",
    ])


if __name__ == "__main__":
    main()
