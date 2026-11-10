"""Experiment 5 (bonus): pre-norm (GPT-2) vs. post-norm (Vaswani et al.).

Post-norm is known to be harder to train, so besides the default learning rate we also run a higher one
(--lr_mult, default 3x) to stress stability. Reported: final loss, whether the run diverged, and gradient-norm
statistics (before clipping) from the training log.

    python experiments/exp5_norm.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (STANDARD_HEADERS, add_sweep_args, enter_repo_root, load_base, md_table, plot_overlay, read_log,
                                run_variants, series, standard_result_rows, write_report)


def grad_stats(r: dict) -> tuple[str, str, str]:
    log = Path(r["dir"]) / "log.csv"
    if not log.exists():
        return "-", "-", "-"
    _, g = series(read_log(log), "grad_norm")
    _, l = series(read_log(log), "train_loss")
    if not g:
        return "-", "-", "-"
    spikes = sum(1 for a, b in zip(l, l[1:]) if b > a + 0.3)   # train loss jumps up by > 0.3 nats between log points
    return f"{sum(g) / len(g):.2f}", f"{max(g):.2f}", str(spikes)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_sweep_args(ap, "configs/exp5_norm.yaml")
    ap.add_argument("--lr_mult", type=float, default=3.0, help="second, higher learning rate = lr_mult x base lr")
    a = ap.parse_args()
    enter_repo_root()
    base = load_base(a)
    lr = base["train"]["lr"]
    variants = []
    for mult, tag in ((1.0, ""), (a.lr_mult, f"@{a.lr_mult:g}xlr")):
        for norm in ("pre", "post"):
            variants.append({"name": f"{norm}-norm{tag}", "overrides": {"model.norm_position": norm, "train.lr": lr * mult,
                                                                         "train.min_lr": base["train"]["min_lr"] * mult}})
    results = run_variants(base, variants, a.runs_dir or f"{a.out}/runs/exp5_norm", a.force)

    out = Path(a.out)
    plot_overlay(results, out / "exp5_norm.png", "Pre-norm vs. post-norm",
                 panels=(("eval_val_loss", "validation loss"), ("train_loss", "train loss (batch avg)"), ("grad_norm", "grad norm (pre-clip)")))
    rows = [row + list(grad_stats(r)) for row, r in zip(standard_result_rows(results), results)]
    write_report(out / "exp5_norm.md", "Experiment 5: pre-norm vs post-norm", [
        f"Base config `{a.base}`; learning rates {lr:g} and {lr * a.lr_mult:g}. A diverged run hit a NaN/inf loss.",
        md_table(STANDARD_HEADERS + ["mean grad norm", "max grad norm", "loss spikes"], rows),
        "![norm](exp5_norm.png)",
    ])


if __name__ == "__main__":
    main()
