"""Experiment 3: training dynamics. Train vs. validation loss and where overfitting starts.

    python experiments/exp3_dynamics.py                      # trains configs/exp_overfit.yaml (if needed), then analyses
    python experiments/exp3_dynamics.py --log checkpoints/small/log.csv   # analyse an existing run only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import COLORS, enter_repo_root, md_table, plt, read_log, series, write_report

from src.train import train
from src.utils import apply_overrides, load_config


def analyze_overfitting(rows: list[dict], margin: float = 0.02, gap_threshold: float = 0.05) -> dict:
    """Locate the point where the model starts to overfit, from the evaluation rows of a training log.

    best_step      step of minimum validation loss (the best checkpoint).
    overfits       True if val loss later rises more than `margin` above that minimum while training goes on.
    gap_open_step  first evaluation where val - train > gap_threshold (generalisation gap opens).
    """
    ev = [(int(r["step"]), float(r["eval_train_loss"]), float(r["eval_val_loss"])) for r in rows if r.get("eval_val_loss")]
    if len(ev) < 2:
        raise ValueError("need at least two evaluations")
    bi = min(range(len(ev)), key=lambda i: ev[i][2])
    best_step, tr_best, best_val = ev[bi]
    later = ev[bi + 1:]
    rise_step = next((s for s, _, v in later if v > best_val + margin), None)
    gap_open = next((s for s, t, v in ev if v - t > gap_threshold), None)
    return {
        "best_step": best_step, "best_val": best_val, "train_at_best": tr_best,
        "final_step": ev[-1][0], "final_train": ev[-1][1], "final_val": ev[-1][2],
        "overfits": rise_step is not None, "val_rises_above_min_at": rise_step,
        "gap_open_step": gap_open, "gap_at_best": best_val - tr_best, "gap_final": ev[-1][2] - ev[-1][1],
        "train_keeps_improving": ev[-1][1] < tr_best - 1e-9,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/exp_overfit.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--log", default=None, help="analyse this log.csv instead of training")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    enter_repo_root()

    if a.log:
        log_path, name = Path(a.log), Path(a.log).parent.name
    else:
        cfg = apply_overrides(load_config(a.base), a.set)
        log_path, name = Path(cfg["train"]["out_dir"]) / "log.csv", Path(cfg["train"]["out_dir"]).name
        if a.force or not (log_path.parent / "summary.json").exists():
            train(cfg)
        else:
            print(f"reusing finished run in {log_path.parent} (use --force to retrain)")
    rows = read_log(log_path)
    r = analyze_overfitting(rows)

    xs_t, ys_t = series(rows, "eval_train_loss")
    xs_v, ys_v = series(rows, "eval_val_loss")
    xs_b, ys_b = series(rows, "train_loss")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(xs_b, ys_b, color=COLORS[0], alpha=0.25, lw=1, label="train (batch avg)")
    ax.plot(xs_t, ys_t, "-", color=COLORS[0], lw=2, label="train (eval)")
    ax.plot(xs_v, ys_v, "-", color=COLORS[1], lw=2, label="validation (eval)")
    ax.axvline(r["best_step"], color="gray", ls="--", lw=1)
    ax.annotate(f"min val {r['best_val']:.3f}\nstep {r['best_step']}", (r["best_step"], r["best_val"]),
                xytext=(8, 22), textcoords="offset points", fontsize=8)
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title(f"Training dynamics: {name}")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(top=min(max(ys_v[1:] + ys_t[1:]) + 0.1, 3.0) if len(ys_v) > 1 else None)
    gap = [v - t for t, v in zip(ys_t, ys_v)]
    ax2.plot(xs_v, gap, color=COLORS[3], lw=2)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("val - train")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp3_dynamics.png", dpi=150)

    if r["overfits"]:
        verdict = (f"**Overfitting detected.** Validation loss is lowest at step {r['best_step']} ({r['best_val']:.4f}) and then rises: "
                   f"it exceeds that minimum by more than 0.02 from step {r['val_rises_above_min_at']} on, ending at {r['final_val']:.4f}, "
                   f"while the training loss " + ("keeps falling" if r["train_keeps_improving"] else "stays flat") +
                   f" ({r['train_at_best']:.4f} -> {r['final_train']:.4f}).")
    else:
        verdict = (f"**No clear overfitting within this run.** Validation loss never rose more than 0.02 above its minimum "
                   f"({r['best_val']:.4f} at step {r['best_step']}); it ended at {r['final_val']:.4f}. Train more steps, remove dropout "
                   "or use a bigger model to reach the overfitting regime.")
    rows_md = [
        ["min validation loss", f"{r['best_val']:.4f} at step {r['best_step']}"],
        ["train loss at that step", f"{r['train_at_best']:.4f}"],
        ["final train / val loss", f"{r['final_train']:.4f} / {r['final_val']:.4f} (step {r['final_step']})"],
        ["first step with val - train > 0.05", str(r["gap_open_step"])],
        ["gap (val - train) at best step / final", f"{r['gap_at_best']:.4f} / {r['gap_final']:.4f}"],
    ]
    write_report(out / "exp3_dynamics.md", "Experiment 3: training dynamics", [
        f"Run: `{log_path}`.", "![dynamics](exp3_dynamics.png)", md_table(["quantity", "value"], rows_md), verdict,
        "How to read it: before the minimum, both curves fall together (the model is learning general structure). "
        "Afterwards the train curve keeps falling while validation turns up: the model is memorising training text. "
        "Regularisation (dropout, weight decay), more data, or early stopping at the best checkpoint (`ckpt_best.pt`) address it.",
    ])


if __name__ == "__main__":
    main()
