"""Experiment 6 (bonus): learned (GPT-2) vs. sinusoidal (Vaswani et al.) positional encoding.

    python experiments/exp6_pos_encoding.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (STANDARD_HEADERS, add_sweep_args, enter_repo_root, load_base, md_table, plot_overlay, run_variants,
                                standard_result_rows, write_report)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_sweep_args(ap)
    a = ap.parse_args()
    enter_repo_root()
    base = load_base(a)
    variants = [{"name": p, "overrides": {"model.pos_encoding": p}} for p in ("learned", "sinusoidal")]
    results = run_variants(base, variants, a.runs_dir or f"{a.out}/runs/exp6_pos", a.force)
    out = Path(a.out)
    plot_overlay(results, out / "exp6_pos_encoding.png", "Learned vs. sinusoidal positional encoding")
    write_report(out / "exp6_pos_encoding.md", "Experiment 6: learned vs sinusoidal positions", [
        f"Base config `{a.base}`. The learned table adds block_size x n_embd parameters; the sinusoidal one adds none "
        "(it also scales token embeddings by sqrt(n_embd), see Phase 1).",
        md_table(STANDARD_HEADERS, standard_result_rows(results)),
        "![pos](exp6_pos_encoding.png)",
    ])


if __name__ == "__main__":
    main()
