"""Shared helpers for the experiment scripts (run them from the repo root)."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import data_dir_from_config
from src.generate import load_model, tokenizer_for_checkpoint
from src.train import train
from src.utils import apply_overrides, load_config, setup_ethiopic_font

setup_ethiopic_font()  # Ge'ez labels (token names, Amharic samples) must render in every plot

# Okabe-Ito colour-blind-safe palette
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def enter_repo_root() -> None:
    """Configs use repo-relative paths (data/input.txt), so scripts chdir to the repo root."""
    os.chdir(ROOT)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------
# Loading a trained model together with its tokenizer and data
# ---------------------------------------------------------------------------------------
def load_bundle(ckpt: str | Path, device: str):
    """-> (model, checkpoint dict, tokenizer, data_dir)."""
    model, ck = load_model(ckpt, device)
    data_dir = data_dir_from_config(ck["config"]["data"])
    tok = tokenizer_for_checkpoint(ck, ckpt)
    return model, ck, tok, data_dir


# ---------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------
def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(path: str | Path, title: str, sections: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def read_log(csv_path: str | Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows: list[dict], col: str) -> tuple[list[int], list[float]]:
    pts = [(int(r["step"]), float(r[col])) for r in rows if r.get(col)]
    return [p[0] for p in pts], [p[1] for p in pts]


# ---------------------------------------------------------------------------------------
# Training sweeps
# ---------------------------------------------------------------------------------------
def add_sweep_args(ap: argparse.ArgumentParser, default_base: str = "configs/exp_base.yaml") -> None:
    ap.add_argument("--base", default=default_base, help="base config YAML")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override the base config (e.g. train.max_iters=200)")
    ap.add_argument("--out", default="results", help="where the .md/.png outputs go")
    ap.add_argument("--runs_dir", default=None, help="where run folders go (default: <out>/runs/<experiment>)")
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run with the same config exists")


def run_variants(base_cfg: dict, variants: list[dict], runs_dir: str | Path, force: bool = False) -> list[dict]:
    """Train each variant ({"name":..., "overrides": {"model.n_layer": 2, ...}}) and collect summaries.

    - Every run uses the same seed, so differences come from the variant, not from luck.
    - Finished runs (summary.json + identical config) are reused: safe to re-run after a Colab disconnect.
    - A run whose loss becomes NaN/inf is recorded as status="diverged" instead of aborting the sweep.
    """
    runs_dir = Path(runs_dir)
    results = []
    for v in variants:
        name, out = v["name"], runs_dir / v["name"]
        cfg = copy.deepcopy(base_cfg)
        for key, val in v.get("overrides", {}).items():   # set directly: YAML would read "1e-05" as a string
            *parents, leaf = key.split(".")
            node = cfg
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = val
        cfg["train"]["out_dir"] = str(out)
        out.mkdir(parents=True, exist_ok=True)
        cfg_file, summ_file = out / "config_used.json", out / "summary.json"
        if not force and summ_file.exists() and cfg_file.exists() and json.loads(cfg_file.read_text()) == cfg:
            summary, status = json.loads(summ_file.read_text()), "cached"
            print(f"[{name}] reusing finished run")
        else:
            cfg_file.write_text(json.dumps(cfg, indent=2))
            summ_file.unlink(missing_ok=True)
            print(f"\n===== training variant: {name} =====")
            try:
                summary, status = train(cfg), "ok"
            except RuntimeError as e:
                if "loss became" not in str(e):
                    raise
                summary, status = {}, "diverged"
                print(f"[{name}] DIVERGED: {e}")
        results.append({"name": name, "status": status, "dir": out, "overrides": v.get("overrides", {}), **summary})
    return results


def plot_overlay(results: list[dict], png: str | Path, title: str,
                 panels=(("eval_val_loss", "validation loss"), ("train_loss", "train loss (batch avg)"))) -> None:
    """Overlay the loss curves of several runs, one panel per metric."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (col, label) in zip(axes, panels):
        for i, r in enumerate(results):
            log = Path(r["dir"]) / "log.csv"
            if not log.exists():
                continue
            xs, ys = series(read_log(log), col)
            ax.plot(xs, ys, color=COLORS[i % len(COLORS)], lw=1.6, label=r["name"])
        ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)


def standard_result_rows(results: list[dict]) -> list[list]:
    rows = []
    for r in results:
        if r["status"] == "diverged":
            rows.append([r["name"], "-", "diverged", "-", "-", "-", "-"])
            continue
        rows.append([r["name"], f"{r['params']:,}", f"{r['best_val_loss']:.4f}", f"{r['final_val_loss']:.4f}",
                     f"{r['val_bits_per_char']:.3f}", f"{r['train_seconds']:.0f}", f"{r['mean_tokens_per_sec']:,.0f}"])
    return rows


STANDARD_HEADERS = ["run", "params", "best val loss", "final val loss", "val bits/char", "train time (s)", "tokens/s"]


def load_base(args) -> dict:
    return apply_overrides(load_config(args.base), args.set)
