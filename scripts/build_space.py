"""Assemble a self-contained folder to push to a Hugging Face Space (Streamlit SDK).

    python scripts/export_for_demo.py                  # slim checkpoints -> app/checkpoints/
    python scripts/build_space.py --out build/space    # copy exactly what the demo needs
    cd build/space && git init && git lfs install && git remote add origin https://huggingface.co/spaces/<user>/<space> \
        && git add . && git commit -m "demo" && git push -u origin main

The Space only needs: the app, the core `src/` package, the Amharic normaliser, the small tokenizers, the bundled
font, the result files the tabs display, and the exported checkpoints. Training code paths stay in the repo.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
pyyaml
matplotlib
streamlit>=1.35
pandas
altair
"""

SPACE_README = """\
---
title: GPT from scratch (English + Amharic)
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: {streamlit_version}
app_file: app/app.py
pinned: false
---

# GPT-2-style language model, built from scratch

Interactive demo: text generation, next-token inspector, temperature comparison, attention maps,
tokenizer playground (character-level / character-BPE / byte-BPE, English vs Amharic) and an Amharic (Ge'ez script)
tokenization study. All model, tokenizer and sampling code is written from scratch in PyTorch (see `src/`);
no pretrained weights or Hugging Face model code.

The checkpoints in `app/checkpoints/` are small CPU-trained demo models.
"""

GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n*.ttf filter=lfs diff=lfs merge=lfs -text\n"

COPY_FILES = [
    "src/__init__.py", "src/model.py", "src/tokenizer.py", "src/generate.py", "src/utils.py",
    "src/data.py", "src/train.py",                     # app imports src.generate -> model; utils/data kept for completeness
    "app/__init__.py", "app/app.py", "app/lib.py",
    "data/__init__.py", "data/amharic/__init__.py", "data/amharic/normalize.py",
    "report/architecture.png", ".streamlit/config.toml",
]
COPY_DIRS = ["app/static", "app/checkpoints", "assets/tokenizers", "assets/fonts", "results/amharic", "results/data_scale"]
COPY_GLOBS = [("results", "*.md"), ("results", "*.png")]


def build(out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied = []
    for rel in COPY_FILES:
        src = ROOT / rel
        if src.exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
            copied.append(rel)
    for rel in COPY_DIRS:
        if (ROOT / rel).exists():
            shutil.copytree(ROOT / rel, out / rel)
            copied.append(rel + "/")
    for base, pattern in COPY_GLOBS:
        for f in (ROOT / base).glob(pattern):
            (out / base).mkdir(exist_ok=True)
            shutil.copy2(f, out / base / f.name)
            copied.append(f"{base}/{f.name}")
    try:
        import streamlit
        version = streamlit.__version__
    except Exception:  # pragma: no cover
        version = "1.40.0"
    (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (out / "README.md").write_text(SPACE_README.format(streamlit_version=version), encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/space")
    a = ap.parse_args()
    out = ROOT / a.out
    copied = build(out)
    n_ckpt = len(list((out / "app" / "checkpoints").glob("*/model.pt"))) if (out / "app" / "checkpoints").exists() else 0
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"built {out} ({len(copied)} entries, {size:.1f} MB, {n_ckpt} checkpoint(s))")
    if n_ckpt == 0:
        print("WARNING: no checkpoints in app/checkpoints/ - run `python scripts/export_for_demo.py` first")


if __name__ == "__main__":
    main()
