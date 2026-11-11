"""Export trained checkpoints for the web demo / a Hugging Face Space.

Training checkpoints contain optimizer state (3x the size) that inference does not need. This writes slim
`app/checkpoints/<name>/model.pt` files (weights + architecture + tokenizer) that the Streamlit app loads.

    python scripts/export_for_demo.py      # demo_english(_full) + demo_amharic(_full) (CPU demo models)
    python scripts/export_for_demo.py --english checkpoints/small/ckpt_best.pt --amharic checkpoints/amharic/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.lib import export_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--english", default="checkpoints/demo_english/ckpt_best.pt")
    ap.add_argument("--amharic", default="checkpoints/demo_amharic/ckpt_best.pt")
    ap.add_argument("--english_full", default="checkpoints/demo_english_full/ckpt_best.pt")
    ap.add_argument("--amharic_full", default="checkpoints/demo_amharic_full/ckpt_best.pt")
    ap.add_argument("--out", default="app/checkpoints")
    a = ap.parse_args()
    for key, src in (("english", a.english), ("amharic", a.amharic), ("english_full", a.english_full), ("amharic_full", a.amharic_full)):
        src_p = ROOT / src
        if not src_p.exists():
            print(f"[skip] {key}: {src} not found")
            continue
        out = export_checkpoint(src_p, ROOT / a.out / key, meta={"source": src})
        print(f"[ok] {key}: {src} -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
