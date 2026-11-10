"""Builds notebooks/demo.ipynb (and optionally executes it so the outputs are saved in the file).

    python notebooks/build_demo_notebook.py            # write the notebook
    python notebooks/build_demo_notebook.py --execute  # write + run it top to bottom with the available checkpoints
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# GPT-2 from scratch: demo notebook

Loads a trained checkpoint, generates text, inspects next-token probabilities and shows attention maps.
Everything uses only this repository's own code (`src/model.py`, `src/tokenizer.py`, `src/generate.py`).

**Run it:** locally from the repo root (`jupyter lab notebooks/demo.ipynb`) or on Colab:
`!git clone <your repo url> && %cd <repo> && !pip install -r requirements.txt`, then run all cells.
By default it picks the best checkpoint it can find (your T4-trained `checkpoints/small`, else the CPU demo models);
change `CKPT_EN` / `CKPT_AM` below to use another.
""")

code("""
import sys, time
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():          # started from notebooks/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import torch, matplotlib.pyplot as plt
%matplotlib inline

from src.generate import generate, load_model, tokenizer_for_checkpoint, encode_prompt, last_token_logits, next_token_probs
from src.tokenizer import token_labels
from src.utils import setup_ethiopic_font

print("Ge'ez-capable plotting font:", setup_ethiopic_font())     # so Amharic labels render in plots
torch.manual_seed(0)
""")

code("""
def first_existing(*candidates):
    for c in candidates:
        if (ROOT / c).exists():
            return ROOT / c
    return None

CKPT_EN = first_existing("checkpoints/small/ckpt_best.pt", "app/checkpoints/english/model.pt", "checkpoints/demo_english/ckpt_best.pt")
CKPT_AM = first_existing("checkpoints/amharic/ckpt_best.pt", "app/checkpoints/amharic/model.pt", "checkpoints/demo_amharic/ckpt_best.pt")
print("English checkpoint:", CKPT_EN)
print("Amharic checkpoint:", CKPT_AM)

def load(path):
    model, ck = load_model(path)
    return model, tokenizer_for_checkpoint(ck, path), ck

model, tok, ck = load(CKPT_EN)
print(f"{model.num_params():,} parameters | vocab {tok.vocab_size} | trained {ck.get('step')} steps | best val loss {ck.get('best_val_loss'):.3f}")
""")

md("## 1. Model: parameter breakdown")
code("model.print_param_breakdown()")

md("## 2. Generate text\nGreedy picks the single most likely token every step; temperature sampling draws from the distribution (lower = safer, higher = more varied).")
code("""
def sample(model, tok, prompt, n=300, **kw):
    idx = encode_prompt(tok, prompt, "cpu")
    out = generate(model, idx, n, generator=torch.Generator().manual_seed(1337), **kw)
    return tok.decode(out[0].tolist())

print("--- temperature 0.8 ---");  print(sample(model, tok, "ROMEO:\\n", 300, temperature=0.8))
print("\\n--- greedy ---");         print(sample(model, tok, "ROMEO:\\n", 200, greedy=True))
""")

md("### Temperature comparison, same prompt and seed")
code("""
for T in (0.2, 0.5, 1.0, 1.5):
    print(f"\\n=== T = {T} ===")
    print(sample(model, tok, "ROMEO:\\n", 150, temperature=T))
""")

md("## 3. Next-token probabilities\nThe model outputs one probability per vocabulary entry. Here are the top 10 after a prompt.")
code("""
prompt = "To be, or not to"
idx = encode_prompt(tok, prompt, "cpu")
probs = next_token_probs(last_token_logits(model, idx)[None])[0]
p, i = probs.topk(10)
labels = token_labels(tok, i.tolist())
plt.figure(figsize=(6, 3.2)); plt.barh(labels[::-1], p.flip(0)); plt.xlabel("probability")
plt.title(f"next token after {prompt!r}"); plt.tight_layout(); plt.show()
""")

md("## 4. Attention maps\nRow = query token, column = key token; the upper triangle is zero because of the causal mask.")
code("""
sentence = "ROMEO:\\nO, where art thou? Speak, speak!"
ids = tok.encode(sentence)[: model.cfg.block_size]
_, _, attns = model(torch.tensor([ids]), return_attn=True)          # list of (1, n_head, T, T), one per layer
toks = token_labels(tok, ids)

n_layer, n_head = model.cfg.n_layer, model.cfg.n_head
fig, axes = plt.subplots(n_layer, n_head, figsize=(2.0 * n_head, 2.0 * n_layer), squeeze=False)
for l in range(n_layer):
    for h in range(n_head):
        axes[l][h].imshow(attns[l][0, h].detach(), cmap="viridis", vmin=0, vmax=1)
        axes[l][h].set_xticks([]); axes[l][h].set_yticks([]); axes[l][h].set_title(f"L{l}H{h}", fontsize=7)
fig.suptitle("All layers x heads"); plt.tight_layout(); plt.show()
""")
code("""
layer, head = 0, 0                                                  # change these to explore
A = attns[layer][0, head].detach()
plt.figure(figsize=(8, 8)); plt.imshow(A, cmap="viridis")
plt.xticks(range(len(toks)), toks, rotation=90, fontsize=7); plt.yticks(range(len(toks)), toks, fontsize=7)
plt.xlabel("key token"); plt.ylabel("query token"); plt.title(f"layer {layer}, head {head}"); plt.colorbar(fraction=0.046); plt.tight_layout(); plt.show()
# a "previous-token" head would show a bright line just below the diagonal; run experiments/exp9_attention.py for statistics
""")

md("## 5. KV cache: same output, faster\nGreedy decoding with and without the cache must give identical text; the cache avoids recomputing keys/values of earlier tokens.")
code("""
idx = encode_prompt(tok, "ROMEO:", "cpu")
n = model.cfg.block_size - idx.size(1)
res = {}
for name, cache in (("no cache", False), ("kv cache", True)):
    t0 = time.time(); out = generate(model, idx, n, greedy=True, use_cache=cache); res[name] = (out, n / (time.time() - t0))
print({k: f"{v[1]:.0f} tokens/s" for k, v in res.items()}, "| identical:", torch.equal(res["no cache"][0], res["kv cache"][0]))
""")

md("## 6. Training curves of this checkpoint")
code("""
import csv
log = CKPT_EN.parent / "log.csv"
if log.exists():
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    def col(c): return [(int(r["step"]), float(r[c])) for r in rows if r.get(c)]
    plt.figure(figsize=(6, 3.5))
    for c, lab in (("eval_train_loss", "train (eval)"), ("eval_val_loss", "val (eval)")):
        xs, ys = zip(*col(c)); plt.plot(xs, ys, "o-", label=lab)
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3); plt.show()
else:
    print("no log.csv next to this checkpoint (exported checkpoints carry only the weights)")
""")

md("## 7. Amharic (Ge'ez script)\nSame code, different tokenizer and corpus. Byte-level BPE needs 3 tokens per Ge'ez letter before any merge; character-level models need a bigger alphabet.")
code("""
if CKPT_AM:
    am_model, am_tok, am_ck = load(CKPT_AM)
    print(f"{am_model.num_params():,} parameters | vocab {am_tok.vocab_size} | best val loss {am_ck.get('best_val_loss'):.3f}")
    for T in (0.5, 0.8, 1.0):
        print(f"\\n=== T = {T} ===")
        print(sample(am_model, am_tok, "የኢትዮጵያ ", 200, temperature=T))
else:
    print("no Amharic checkpoint found: train one with `python -m src.train --config configs/demo_amharic.yaml`")
""")
code("""
from src.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer
text = "ሰላም ለዓለም። የኢትዮጵያ ሕዝብ ሰላም ነው።"
tok_dir = ROOT / "assets" / "tokenizers"
if (tok_dir / "amharic_bpe_byte.json").exists():
    for name in ("char", "bpe_char", "bpe_byte"):
        t = load_tokenizer(tok_dir / f"amharic_{name}.json")
        t = t.truncated(1000) if name != "char" else t
        ids = t.encode(text)
        print(f"{name:9s} vocab {t.vocab_size:5d}  {len(ids):3d} tokens  ({len(text.encode()) / len(ids):.2f} bytes/token):", " | ".join(token_labels(t, ids, visible=False)))
else:
    print("run `python experiments/exp11_amharic_study.py` to build assets/tokenizers/")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    path = HERE / "demo.ipynb"
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).preprocess(nb)
    nbf.write(nb, path)
    print("wrote", path, "(executed)" if a.execute else "")


if __name__ == "__main__":
    main()
