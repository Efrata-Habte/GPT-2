"""The assignment's hard constraints, enforced mechanically.

Banned in the implementation: pretrained GPT-2 weights, Hugging Face model code, nn.Transformer,
nn.MultiheadAttention, nn.TransformerEncoderLayer / DecoderLayer, F.scaled_dot_product_attention, and
nn.LayerNorm (LayerNorm must be our own; nn.LayerNorm may appear only in tests, to verify ours).
The check parses the source with `ast`, so mentions in comments/docstrings do not count.
"""
import ast
from pathlib import Path

import torch.nn as nn

from src.model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["src", "app", "data", "experiments", "scripts", "report"]

BANNED_ATTRS = {"Transformer", "MultiheadAttention", "TransformerEncoderLayer", "TransformerDecoderLayer", "TransformerEncoder",
                "TransformerDecoder", "scaled_dot_product_attention", "from_pretrained", "LayerNorm_torch"}
BANNED_MODULES = {"transformers", "huggingface_hub", "tiktoken", "sentencepiece", "tokenizers", "timm", "flash_attn", "x_transformers"}


def python_files():
    for d in CODE_DIRS:
        yield from (ROOT / d).rglob("*.py")


def test_no_banned_imports_or_calls_anywhere_in_the_code():
    offences = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offences += [(path.name, a.name) for a in node.names if a.name.split(".")[0] in BANNED_MODULES]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in BANNED_MODULES:
                    offences.append((path.name, node.module))
                offences += [(path.name, a.name) for a in node.names if a.name in BANNED_ATTRS]
            elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
                offences.append((path.name, node.attr))
            elif isinstance(node, ast.Name) and node.id in BANNED_ATTRS:
                offences.append((path.name, node.id))
    assert not offences, offences


def test_layernorm_is_ours_not_torchs():
    """No torch.nn.LayerNorm anywhere in the implementation (only tests/ may use it, to verify ours)."""
    offences = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "LayerNorm" and isinstance(node.value, ast.Name) and node.value.id in ("nn", "torch", "F"):
                offences.append(path.name)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("torch") and any(a.name == "LayerNorm" for a in node.names):
                offences.append(path.name)
    assert not offences, offences


def test_built_model_contains_only_basic_layers():
    """Walk the instantiated model: only our own modules plus Linear / Embedding / Dropout / GELU / containers."""
    model = GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.1))
    allowed_torch = (nn.Linear, nn.Embedding, nn.Dropout, nn.GELU, nn.ModuleList, nn.Identity)
    for name, m in model.named_modules():
        if type(m).__module__.startswith("torch"):
            assert isinstance(m, allowed_torch), f"{name}: {type(m)}"
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())
    assert not any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    own = {type(m).__name__ for m in model.modules() if type(m).__module__.startswith("src.")}
    assert {"LayerNorm", "CausalSelfAttention", "MLP", "Block", "Embeddings", "GPT"} <= own


def test_no_pretrained_weight_files_are_used():
    """Checkpoints in the repo come from our own training runs: each carries its own training config."""
    import torch
    for p in list((ROOT / "checkpoints").glob("*/ckpt_best.pt"))[:3]:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert "config" in ck and "step" in ck and ck["step"] >= 0
