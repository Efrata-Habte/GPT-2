import math

import pytest
import torch
import torch.nn as nn

from src.model import MLP, Block, CausalSelfAttention, GPTConfig, LayerNorm


def make_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=50, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    mine, ref = LayerNorm(32), nn.LayerNorm(32, eps=1e-5)
    with torch.no_grad():  # non-trivial gamma/beta so we test them too
        g, b = torch.randn(32), torch.randn(32)
        mine.gamma.copy_(g)
        ref.weight.copy_(g)
        mine.beta.copy_(b)
        ref.bias.copy_(b)
    x = torch.randn(4, 10, 32) * 3 + 1
    x2 = x.clone().requires_grad_()
    x1 = x.clone().requires_grad_()
    y1, y2 = mine(x1), ref(x2)
    assert torch.allclose(y1, y2, atol=1e-5)
    g = torch.randn_like(y1)
    y1.backward(g)
    y2.backward(g)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(mine.gamma.grad, ref.weight.grad, atol=1e-4)


def test_layernorm_output_statistics():
    y = LayerNorm(64)(torch.randn(3, 5, 64) * 10 + 7)
    assert torch.allclose(y.mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(y.var(-1, unbiased=False), torch.ones(3, 5), atol=1e-3)


def test_attention_shapes_and_weights():
    cfg = make_cfg()
    attn = CausalSelfAttention(cfg).eval()
    y, w = attn(torch.randn(2, 10, cfg.n_embd), return_attn=True)
    assert y.shape == (2, 10, cfg.n_embd)
    assert w.shape == (2, cfg.n_head, 10, 10)
    assert torch.allclose(w.sum(-1), torch.ones(2, cfg.n_head, 10), atol=1e-5)  # rows sum to 1
    assert torch.all(w.triu(diagonal=1) == 0)                                    # nothing attends to the future
    assert attn(torch.randn(2, 10, cfg.n_embd))[1] is None


def test_attention_matches_naive_per_head_loop():
    """Vectorised multi-head attention == explicit loop over heads with plain 2-D matmuls."""
    torch.manual_seed(0)
    cfg = make_cfg()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 7, cfg.n_embd)
    y, _ = attn(x)

    W, b = attn.c_attn.weight, attn.c_attn.bias             # (3C, C), (3C,)
    hs, C = attn.head_size, cfg.n_embd
    outs = []
    for bi in range(2):
        heads = []
        for h in range(cfg.n_head):
            q = x[bi] @ W[h * hs:(h + 1) * hs].T + b[h * hs:(h + 1) * hs]                      # (T, hs)
            k = x[bi] @ W[C + h * hs:C + (h + 1) * hs].T + b[C + h * hs:C + (h + 1) * hs]
            v = x[bi] @ W[2 * C + h * hs:2 * C + (h + 1) * hs].T + b[2 * C + h * hs:2 * C + (h + 1) * hs]
            s = q @ k.T / math.sqrt(hs)
            s = s.masked_fill(torch.triu(torch.ones(7, 7), 1).bool(), float("-inf"))
            heads.append(torch.softmax(s, -1) @ v)
        outs.append(attn.c_proj(torch.cat(heads, dim=-1)))
    assert torch.allclose(y, torch.stack(outs), atol=1e-5)


@pytest.mark.parametrize("norm", ["pre", "post"])
def test_block_shape_and_causality(norm):
    torch.manual_seed(0)
    cfg = make_cfg(norm_position=norm)
    blk = Block(cfg).eval()
    x = torch.randn(2, 12, cfg.n_embd)
    y, _ = blk(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 8:] += torch.randn_like(x2[:, 8:])           # perturb only "future" positions >= 8
    y2, _ = blk(x2)
    assert torch.allclose(y[:, :8], y2[:, :8], atol=1e-5)   # earlier outputs unchanged
    assert not torch.allclose(y[:, 8:], y2[:, 8:])          # later outputs do change


def test_mlp_shape_and_hidden_size():
    cfg = make_cfg()
    mlp = MLP(cfg)
    assert mlp.c_fc.out_features == 4 * cfg.n_embd
    assert mlp(torch.randn(2, 5, cfg.n_embd)).shape == (2, 5, cfg.n_embd)


def test_block_pre_vs_post_differ_and_return_attn():
    cfg = make_cfg()
    blk = Block(cfg).eval()
    _, w = blk(torch.randn(1, 6, cfg.n_embd), return_attn=True)
    assert w.shape == (1, cfg.n_head, 6, 6)


def test_n_embd_must_divide_n_head():
    with pytest.raises(AssertionError):
        make_cfg(n_embd=30, n_head=4)
