import math

import pytest
import torch

from src.model import GPT, GPTConfig


def make_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


VARIANTS = [
    dict(),
    dict(pos_encoding="sinusoidal"),
    dict(norm_position="post"),
    dict(tie_weights=False),
]


@pytest.mark.parametrize("kw", VARIANTS)
def test_forward_shapes(kw):
    cfg = make_cfg(**kw)
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = m(idx)
    assert logits.shape == (3, 10, cfg.vocab_size) and loss is None
    logits, loss, attns = m(idx, idx, return_attention=True)
    assert loss.ndim == 0
    assert len(attns) == cfg.n_layer and attns[0].shape == (3, cfg.n_head, 10, 10)


@pytest.mark.parametrize("kw", VARIANTS)
def test_initial_loss_is_ln_vocab(kw):
    torch.manual_seed(0)
    cfg = make_cfg(vocab_size=65, n_embd=64, **kw)
    m = GPT(cfg)
    idx = torch.randint(0, 65, (16, 16))
    _, loss = m(idx, torch.randint(0, 65, (16, 16)))
    assert loss.item() == pytest.approx(math.log(65), abs=0.15)


def test_weight_tying():
    tied, untied = GPT(make_cfg()), GPT(make_cfg(tie_weights=False))
    assert tied.output_head.weight is tied.embedding.token_embedding.weight
    assert untied.num_params() - tied.num_params() == 50 * 32
    # breakdown must agree with PyTorch's own unique-parameter count
    assert tied.num_params() == sum(p.numel() for p in tied.parameters())
    assert untied.num_params() == sum(p.numel() for p in untied.parameters())


def test_gpt2_init_statistics():
    torch.manual_seed(0)
    cfg = make_cfg(n_layer=4, n_embd=128, n_head=4)
    m = GPT(cfg)
    assert m.blocks[0].attention.query_key_value.weight.std().item() == pytest.approx(0.02, rel=0.1)
    expected = 0.02 / math.sqrt(2 * cfg.n_layer)
    for blk in m.blocks:
        assert blk.attention.output_projection.weight.std().item() == pytest.approx(expected, rel=0.1)
        assert blk.feed_forward.output_projection.weight.std().item() == pytest.approx(expected, rel=0.1)
    assert torch.all(m.blocks[0].attention.query_key_value.bias == 0)


@pytest.mark.parametrize("kw", VARIANTS)
def test_model_is_causal(kw):
    """Changing a future token must not change logits at earlier positions."""
    torch.manual_seed(0)
    m = GPT(make_cfg(**kw)).eval()
    idx = torch.randint(0, 50, (2, 12))
    idx2 = idx.clone()
    idx2[:, 8] = (idx2[:, 8] + 1) % 50
    l1, _ = m(idx)
    l2, _ = m(idx2)
    assert torch.allclose(l1[:, :8], l2[:, :8], atol=1e-5)
    assert not torch.allclose(l1[:, 8:], l2[:, 8:])


@pytest.mark.parametrize("kw", [dict(), dict(pos_encoding="sinusoidal"), dict(norm_position="post")])
def test_overfit_single_batch(kw):
    torch.manual_seed(0)
    cfg = make_cfg(vocab_size=20, n_embd=64, **kw)
    m = GPT(cfg)
    x = torch.randint(0, 20, (4, 16))
    y = torch.randint(0, 20, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
