import math

import pytest
import torch

from src.model import GPTConfig, TokenAndPositionEmbedding, sinusoidal_encoding


def make_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


@pytest.mark.parametrize("pos", ["learned", "sinusoidal"])
def test_output_shape(pos):
    cfg = make_cfg(pos_encoding=pos)
    emb = TokenAndPositionEmbedding(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 10))
    assert emb(idx).shape == (4, 10, cfg.n_embd)


def test_sinusoidal_matches_formula():
    T, C = 16, 32
    pe = sinusoidal_encoding(T, C)
    assert pe.shape == (T, C)
    for pos in range(T):
        for i in range(C // 2):
            angle = pos / (10000 ** (2 * i / C))
            assert pe[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
            assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)


def test_sinusoidal_has_no_parameters_but_learned_does():
    sin_emb = TokenAndPositionEmbedding(make_cfg(pos_encoding="sinusoidal"))
    learned = TokenAndPositionEmbedding(make_cfg(pos_encoding="learned"))
    names_sin = {n for n, _ in sin_emb.named_parameters()}
    names_learned = {n for n, _ in learned.named_parameters()}
    assert names_sin == {"token_embedding.weight"}
    assert names_learned == {"token_embedding.weight", "position_embedding.weight"}


def test_position_matters():
    """Same token at different positions must give different vectors."""
    for pos in ["learned", "sinusoidal"]:
        emb = TokenAndPositionEmbedding(make_cfg(pos_encoding=pos)).eval()
        out = emb(torch.full((1, 5), 7))  # (1, 5, C): token 7 repeated
        assert not torch.allclose(out[0, 0], out[0, 1])


def test_rejects_too_long_sequence():
    cfg = make_cfg()
    with pytest.raises(AssertionError):
        TokenAndPositionEmbedding(cfg)(torch.zeros(1, cfg.block_size + 1, dtype=torch.long))


def test_gradients_flow_to_embeddings():
    emb = TokenAndPositionEmbedding(make_cfg())
    emb(torch.randint(0, 50, (2, 8))).sum().backward()
    assert emb.token_embedding.weight.grad is not None and emb.position_embedding.weight.grad is not None
