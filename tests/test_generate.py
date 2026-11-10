import pytest
import torch

from src.generate import benchmark, generate, sample_next
from src.model import GPT, GPTConfig

VARIANTS = [dict(), dict(pos_encoding="sinusoidal"), dict(norm_position="post")]


def make_model(seed=0, **kw) -> GPT:
    torch.manual_seed(seed)
    cfg = dict(vocab_size=40, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    cfg.update(kw)
    m = GPT(GPTConfig(**cfg))
    # random init gives near-uniform logits; perturb so greedy decisions are meaningful
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.eval()


# ---------------- KV cache correctness ----------------
@pytest.mark.parametrize("kw", VARIANTS)
def test_cached_logits_match_full_forward(kw):
    """Prefill a prefix, then decode token by token: logits equal a full forward at every step."""
    m = make_model(**kw)
    idx = torch.randint(0, 40, (2, 14))
    full, _ = m(idx)                                            # (2, 14, V)
    _, _, past = m(idx[:, :5], use_cache=True)
    for t in range(5, 14):
        step, _, past = m(idx[:, t:t + 1], past_kvs=past, use_cache=True)
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-5), t
    assert past[0][0].shape == (2, 2, 14, 16)                   # (B, nh, S, hs)


@pytest.mark.parametrize("kw", VARIANTS)
@pytest.mark.parametrize("prompt_len,new", [(3, 10), (4, 40)])   # second case runs far past block_size=16
def test_greedy_identical_with_and_without_cache(kw, prompt_len, new):
    m = make_model(**kw)
    prompt = torch.randint(0, 40, (2, prompt_len))
    a = generate(m, prompt, new, greedy=True, use_cache=False)
    b = generate(m, prompt, new, greedy=True, use_cache=True)
    assert a.shape == (2, prompt_len + new)
    assert torch.equal(a, b)


def test_seeded_sampling_identical_with_and_without_cache():
    m = make_model()
    prompt = torch.randint(0, 40, (1, 3))
    kw = dict(temperature=0.9, top_k=10, top_p=0.9)
    a = generate(m, prompt, 25, use_cache=False, generator=torch.Generator().manual_seed(7), **kw)
    b = generate(m, prompt, 25, use_cache=True, generator=torch.Generator().manual_seed(7), **kw)
    assert torch.equal(a, b)


def test_generate_is_reproducible_and_restores_mode():
    m = make_model().train()
    p = torch.zeros(1, 2, dtype=torch.long)
    a = generate(m, p, 10, generator=torch.Generator().manual_seed(1))
    b = generate(m, p, 10, generator=torch.Generator().manual_seed(1))
    c = generate(m, p, 10, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert m.training  # generate() put the model back in train mode


def test_benchmark_runs_and_matches():
    r = benchmark(make_model(), prompt_len=4, new_tokens=8, device="cpu")
    assert r["identical_output"] and r["cache_tps"] > 0 and r["no_cache_tps"] > 0


# ---------------- sampling ----------------
LOGITS = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])


def draw(n=3000, **kw):
    g = torch.Generator().manual_seed(0)
    return torch.cat([sample_next(LOGITS, generator=g, **kw) for _ in range(n)]).squeeze(1)


def test_greedy_and_zero_temperature_are_argmax():
    assert sample_next(LOGITS, greedy=True).item() == 0
    assert sample_next(LOGITS, temperature=0).item() == 0


def test_top_k_restricts_support():
    assert set(draw(top_k=2).tolist()) <= {0, 1}
    assert set(draw(top_k=1).tolist()) == {0}
    assert len(set(draw(top_k=100).tolist())) == 5   # k > V is fine


def test_top_p_keeps_smallest_nucleus():
    probs = torch.softmax(LOGITS, -1)[0]
    assert probs[0] < 0.6                               # sanity for the thresholds below
    assert set(draw(top_p=0.01).tolist()) == {0}        # always keep the best token
    nucleus = {0, 1} if probs[0] < 0.65 <= probs[0] + probs[1] else None
    assert nucleus and set(draw(top_p=0.65).tolist()) == nucleus
    assert len(set(draw(top_p=1.0).tolist())) == 5


def test_temperature_changes_sharpness():
    frac0 = lambda t: (draw(temperature=t) == 0).float().mean().item()
    assert frac0(0.2) > frac0(1.0) > frac0(3.0)


def test_sampling_matches_softmax_distribution():
    freq = torch.bincount(draw(n=20000), minlength=5).float() / 20000
    assert torch.allclose(freq, torch.softmax(LOGITS, -1)[0], atol=0.02)


# ---------------- streaming + next-token distribution (used by the web demo) ----------------
from src.generate import generate_stream, last_token_logits, next_token_probs  # noqa: E402


def test_stream_yields_the_same_tokens_as_generate():
    m = make_model()
    prompt = torch.randint(0, 40, (1, 3))
    streamed = [t.item() for t in generate_stream(m, prompt, 20, greedy=True)]
    full = generate(m, prompt, 20, greedy=True)[0, 3:].tolist()
    assert streamed == full and len(streamed) == 20


def test_stream_is_lazy_and_restores_mode():
    m = make_model().train()
    it = generate_stream(m, torch.zeros(1, 2, dtype=torch.long), 5, greedy=True)
    next(it)
    it.close()                       # stop early (e.g. the user pressed Stop in the UI)
    assert m.training


def test_next_token_probs_is_a_distribution_and_respects_filters():
    p = next_token_probs(LOGITS, temperature=1.0)
    assert torch.allclose(p.sum(-1), torch.ones(1)) and torch.allclose(p, torch.softmax(LOGITS, -1))
    assert (next_token_probs(LOGITS, top_k=2) > 0).sum().item() == 2
    assert (next_token_probs(LOGITS, top_p=0.01) > 0).sum().item() == 1
    sharp, flat = next_token_probs(LOGITS, temperature=0.3), next_token_probs(LOGITS, temperature=3.0)
    assert sharp.max() > p.max() > flat.max()


def test_next_token_probs_matches_what_sample_next_draws():
    freq = torch.bincount(draw(n=20000, temperature=0.7, top_k=3), minlength=5).float() / 20000
    assert torch.allclose(freq, next_token_probs(LOGITS, temperature=0.7, top_k=3)[0], atol=0.02)


def test_last_token_logits_matches_forward():
    m = make_model()
    idx = torch.randint(0, 40, (1, 9))
    assert torch.allclose(last_token_logits(m, idx), m(idx)[0][0, -1], atol=1e-6)
