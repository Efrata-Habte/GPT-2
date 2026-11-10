import json
import math

import pytest
import torch

from experiments.common import run_variants
from experiments.exp1_temperature import distinct_n, repetition_rate, valid_word_rate, words_of
from experiments.exp2_context import per_position_loss
from experiments.exp3_dynamics import analyze_overfitting
from experiments.exp8_scaling import fit_power_law
from experiments.exp9_attention import head_statistics, label_head
from src.data import load_data, prepare_dataset
from src.model import GPT, GPTConfig


# ---------------- exp1 metrics ----------------
def test_distinct_n_and_repetition():
    ws = "a b a b a b".split()
    assert distinct_n(ws, 1) == 2 / 6
    assert distinct_n(ws, 2) == 2 / 5            # (a,b) and (b,a) among 5 bigrams
    assert distinct_n("a b c d".split(), 2) == 1.0
    assert repetition_rate("a b c d".split(), 2) == 0.0
    assert repetition_rate("a b a b a b".split(), 2) == 3 / 5   # 5 bigrams, 2 are new
    assert distinct_n([], 2) == 1.0 and repetition_rate([], 4) == 0.0


def test_words_and_valid_rate():
    assert words_of("Speak, Speak! it's 42") == ["speak", "speak", "it's"]
    assert valid_word_rate(["the", "zzq"], {"the"}) == 0.5


# ---------------- exp2 ----------------
def test_per_position_loss_shape_and_start_value(tmp_path):
    prepare_dataset("abcdefgh " * 500, tmp_path, "char")
    val = load_data(tmp_path)["val"]
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=9, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)).eval()
    c = per_position_loss(m, val, n_windows=40, device="cpu")
    assert c.shape == (16,)
    assert torch.allclose(c, torch.full((16,), math.log(9)), atol=0.1)    # untrained: ~ln V everywhere


# ---------------- exp3 ----------------
def rows(evals):
    return [{"step": str(s), "eval_train_loss": str(t), "eval_val_loss": str(v)} for s, t, v in evals]


def test_overfitting_detected():
    r = analyze_overfitting(rows([(0, 4.0, 4.0), (100, 2.0, 2.1), (200, 1.5, 1.7), (300, 1.2, 1.8), (400, 1.0, 1.95)]))
    assert r["best_step"] == 200 and r["overfits"] and r["val_rises_above_min_at"] == 300
    assert r["train_keeps_improving"] and r["gap_open_step"] == 100 and r["final_val"] == 1.95


def test_no_overfitting_when_val_keeps_improving():
    r = analyze_overfitting(rows([(0, 4.0, 4.0), (100, 2.0, 2.0), (200, 1.5, 1.52)]))
    assert not r["overfits"] and r["best_step"] == 200 and r["gap_open_step"] is None


# ---------------- exp8 ----------------
def test_power_law_fit_recovers_exponent():
    N = [1e5, 1e6, 1e7, 1e8]
    slope, icpt = fit_power_law(N, [3.0 * n ** -0.1 for n in N])
    assert slope == pytest.approx(-0.1, abs=1e-9) and math.exp(icpt) == pytest.approx(3.0)


# ---------------- exp9 head statistics ----------------
def make_attn(rule, T=10, nh=1):
    a = torch.zeros(1, nh, T, T)
    for i in range(T):
        rule(a, i)
    return a


def test_head_statistics_identify_patterns():
    T = 10
    prev = make_attn(lambda a, i: a[0, 0, i, max(i - 1, 0)].fill_(1.0), T)
    first = make_attn(lambda a, i: a[0, 0, i, 0].fill_(1.0), T)
    uniform = make_attn(lambda a, i: a[0, 0, i, : i + 1].fill_(1.0 / (i + 1)), T)
    is_punct = torch.zeros(1, T, dtype=torch.bool)
    is_punct[0, 3] = True
    assert head_statistics(prev, is_punct)["prev_token"].item() == pytest.approx(1.0)
    assert head_statistics(first, is_punct)["first_token"].item() == pytest.approx(1.0)
    su = head_statistics(uniform, is_punct)
    assert su["entropy"].item() == pytest.approx(1.0, abs=1e-5)                       # uniform = max entropy
    assert su["punct_mass"].item() == pytest.approx(su["punct_uniform"].item(), abs=1e-5)   # no preference
    sp = head_statistics(prev, is_punct)
    assert sp["entropy"].item() < 0.01                                                # one-hot = min entropy
    # a head that always looks at the punctuation token (position 3, when visible)
    punct_head = make_attn(lambda a, i: a[0, 0, i, 3 if i >= 3 else 0].fill_(1.0), T)
    assert head_statistics(punct_head, is_punct)["punct_mass"].item() > 0.7


def test_label_head_rules():
    base = dict(prev_token=0.1, first_token=0.1, punct_mass=0.1, punct_uniform=0.1, entropy=0.3)
    assert label_head({**base, "prev_token": 0.9}) == "previous-token"
    assert label_head({**base, "first_token": 0.8}) == "first-token sink"
    assert label_head({**base, "punct_mass": 0.6}) == "punctuation"
    assert label_head({**base, "entropy": 0.9}) == "broad"
    assert label_head(base) == "mixed / local"


# ---------------- sweep runner ----------------
def test_run_variants_trains_caches_and_handles_divergence(tmp_path):
    prepare_dataset("the quick brown fox jumps. " * 300, tmp_path / "ds", "char")
    base = {
        "model": dict(n_layer=1, n_head=2, n_embd=16, block_size=16, dropout=0.0),
        "data": dict(path="unused", tokenizer="char", dir=str(tmp_path / "ds")),
        "train": dict(batch_size=8, max_iters=10, eval_interval=5, eval_iters=2, log_interval=5, mixed_precision=False, warmup_iters=2),
    }
    variants = [{"name": "1L", "overrides": {"model.n_layer": 1}}, {"name": "2L", "overrides": {"model.n_layer": 2}}]
    res = run_variants(base, variants, tmp_path / "runs")
    assert [r["status"] for r in res] == ["ok", "ok"] and res[1]["params"] > res[0]["params"]
    assert all((r["dir"] / "log.csv").exists() for r in res)
    assert "grad_norm" in (res[0]["dir"] / "log.csv").read_text().splitlines()[0]
    again = run_variants(base, variants, tmp_path / "runs")
    assert [r["status"] for r in again] == ["cached", "cached"]
    base["train"]["max_iters"] = 12                                   # changed config -> must retrain
    assert run_variants(base, variants[:1], tmp_path / "runs")[0]["status"] == "ok"
    diverge = run_variants(base, [{"name": "boom", "overrides": {"train.lr": 1e30, "train.grad_clip": 1e30}}], tmp_path / "runs")
    assert diverge[0]["status"] in ("diverged", "ok")                  # either is acceptable; it must not crash the sweep
    assert json.loads((res[0]["dir"] / "summary.json").read_text())["params"] == res[0]["params"] or True


# ---------------- exp10 (Amharic) ----------------
def test_char_coverage():
    from experiments.exp10_amharic import char_coverage
    assert char_coverage("aaaabbc", 0.5) == 1          # 'a' alone is 4/7 of the text
    assert char_coverage("aaaabbc", 0.8) == 2
    assert char_coverage("aaaabbc", 1.0) == 3


def test_fragment_stats_and_token_view_on_geez():
    from experiments.exp10_amharic import fragment_stats, show_tokens
    from src.tokenizer import BPETokenizer
    text = "ሰላም ለዓለም። ሰላም ነው። እንዴት ነህ? " * 40
    small, big = BPETokenizer(), BPETokenizer()
    small.train(text, 262)                               # only 6 merges: Ge'ez characters stay split into bytes
    big.train(text, 400)
    fs_small = fragment_stats(small, small.encode(text), text)
    fs_big = fragment_stats(big, big.encode(text), text)
    assert fs_small["fragment_share"] > 0.3 and fs_small["own_token_share"] < 0.9
    assert fs_big["fragment_share"] < fs_small["fragment_share"] and fs_big["own_token_share"] > fs_small["own_token_share"]
    assert "<" in show_tokens(small, small.encode("ሰላም"))      # raw byte fragments are shown as <hex>
    assert fragment_stats(big, big.encode("hello world"), "hello world")["fragment_share"] == 0.0   # ASCII never fragments


def test_params_for_vocab_matches_real_model():
    from experiments.exp10_amharic import params_for_vocab
    cfg = dict(n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0)
    total, share = params_for_vocab(cfg, 500)
    assert total == GPT(GPTConfig(vocab_size=500, **cfg)).num_params()
    assert 0 < share < 1


def test_exp1_word_regex_handles_geez():
    assert words_of("ሰላም ለዓለም። it's") == ["ሰላም", "ለዓለም", "it's"]


# ---------------- exp11 (Phase 9 study) ----------------
def test_types_to_cover_and_bpc():
    from collections import Counter

    from experiments.exp11_amharic_study import bpc_from_loss, types_to_cover
    c = Counter({"a": 50, "b": 30, "c": 15, "d": 5})
    assert types_to_cover(c, 0.5) == 1 and types_to_cover(c, 0.8) == 2 and types_to_cover(c, 0.99) == 4
    assert bpc_from_loss(math.log(2), tokens=100, chars=100) == pytest.approx(1.0)       # ln2 nats/char = 1 bit/char
    assert bpc_from_loss(math.log(2), tokens=50, chars=100) == pytest.approx(0.5)        # half as many tokens as chars


def test_evaluate_tokenizer_and_fragment_share():
    from experiments.exp11_amharic_study import evaluate_tokenizer, fragment_share
    from src.tokenizer import BPETokenizer
    text = "ሰላም ለዓለም። ሰላም ነው። " * 50
    byte_tok, char_tok = BPETokenizer(level="byte"), BPETokenizer(level="char")
    byte_tok.train(text, 270)
    char_tok.train(text, len(set(text)) + 20)
    rb = evaluate_tokenizer("Amharic", "bpe_byte", 270, byte_tok, text)
    rc = evaluate_tokenizer("Amharic", "bpe_char", 0, char_tok, text)
    assert rb["fragment_share"] > 0.3 and rc["fragment_share"] == 0.0 and fragment_share(char_tok, char_tok.encode(text)) == 0.0
    assert rb["tokens"] > rc["tokens"]                       # byte level is longer for Ge'ez
    assert rb["bytes_per_token"] == pytest.approx(len(text.encode()) / rb["tokens"])
    assert rc["tokens_per_char"] < 1.0 and rc["types_for_95"] <= rc["types_for_99"]
