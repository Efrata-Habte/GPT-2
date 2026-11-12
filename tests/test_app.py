"""Tests for the Streamlit demo: its pure logic (app/lib.py) and the UI itself (headless AppTest)."""
from pathlib import Path

import pytest
import torch

from app import lib
from src.data import prepare_dataset
from src.tokenizer import BPETokenizer, CharTokenizer
from src.train import train

ROOT = Path(__file__).resolve().parents[1]
EN = "To be, or not to be: that is the question. Whether tis nobler in the mind to suffer. " * 40
AM = "ሰላም ለዓለም። የኢትዮጵያ ሕዝብ ሰላም ነው። ሰላም ለሁሉም፣ ሰላም ለሁላችን። ሐገር ሠላም ዐለም። " * 40


def _train_tiny(tmp, name, text, tokenizer):
    ddir, out = tmp / f"ds_{name}", tmp / f"run_{name}"
    prepare_dataset(text, ddir, tokenizer, bpe_vocab_size=300)
    cfg = {"model": dict(n_layer=2, n_head=2, n_embd=32, block_size=32, dropout=0.0),
           "data": dict(path="unused", tokenizer=tokenizer, dir=str(ddir)),
           "train": dict(batch_size=8, max_iters=6, eval_interval=3, eval_iters=1, log_interval=3,
                         mixed_precision=False, out_dir=str(out))}
    train(cfg)
    return out / "ckpt_best.pt"


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    """app/checkpoints-style folder with a tiny char-level English and a tiny byte-BPE Amharic model."""
    tmp = tmp_path_factory.mktemp("demo")
    root = tmp / "exported"
    lib.export_checkpoint(_train_tiny(tmp, "en", EN, "char"), root / "english")
    lib.export_checkpoint(_train_tiny(tmp, "am", AM, "bpe_byte"), root / "amharic")
    lib.export_checkpoint(_train_tiny(tmp, "enf", EN + EN.upper(), "char"), root / "english_full")   # bigger "full-data" English
    return root


# ---------------- export / loading ----------------
def test_exported_checkpoint_is_slim_and_loads(ckpt_dir):
    p = ckpt_dir / "amharic" / "model.pt"
    ck = torch.load(p, weights_only=False)
    assert "optimizer" not in ck and {"model", "model_config", "tokenizer"} <= set(ck)
    model, tok, info = lib.load_demo_model(p)
    assert info["vocab_size"] == tok.vocab_size == model.config.vocab_size and info["params"] == model.num_params()
    assert tok.decode(tok.encode(AM[:50])) == AM[:50]


def test_find_checkpoint_respects_env_override(ckpt_dir, monkeypatch):
    monkeypatch.setenv("GPT_DEMO_CKPT_DIR", str(ckpt_dir))
    assert lib.find_checkpoint("english") == ckpt_dir / "english" / "model.pt"
    assert lib.find_checkpoint("klingon") is None


# ---------------- streaming ----------------
def test_incremental_decoder_never_emits_half_a_character():
    tok = BPETokenizer(level="byte")
    tok.train(AM, 280)
    text = "ሰላም ለዓለም።"
    dec, pieces = lib.IncrementalDecoder(tok), []
    for i in tok.encode(text):
        piece = dec.push(i)
        assert "�" not in piece
        pieces.append(piece)
    pieces.append(dec.flush())
    assert "".join(pieces) == text
    assert any(p == "" for p in pieces[:-1])        # some steps correctly produced nothing (partial character)


def test_stream_text_equals_generate_text_for_both_models(ckpt_dir):
    for key, prompt in (("english", "To be"), ("amharic", "ሰላም ")):
        model, tok, _ = lib.load_demo_model(ckpt_dir / key / "model.pt")
        streamed = "".join(lib.stream_text(model, tok, prompt, 25, method="greedy"))
        assert streamed == lib.generate_text(model, tok, prompt, 25, greedy=True)

        def seeded():
            return "".join(lib.stream_text(model, tok, prompt, 25, "top-p", 0.9, top_p=0.8, seed=5))

        assert seeded() == seeded()                 # same seed, same text


def test_decoding_kwargs():
    assert lib.decoding_kwargs("greedy", 0.5, 10, 0.5) == {"greedy": True}
    assert lib.decoding_kwargs("temperature", 0.5, 10, 0.5) == {"temperature": 0.5}
    assert lib.decoding_kwargs("top-k", 0.7, 10, 0.5) == {"temperature": 0.7, "top_k": 10}
    assert lib.decoding_kwargs("top-p", 0.7, 10, 0.5) == {"temperature": 0.7, "top_p": 0.5}
    with pytest.raises(ValueError):
        lib.decoding_kwargs("beam", 1, 1, 1)


def test_prompt_with_unseen_characters_is_handled(ckpt_dir):
    model, tok, _ = lib.load_demo_model(ckpt_dir / "english" / "model.pt")      # char-level English
    ids, dropped = lib.encode_prompt_safe(tok, "To be ሰ,")
    assert dropped == ["ሰ"] and tok.decode(ids) == "To be ,"
    assert lib.encode_prompt_safe(tok, "") == ([0], [])
    assert isinstance("".join(lib.stream_text(model, tok, "ሰ", 3, method="greedy")), str)   # does not crash


# ---------------- inspector / attention ----------------
def test_top_next_tokens_are_sorted_probabilities(ckpt_dir):
    model, tok, _ = lib.load_demo_model(ckpt_dir / "amharic" / "model.pt")
    ids = tok.encode("ሰላም ")
    top = lib.top_next_tokens(model, tok, ids, k=10)
    probs = [t["prob"] for t in top]
    assert len(top) == 10 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-6
    assert all(set(t) == {"id", "label", "text", "prob"} for t in top)
    only = lib.top_next_tokens(model, tok, ids, k=10, top_k=3)
    assert sum(t["prob"] > 0 for t in only) == 3                                 # top-k filter is reflected
    flat = lib.top_next_tokens(model, tok, ids, k=1, temperature=5.0)[0]["prob"]
    sharp = lib.top_next_tokens(model, tok, ids, k=1, temperature=0.2)[0]["prob"]
    assert sharp > flat


def test_attention_for_text_shapes_and_causality(ckpt_dir):
    model, tok, _ = lib.load_demo_model(ckpt_dir / "english" / "model.pt")
    labels, attn, dropped = lib.attention_for_text(model, tok, "To be, or not")
    T = len(labels)
    assert attn.shape == (model.config.n_layer, model.config.n_head, T, T) and not dropped
    assert torch.allclose(attn.sum(-1), torch.ones(attn.shape[:3]), atol=1e-5) and torch.all(attn.triu(1) == 0)
    long_labels, _, _ = lib.attention_for_text(model, tok, "e" * 200)
    assert len(long_labels) == model.config.block_size                              # cropped to the context window


# ---------------- tokenizer playground ----------------
def test_tokens_to_html_escapes_user_text():
    out = lib.tokens_to_html(["<script>alert(1)</script>", "a b", "ሰ"])
    assert "<script>" not in out and "&lt;script&gt;" in out and "ሰ" in out and out.count("<span") == 3


def test_normalization_diff_highlights_only_changed_characters():
    raw = "ሐ ሰ ዐ"
    norm, n = lib.apply_normalization(raw)
    assert norm == "ሀ ሰ አ" and n == 2
    out = lib.normalization_diff_html(raw, norm)
    assert out.count("background:rgba(213,94,0,0.35)") == 2 and "ሀ" in out


def test_load_playground_tokenizers_truncates_and_skips(tmp_path):
    chars = CharTokenizer.from_text(AM)
    cbpe, bbpe = BPETokenizer(level="char"), BPETokenizer(level="byte")
    cbpe.train(AM, len(set(AM)) + 50)
    bbpe.train(AM, 256 + 50)
    chars.save(tmp_path / "amharic_char.json")
    cbpe.save(tmp_path / "amharic_bpe_char.json")
    bbpe.save(tmp_path / "amharic_bpe_byte.json")
    both = lib.load_playground_tokenizers("amharic", 280, tmp_path)
    assert set(both) == {"char", "bpe_char", "bpe_byte"} and both["bpe_byte"].vocab_size == 280
    assert both["bpe_byte"].encode(AM[:80]) == bbpe.truncated(280).encode(AM[:80])
    small = lib.load_playground_tokenizers("amharic", 5, tmp_path)               # below the alphabet size -> char-BPE skipped
    assert "bpe_char" not in small and "char" in small
    assert lib.load_playground_tokenizers("english", 500, tmp_path) == {}


def test_tokenize_for_display_statistics():
    text = "ሰላም ነው"
    bbpe, chars = BPETokenizer(level="byte"), CharTokenizer.from_text(text)
    bbpe.train((text + " ") * 30, 258)
    r = lib.tokenize_for_display(bbpe, text)
    assert r["n_chars"] == len(text) and r["n_bytes"] == len(text.encode()) and r["fragments"] > 0
    assert r["bytes_per_token"] == pytest.approx(r["n_bytes"] / r["n_tokens"])
    c = lib.tokenize_for_display(chars, text + " ZZ")
    assert c["dropped"] == ["Z"] and c["n_tokens"] == len(text) + 1              # the trailing space is kept, 'ZZ' skipped
    assert lib.tokenize_for_display(chars, "")["n_tokens"] == 0


# ---------------- results helpers ----------------
def test_split_markdown_images(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    parts = lib.split_markdown_images("# T\n\n![fig](a.png)\n\ntext ![x](missing.png) end", tmp_path)
    kinds = [k for k, _ in parts]
    assert kinds[:2] == ["md", "img"] and parts[1][1].endswith("a.png") and any("missing image" in p for _, p in parts)


def test_find_runs_and_curves(tmp_path):
    run = tmp_path / "checkpoints" / "r1"
    run.mkdir(parents=True)
    (run / "log.csv").write_text("step,lr,train_loss,tokens_per_sec,eval_train_loss,eval_val_loss,elapsed_s\n"
                                 "0,0.1,,,4.0,4.1,1\n10,0.1,3.5,100,,,2\n20,0.1,,,3.0,3.2,3\n")
    runs = lib.find_training_runs(tmp_path)
    assert list(runs) == ["checkpoints/r1"]
    c = lib.run_curves(runs["checkpoints/r1"])
    assert c["val (eval)"] == [(0, 4.1), (20, 3.2)] and c["train (batch avg)"] == [(10, 3.5)]


def test_param_breakdown_rows_total(ckpt_dir):
    model, _, _ = lib.load_demo_model(ckpt_dir / "english" / "model.pt")
    assert lib.param_breakdown_rows(model)[-1] == ("TOTAL", model.num_params())


# ---------------- the UI itself (headless) ----------------
streamlit_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture()
def app(ckpt_dir, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.setenv("GPT_DEMO_CKPT_DIR", str(ckpt_dir))
    return streamlit_testing.AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120)


def test_app_starts_and_has_all_tabs_and_selector(app):
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 9 and app.tabs[0].label.endswith("Generate")
    assert app.sidebar.radio[0].options == ["English", "Amharic"]


def test_app_generate_works_for_both_models(app):
    app.run()
    for model_name in ("English", "Amharic"):
        app.sidebar.radio[0].set_value(model_name).run()
        assert not app.exception, [e.value for e in app.exception]
        next(b for b in app.button if b.key == "gen_go").click().run()
        assert not app.exception, [e.value for e in app.exception]


def test_app_next_token_inspector_click_appends_a_token(app):
    app.run()
    before = list(app.session_state["inspect_ids_english"])
    next(b for b in app.button if b.key == "pick_english_0").click().run()
    assert not app.exception, [e.value for e in app.exception]
    after = list(app.session_state["inspect_ids_english"])
    assert len(after) == len(before) + 1 and after[:-1] == before
    next(b for b in app.button if b.key == "undo_english").click().run()
    assert list(app.session_state["inspect_ids_english"]) == before


def test_app_temperature_comparison_for_both_models(app):
    app.run()
    next(b for b in app.button if b.key == "cmp_go").click().run()
    assert not app.exception, [e.value for e in app.exception]
    app.sidebar.radio[0].set_value("Amharic").run()
    next(b for b in app.button if b.key == "cmp_go").click().run()
    assert not app.exception, [e.value for e in app.exception]


def test_app_runs_in_a_fresh_process_like_streamlit_does(ckpt_dir):
    """Regression: `streamlit run app/app.py` puts app/ on sys.path, so `import app` could resolve to app.py itself.
    In-process AppTests miss this because pytest has already imported the `app` package; a fresh interpreter does not."""
    import os
    import subprocess
    import sys

    code = ("from streamlit.testing.v1 import AppTest\n"
            "at = AppTest.from_file('app/app.py', default_timeout=180); at.run()\n"
            "assert not at.exception, [e.value for e in at.exception]\n"
            "print('TABS', len(at.tabs))\n")
    env = {**os.environ, "GPT_DEMO_CKPT_DIR": str(ckpt_dir), "PYTHONUTF8": "1"}
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and "TABS 9" in r.stdout, (r.stdout[-500:], r.stderr[-1500:])


def test_app_training_data_selector_and_comparison_tab(app):
    app.run()
    assert app.sidebar.radio[1].options == ["Small dataset", "Full dataset"]
    app.sidebar.radio[1].set_value("Full dataset").run()                     # English full-data model exists
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.session_state["inspect_ids_english_full"]) > 0          # state is kept per model
    next(b for b in app.button if b.key == "data_go").click().run()          # side-by-side generation, both models
    assert not app.exception, [e.value for e in app.exception]
    app.sidebar.radio[0].set_value("Amharic").run()                          # no Amharic full-data model -> falls back
    assert not app.exception, [e.value for e in app.exception]
    assert any("showing the small-data model" in w.value for w in app.sidebar.warning)
