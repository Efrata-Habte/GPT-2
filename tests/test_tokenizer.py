from pathlib import Path

import pytest

from src.tokenizer import BPETokenizer, CharTokenizer, _PRETOKENIZE, load_tokenizer

SAMPLE = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n"
) * 20


def test_char_round_trip():
    tok = CharTokenizer.from_text(SAMPLE)
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE
    assert tok.vocab_size == len(set(SAMPLE))
    assert max(ids) < tok.vocab_size


def test_pretokenizer_is_lossless():
    for s in [SAMPLE, "a  b   c", "  lead", "trail  ", "don't won't", "héllo wörld ☃", "\n\n\t x"]:
        assert "".join(_PRETOKENIZE.findall(s)) == s


def test_bpe_round_trip_and_compression():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert 256 < tok.vocab_size <= 300
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE
    assert len(ids) < len(SAMPLE.encode("utf-8"))  # merges must compress


def test_bpe_handles_unseen_and_unicode_text():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    for s in ["zebra xylophone 12345", "héllo wörld ☃ 😀", "", " "]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(SAMPLE, 300)
    b.train(SAMPLE, 300)
    assert a.merges == b.merges


def test_bpe_truncated_equals_smaller_training():
    big = BPETokenizer()
    big.train(SAMPLE, 320)
    small = BPETokenizer()
    small.train(SAMPLE, 290)
    assert big.truncated(290).merges == small.merges


def test_save_load(tmp_path: Path):
    bpe = BPETokenizer()
    bpe.train(SAMPLE, 300)
    bpe.save(tmp_path / "bpe.json")
    chars = CharTokenizer.from_text(SAMPLE)
    chars.save(tmp_path / "char.json")
    bpe2, chars2 = load_tokenizer(tmp_path / "bpe.json"), load_tokenizer(tmp_path / "char.json")
    assert bpe2.encode(SAMPLE) == bpe.encode(SAMPLE)
    assert chars2.encode(SAMPLE) == chars.encode(SAMPLE)


@pytest.mark.skipif(not Path("data/input.txt").exists(), reason="run `python data/prepare.py` first")
def test_round_trip_on_real_data():
    text = Path("data/input.txt").read_text(encoding="utf-8")[:50_000]
    bpe = BPETokenizer()
    bpe.train(text, 400)
    assert bpe.decode(bpe.encode(text)) == text
    chars = CharTokenizer.from_text(text)
    assert chars.decode(chars.encode(text)) == text
