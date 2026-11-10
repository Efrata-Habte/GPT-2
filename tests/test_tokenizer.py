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


def naive_bpe_merges(text: str, vocab_size: int) -> dict:
    """Reference implementation: recount ALL pairs from scratch after every merge (slow, obviously right)."""
    from collections import Counter

    from src.tokenizer import _merge_pair

    freq = Counter(_PRETOKENIZE.findall(text))
    words = {w: list(w.encode("utf-8")) for w in freq}
    merges = {}
    for new_id in range(256, vocab_size):
        counts = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                counts[p] += freq[w]
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        if counts[best] < 2:
            break
        merges[best] = new_id
        words = {w: _merge_pair(ids, best, new_id) for w, ids in words.items()}
    return merges


@pytest.mark.parametrize("text", [SAMPLE, "ሰላም ለዓለም። ሰላም ነው። እንዴት ነህ? " * 30, "aaaa aaab abab baba " * 40])
def test_incremental_bpe_matches_naive_reference(text):
    tok = BPETokenizer()
    tok.train(text, vocab_size=400)
    assert tok.merges == naive_bpe_merges(text, 400)


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


# ---------------- char-level BPE and Amharic (Phase 1 / Phase 6 additions) ----------------
AMHARIC = "ሰላም ለዓለም። እንዴት ነህ? የኢትዮጵያ ሕዝብ ሰላም ነው። ሰላም ለሁሉም፣ ሰላም ለሁላችን። " * 25
ENGLISH_AND_AMHARIC = "Hello, world! ሰላም ለዓለም። 123 don't ሰላም\n\nnew line"


def make_tokenizers(text: str) -> dict:
    chars = CharTokenizer.from_text(text)
    cbpe = BPETokenizer(level="char")
    cbpe.train(text, vocab_size=len(set(text)) + 40)
    bbpe = BPETokenizer(level="byte")
    bbpe.train(text, vocab_size=256 + 40)
    return {"char": chars, "bpe_char": cbpe, "bpe_byte": bbpe}


@pytest.mark.parametrize("text", [SAMPLE, AMHARIC, ENGLISH_AND_AMHARIC])
def test_all_tokenizers_round_trip_english_and_amharic(text):
    for name, tok in make_tokenizers(text).items():
        ids = tok.encode(text)
        assert tok.decode(ids) == text, name
        assert max(ids) < tok.vocab_size, name


def test_byte_bpe_handles_unseen_amharic_but_char_bpe_does_not():
    train_text = SAMPLE                                     # English only
    byte_tok, char_tok = BPETokenizer(level="byte"), BPETokenizer(level="char")
    byte_tok.train(train_text, 300)
    char_tok.train(train_text, len(set(train_text)) + 20)
    assert byte_tok.decode(byte_tok.encode("ሰላም")) == "ሰላም"          # bytes cover everything
    with pytest.raises(KeyError):
        char_tok.encode("ሰላም")                                        # unseen character: no <unk>


def test_char_bpe_alphabet_argument_and_base_size():
    tok = BPETokenizer(level="char")
    tok.train("abab abab", vocab_size=10, alphabet="xyz")             # extra alphabet chars are in the base vocab
    assert tok.base_size == len(set("ab ") | set("xyz")) and tok.decode(tok.encode("xyz ab")) == "xyz ab"
    assert tok.vocab_size == tok.base_size + len(tok.merges)


def test_char_bpe_starts_one_token_per_character_byte_bpe_starts_at_three_for_geez():
    char_tok, byte_tok = BPETokenizer(level="char"), BPETokenizer(level="byte")
    char_tok.train("ሰላም", 3)                                         # no useful merges yet (nothing repeats)
    byte_tok.train("ሰላም", 256)
    assert len(char_tok.encode("ሰላም")) == 3 and len(byte_tok.encode("ሰላም")) == 9   # 3 chars x 3 bytes


@pytest.mark.parametrize("level", ["char", "byte"])
def test_truncated_and_persistence_keep_level(level, tmp_path):
    big = BPETokenizer(level=level)
    big.train(AMHARIC, 400 if level == "byte" else len(set(AMHARIC)) + 60)
    small = big.truncated(big.base_size + 10)
    assert small.level == level and len(small.merges) == 10 and small.vocab_size == big.base_size + 10
    assert len(small.encode(AMHARIC)) > len(big.encode(AMHARIC))     # fewer merges -> longer sequences
    big.save(tmp_path / "t.json")
    again = load_tokenizer(tmp_path / "t.json")
    assert again.level == level and again.encode(AMHARIC) == big.encode(AMHARIC)
    from src.tokenizer import tokenizer_from_dict, tokenizer_to_dict
    assert tokenizer_from_dict(tokenizer_to_dict(big)).encode(AMHARIC) == big.encode(AMHARIC)


def test_incremental_char_bpe_matches_naive_reference():
    from collections import Counter

    from src.tokenizer import _merge_pair

    text = AMHARIC + SAMPLE
    tok = BPETokenizer(level="char")
    tok.train(text, vocab_size=len(set(text)) + 60)
    alphabet = sorted(set(text))
    c2i = {c: i for i, c in enumerate(alphabet)}
    freq = Counter(_PRETOKENIZE.findall(text))
    words = {w: [c2i[c] for c in w] for w in freq}
    merges = {}
    for new_id in range(len(alphabet), len(alphabet) + 60):
        counts = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                counts[p] += freq[w]
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        if counts[best] < 2:
            break
        merges[best] = new_id
        words = {w: _merge_pair(ids, best, new_id) for w, ids in words.items()}
    assert tok.merges == merges


def test_token_labels_shows_fragments_and_visible_whitespace():
    from src.tokenizer import token_labels

    byte_tok = BPETokenizer(level="byte")
    byte_tok.train("ሰላም ", 256)
    labels = token_labels(byte_tok, byte_tok.encode("ሰ a\n"))
    assert labels[0].startswith("<E1") and "·" in "".join(labels) and "\\n" in "".join(labels)
