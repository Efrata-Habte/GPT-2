import json

import pytest

from data.amharic.normalize import (FAMILIES, build_table, changed_positions, homophone_counts, normalize_amharic,
                                    normalize_homophones)
from src.data import default_data_dir, load_meta, normalize_text, prepare_dataset
from src.tokenizer import load_tokenizer

TEXT = "ሰላም ሠላም ሐገር ሀገር ኀይል ሀይል ዓለም ኣለም ፀሐይ ጸሐይ። ሑሉ ሁሉ ዐ አ ፣ ፤ ፡ end"


def test_homophone_folding_maps_each_family_to_its_canonical_letter():
    assert normalize_homophones("ሐ ኀ") == "ሀ ሀ"                     # h family  -> ሀ
    assert normalize_homophones("ሠ") == "ሰ"                         # s family  -> ሰ
    assert normalize_homophones("ዐ") == "አ"                         # a family  -> አ
    assert normalize_homophones("ፀ") == "ጸ"                         # ts family -> ጸ
    assert normalize_homophones("ሑ ሒ ሓ ሔ ሕ ሖ") == "ሁ ሂ ሃ ሄ ህ ሆ"      # all vowel orders move together
    assert normalize_homophones("ኃይል ዓለም ፅ") == "ሃይል ኣለም ጽ"        # ዓ is the 4th order, so it maps to ኣ (4th order), not አ


def test_homophone_spellings_of_one_word_collapse_to_one():
    w = normalize_homophones(TEXT).split()
    assert w[0] == w[1]        # ሰላም / ሠላም
    assert w[2] == w[3]        # ሐገር / ሀገር
    assert w[4] == w[5]        # ኀይል / ሀይል
    assert w[6] == w[7]        # ዓለም / ኣለም
    assert w[8] == w[9].rstrip("።")   # ፀሐይ vs ጸሐይ followed by ።
    assert w[10] != "" and normalize_homophones("ኀይል") != normalize_homophones("ኃይል")   # different vowel orders stay different


def test_folding_is_length_preserving_idempotent_and_leaves_other_text_alone():
    out = normalize_homophones(TEXT)
    assert len(out) == len(TEXT) and normalize_homophones(out) == out
    for ch in "።፣፤፡ end":                                        # punctuation, spaces and ASCII untouched
        assert out.count(ch) == TEXT.count(ch)
    assert normalize_homophones("ሰላም ነው") == "ሰላም ነው"            # canonical letters unchanged


def test_table_structure_and_eighth_slot_not_folded():
    table = build_table()
    assert len(table) == 7 * sum(len(f["variants"]) for f in FAMILIES.values())
    assert all(0x1200 <= k <= 0x135A and 0x1200 <= v <= 0x135A for k, v in table.items())   # only Ge'ez letters involved
    assert chr(0x1217) not in "".join(map(chr, table))             # ሗ (HHWA, labialised 8th slot) is not folded


def test_modes_and_wordspace():
    assert normalize_amharic(TEXT, None) == TEXT and normalize_amharic(TEXT, "none") == TEXT
    assert normalize_amharic("ሐ", "homophones") == "ሀ"
    assert normalize_amharic("ሰላም፡ነው", "homophones") == "ሰላም፡ነው"                       # wordspace kept by default
    assert normalize_amharic("ሰላም፡ነው", "homophones+wordspace") == "ሰላም ነው"
    with pytest.raises(ValueError):
        normalize_amharic(TEXT, "bogus")


def test_counts_and_changed_positions():
    c = homophone_counts("ሐሐ ሠ ዐ ሰ")
    assert c["ሐ"] == 2 and c["ሠ"] == 1 and c["ዐ"] == 1 and "ሰ" not in c
    assert changed_positions("ሐ ሰ", normalize_homophones("ሐ ሰ")) == [0]


# ---------------- pipeline integration ----------------
def test_default_data_dir_names():
    assert default_data_dir("data/x.txt", "char", 512).name == "x_char"
    assert default_data_dir("data/x.txt", "bpe", 1000).name == "x_bpe1000"
    assert default_data_dir("data/x.txt", "bpe_byte", 1000).name == "x_bpe1000"
    assert default_data_dir("data/x.txt", "bpe_char", 1000).name == "x_cbpe1000"
    assert default_data_dir("data/x.txt", "char", 512, "homophones").name == "x_char_norm"
    assert default_data_dir("data/x.txt", "char", 512, "none").name == "x_char"


@pytest.mark.parametrize("tokenizer", ["char", "bpe_char", "bpe_byte"])
def test_prepare_dataset_for_amharic_with_every_tokenizer(tokenizer, tmp_path):
    text = (TEXT + "\n") * 60
    meta = prepare_dataset(text, tmp_path, tokenizer, bpe_vocab_size=400, normalize="homophones")
    assert meta["normalize"] == "homophones" and meta["train_chars"] == int(0.9 * len(text))   # folding keeps the length
    tok = load_tokenizer(tmp_path / "tokenizer.json")
    norm = normalize_text(text, "homophones")
    assert tok.decode(tok.encode(norm)) == norm
    assert "ሐ" not in norm and "ሠ" not in norm and load_meta(tmp_path)["vocab_size"] == tok.vocab_size


def test_normalization_shrinks_the_character_vocabulary(tmp_path):
    text = (TEXT + "\n") * 60
    raw = prepare_dataset(text, tmp_path / "raw", "char")
    norm = prepare_dataset(text, tmp_path / "norm", "char", normalize="homophones")
    assert norm["vocab_size"] < raw["vocab_size"]
