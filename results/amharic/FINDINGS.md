# Findings: Amharic (Ge'ez script) tokenization study

Numbers come from `tokenizer_analysis.md` / `tokenizer_stats.csv` (English and Amharic each cut to 1,000,000 characters;
BPE trained on the first 90%, measured on the held-out 10%) and from the CPU demo models (same 0.42-0.46M-parameter
architecture, 2,500 steps, same seed). This note is hand-written from those numbers; regenerate the tables with
`python experiments/exp11_amharic_study.py` (add `--reuse_analysis` to skip re-training the tokenizers).

## 1. The character inventory is ~5x larger, and it is Zipfian

- 351 distinct characters in 1M characters of Amharic news vs 65 in Tiny Shakespeare. Covering 95% / 99% of the text takes
  124 / 187 characters (English: 34 / 48).
- Each Ge'ez letter is a consonant+vowel syllable and takes 3 UTF-8 bytes; with spaces, digits and Latin text mixed in the
  corpus averages **2.58 bytes per character** (English: 1.00).
- Effect on the model is smaller than the inventory suggests: in the 0.4M-parameter demo model the tied embedding grows from
  65x128 to 377x128 (+40k parameters, ~9% of the model); in the 10.8M `small` model it would be ~1% of parameters.

## 2. Byte-level BPE is costly for Ge'ez; character-level BPE is not

- A byte-level BPE starts every Ge'ez letter as **3 tokens**. Its first merges are spent re-assembling characters: the first
  five merges are UTF-8 lead-byte pairs (`E1+88`, `E1+89`, `E1+8B`, `E1+8A`, `E1+8C`), and the first complete letter
  (`ን`) only appears at merge 9.
- At vocabulary 500, **17.4% of byte-BPE tokens are partial characters** (invalid UTF-8 on their own); 4.6% at 1,000 and
  0.9% at 4,000. English never produces partial characters.
- Sequence-length cost: at vocab 500, byte-BPE needs 3.98 tokens/word vs 3.67 for char-BPE (+8%); the gap shrinks to
  3.03 vs 2.90 at 1,000 and 2.05 vs 2.03 at 4,000. With no merges at all, byte-level sequences are 2.58x longer than
  character-level ones.
- Interesting side effect in English: at small vocabularies char-BPE also beats byte-BPE (at vocab 300: 1.85 vs 1.38
  characters/token), because byte-level spends 256 slots on byte values while English uses only 65 characters.

## 3. Amharic needs more tokens per word than English at every vocabulary size

| vocab | English tokens/word (char-BPE) | Amharic tokens/word (char-BPE) |
|---|---|---|
| char-level | 5.47 | 5.16 |
| 500 | 2.58 | 3.67 |
| 1,000 | 2.26 | 2.90 |
| 4,000 | 1.77 | 2.03 |

- Character-level, the two are similar (Amharic words have about as many letters as English words, though each letter is a syllable).
- After merging, Amharic needs 15-42% more tokens per word (42% at vocab 500, 28% at 1,000, 15% at 4,000): the language is morphologically rich. The training split has
  **38,143 word types vs 10,994** for English, and **16.6% of validation words never occur in training** (English 5.5%), so
  the same vocabulary budget covers less of the language.

## 4. What BPE discovers

- Char-BPE's first merges attach a space to the most common **word-initial syllables**: `·የ`, `·በ`, `·አ`, `·እ`, `·መ`, `·ከ`, `·ተ`, `·ለ`.
  These include the prefixes የ- (of / relative), በ- (in, by), ከ- (from), ለ- (for).
- Then come suffixes and whole function words. Among the most frequent multi-character tokens are `ነው`, `ላይ`, `እና`, `የሚ`,
  `የተ`, `ውስጥ`, `ነበር`, `ወደ`, `ጋር`, `ግን`, the plural suffixes `ዎች` / `ቶች`, `ቸው`, and content words such as `ሰዎች`, `ጊዜ`, `ዓመት`.
  (Glosses here are my reading. Please check them; you know the language better than I do.)
- English char-BPE at the same size learns `·the`, `·to`, `·of`, `·and`, `'s`, `ing`, `ed`: similar kinds of units (function words, affixes).

## 5. Homophone normalization helps a little, and it is lossy

- Folding ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ (all seven vowel orders) changes only **1.2% of characters**, but removes 28 distinct
  characters (351 → 323) and 678 word types (38,143 → 37,465, −1.8%). Unseen validation words fall from 16.6% to 16.2%.
- Fertility moves by about 1-3% (char-BPE at 500: 3.67 → 3.57 tokens/word; at 1,000: 2.90 → 2.87).
- The benefit is consistency (one spelling per word); the cost is that a model trained on normalized text can never produce the
  folded-away spellings. Train one with `--set data.normalize=homophones` to measure the effect on loss (score it on normalized text,
  which is easier by construction, so the comparison is not like-for-like).

## 6. English vs Amharic models: choose the unit carefully

| model (T4, 10.8-10.9M parameters, 3,000 steps) | val bits/char | val bits/byte | val bits/word |
|---|---|---|---|
| English, char-level | 2.164 | 2.164 | 11.8 |
| Amharic, char-level | 2.388 | 0.925 | 12.4 |

- Per character, Amharic looks harder (a syllable carries more information than a Latin letter). Per byte it looks much easier, and per
  word the two are close (12.4 against 11.8 bits). None of these is "the" answer; the report should say which one it uses and why.
- Confound: the English corpus is 1.1M characters and the Amharic one 3.75M, so 3,000 steps of 64 x 256 characters is about 49 epochs of
  English but 15 of Amharic. The English model overfits (best validation loss at step 1,750); the Amharic one does not. The corpora also
  differ in genre (plays vs news).

## 7. Samples

- With the T4 model, temperature 0.8 produces sentences that look like Amharic news: real names, ministries and places, Ethiopic
  punctuation, and locally grammatical phrases, but no coherent story across sentences. Greedy decoding still tends to repeat phrases.
- See `samples.json` / `tokenizer_analysis.md` for samples at several temperatures, and the demo's Generate tab to try your own prompts.
  (The glosses and the judgement of grammaticality are mine; a native speaker should check them.)

## 8. More data: Wikipedia + news vs news only (experiment 12)

Same 10.9M-parameter architecture, 3,000 steps and seed on a Colab T4; the full model's training text is 9.9M characters
(Wikipedia + the same news) against 3.4M. Both are scored on text neither saw (`results/data_scale/data_scale.md`):

| test text | news-only model | Wikipedia + news model |
|---|---|---|
| held-out news | **2.385** bits/char | 2.500 (4.8% worse) |
| held-out Wikipedia | 3.784 | **2.729** (27.9% better) |

- The news-only model is far out of domain on encyclopedic text (3.78 against 2.39 bits/char on news): the topic and register shift
  matters more than the script.
- The mixed model generalises much better, but with a fixed budget of 3,000 steps it saw about a third as much news, so it is slightly
  worse on news. The same pattern appeared with the small CPU models (15.9% better on Wikipedia, 5.9% worse on news).

## Limitations

- One seed per model and small CPU models: differences in the second decimal of bits/char are not reliable.
- News text only; Amharic Wikipedia or literature would shift the vocabulary and the merge lists.
- My pre-tokenizer splits on `\w` (Ethiopic letters count as word characters, Ethiopic punctuation does not), which suits Amharic,
  but merges never cross word boundaries, so multi-word expressions are never single tokens.
