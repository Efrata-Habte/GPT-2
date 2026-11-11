# Findings: Amharic (Ge'ez script) tokenization study

Numbers come from `tokenizer_analysis.md` / `tokenizer_stats.csv` (English and Amharic each cut to 1,000,000 characters;
BPE trained on the first 90%, measured on the held-out 10%) and from three CPU demo models (same 0.42-0.46M-parameter
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
- Trained model: **3.073 → 3.055 val bits/char** (−0.6%). Caveat: the normalized model is scored on normalized text, which is easier by
  construction, and with one seed per model I cannot tell whether a 0.6% difference is real. Read this as "small, not obviously useful for loss";
  the bigger benefit is consistency (one spelling per word), and the cost is that the model can never produce the folded-away spellings.

## 6. English vs Amharic models: choose the unit carefully

| model (CPU demo, 2,500 steps) | val bits/char | val bits/byte | val bits/word |
|---|---|---|---|
| English, char-level | 2.348 | 2.348 | 12.8 |
| Amharic, char-level | 3.073 | 1.190 | 16.0 |
| Amharic normalized | 3.055 | 1.184 | 15.9 |

- Per character, Amharic looks harder (a syllable carries more information than a Latin letter). Per byte it looks easier.
  Per word it is ~25% more expensive, consistent with the larger word inventory. None of these is "the" answer; the report should say which one it uses and why.
- Confound: the English corpus is 1.1M characters and the Amharic one 3.75M, so 2,500 steps is about 10 epochs of English but 3 of Amharic.
  That is visible as a larger train/val gap for English (0.29 vs 0.09 bits/char). The two corpora also differ in genre (plays vs news).

## 7. Samples

- Greedy decoding loops on frequent words (`አስተዳደር አስተዳደር ...`), as it does in English.
- T = 0.5-0.8 produces runs of real-looking words with Ethiopic punctuation (`።`, `፣`) and news vocabulary (ministry, Europe, football club).
  It is not coherent text, as expected from a 0.46M-parameter model. T = 1.3 invents non-words and stray symbols.
- See `samples.json` / `tokenizer_analysis.md` for the full samples, and the demo's Generate tab to try your own prompts.

## 8. More data: Wikipedia + news vs news only (experiment 12)

Same architecture, steps and seed; the full model's training text is 9.9M characters (Wikipedia + the same news) vs 3.4M.
Both are scored on text neither saw (`results/data_scale/data_scale.md`):

| test text | news-only model | Wikipedia + news model |
|---|---|---|
| held-out news | **3.072** bits/char | 3.252 (5.9% worse) |
| held-out Wikipedia | 4.188 | **3.525** (15.9% better) |

- The news-only model is badly out of domain on encyclopedic text (4.19 vs 3.07 bits/char on news): the topic and register shift matters
  more than the script.
- The mixed model generalises much better, but with a fixed budget of 2,500 steps it saw about a third as much news, so it is worse on news.
  More data helps most when the model also gets enough capacity and steps to use it; that is what the T4 runs can show.

## Limitations

- One seed per model and small CPU models: differences in the second decimal of bits/char are not reliable.
- News text only; Amharic Wikipedia or literature would shift the vocabulary and the merge lists.
- My pre-tokenizer splits on `\w` (Ethiopic letters count as word characters, Ethiopic punctuation does not), which suits Amharic,
  but merges never cross word boundaries, so multi-word expressions are never single tokens.
