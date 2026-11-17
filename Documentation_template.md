# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Solaris
**Team Members:** Abdul Rahman RSA, Shaila Neelofar K, Pravin Lenin Naidu, Shamanth S
**Submission Date:** [fill in on final submission]

---

## 1. Executive Summary

We resolve Source-2/Source-3 business records to Source-1 reference entities with a
classical (non-neural) pipeline: script-aware name/address normalization, DuckDB
token-blocking on both name *and* address, and a LightGBM classifier scoring the
resulting candidates, with its decision threshold tuned directly against the F_0.5
metric rather than a generic classification threshold. The key insight driving the
design is that name similarity alone fails for transliterated business names (India
records span Devanagari, Tamil, and Gujarati script), while address similarity for
the *same* true-matched pairs stays consistently high — so address is treated as a
first-class blocking and matching signal throughout, not a fallback.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA was run directly against the full provided files (DuckDB, not a sample) to
confirm scale and structure before designing the pipeline:

- **Scale:** train = 2,206,821 Source-1 / 5,034,617 Source-2 / 5,285,604 Source-3
  records; test = 1,732,545 / 4,887,274 / 5,082,317. Far larger than a
  naive-comparison approach could handle — motivates blocking as the first-order
  design decision, per the challenge's own tip that blocking recall is the ceiling
  on final score.
- **Country:** train covers US (1,323,633) and India (883,188) only; test adds
  France (259,452, ~15% of test Source-1) with no train examples — the pipeline
  treats country as an open-set label throughout (no hardcoded `{US, India}`
  assumptions anywhere in normalization, blocking, or features).
- **Ground truth structure:** exclusivity is clean — 0 Source-2/3 records matched to
  more than one Source-1 entity, and 0 cross-country true matches (used as a hard,
  zero-cost blocking filter). Singleton rate 5.59%; median 3 matches per matched
  entity. 38.3% of Source-1 entities share an exact business name with another
  Source-1 entity (name alone is not sufficient to disambiguate — address matters).
  15.8% of true matched pairs share *zero* name tokens even after normalization.
  Train orphan rate (Source-2/3 records with no true match) is ~26% for both
  sources.
- **Noise patterns confirmed in the raw data:** legal-suffix inconsistency (Pvt/
  Private, Ltd/Limited, Corp/Corporation), typos, word-reordering, landmark-style
  addresses, missing postal codes/state fields, a business name that was literally
  a web domain (`agrosheronhotels.com`), and — the most consequential finding —
  India business names appearing in multiple non-Latin scripts (Devanagari, Tamil,
  and Gujarati observed directly in samples, not just Hindi as initially assumed).
  For those transliterated-name pairs, measured name similarity after
  transliteration averages only ~48 (fuzzy ratio, barely above a ~38 noise floor
  for unrelated names), while address similarity for the *same* true pairs
  averages ~96 (token_set_ratio) — address consistently stays in Latin
  script/numerals even when the name doesn't.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (token-based candidate generation feeding
a gradient-boosted binary classifier).

**Core Innovation:** Address is used as a first-class, independent blocking and
matching signal alongside name, not a secondary feature — motivated directly by
the transliteration finding above. A generic multilingual embedding model was
considered and deliberately not used: it would have added a GPU/pretrained-model
dependency for a problem that a rule-based transliteration step
(`indic_transliteration`, MIT-licensed, covers Devanagari/Tamil/Gujarati/Bengali/
Kannada/Malayalam/Telugu/Gurmukhi/Oriya) plus classical string similarity already
addresses without any pretrained-weights license question.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** normalized name tokens *and* normalized address tokens
  (both channels, unioned — not name-only with address as a fallback), joined
  under a hard country-match filter (zero-cost per the exclusivity finding above).
  Each Source-1 entity is restricted to its **N most discriminative (rarest)**
  tokens by candidate-pool document frequency before the join — this bounds join
  cost to scale linearly with entity count instead of quadratically with shared
  vocabulary, which is what makes token-blocking tractable at the full ~2.2M x
  ~10.3M scale. Overly common tokens are additionally dropped via an absolute
  document-frequency cap (a percentage-only cutoff is insufficient once the corpus
  is large: 2% of a 10M-record pool is still ~200,000 occurrences for a single
  token). Candidates are ranked per Source-1 entity by shared-token count and
  capped at the top 50.
- **Candidate pairs generated:** [fill in from the full-scale run — see
  `output/candidate_pairs.tsv`]
- **How true matches were not lost:** recall was measured directly against real
  ground-truth pairs (not a synthetic proxy) on a held-out sample built from actual
  training data: **96.75% recall** at top_k=50, with diminishing returns confirmed
  at top_k=100 (97.3%) and top_k=200 (98.1%) — the marginal recall gain did not
  justify the 2x/4x growth in candidate volume at full scale, so top_k=50 was kept
  as the production setting.

---

## 4. Matching Model

**Features used** (12 total, computed per candidate pair):
- Name features: Levenshtein ratio, token-sort ratio, token-set ratio, partial
  ratio (all via `rapidfuzz`), Jaccard token overlap, and TF-IDF cosine similarity
  over character n-grams (2-4 grams, robust to typos and transliteration noise in
  a way word-level TF-IDF is not)
- Address features: the same ratio/Jaccard set as name (address TF-IDF was not
  found to add signal beyond the ratio-based features and was left out to keep the
  feature set lean)
- Other: `shared_tokens` (carried through from the blocking stage), `name_len_diff`

**Model type:** LightGBM (gradient-boosted decision trees), trained from scratch
on our own engineered features — not a pretrained/redistributed model, so it
carries no license question under the constraint that any such model be
MIT/Apache-2.0 and <=8B parameters; LightGBM itself is MIT-licensed regardless.
300 trees, 31 leaves each, `is_unbalance=True` to account for the heavy
class imbalance in the labeled candidate set (true matches are a small minority
of blocked candidates).

**Threshold selection method:** direct F_0.5 optimization — thresholds are swept
against the real macro-averaged F_0.5 formula (matching the evaluation criteria
exactly, including singleton handling) on a held-out validation split by
Source-1 entity, not a generic metric like accuracy or plain F1. This reflects
F_0.5's precision weighting: the tuned threshold landed high (0.95), consistent
with the metric penalizing false merges twice as heavily as missed matches.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9655-0.9743 measured on held-out validation splits
  during development (~25k-entity samples built from real training data with
  genuine embedded ground-truth pairs). [Update with the full-scale validation
  score once the complete run finishes.]
- **Common false positives (wrong merges):** candidates that share several
  address tokens (city, generic street words) but differ in the specific
  identifying details (unit/plot number, distinctive surname) — the model's
  precision-weighted threshold (0.95) was chosen specifically to suppress these.
- **Common false negatives (missed matches):** the hardest cases are
  transliterated names paired with *also* significantly differing addresses
  (e.g., a shortened or reformatted address on one side) — when neither channel
  carries strong signal, blocking may still find the pair as a low-ranked
  candidate, but the classifier's conservative threshold can reject it.

---

## 6. Conclusion

Business name matching alone is insufficient for a meaningfully multilingual,
multi-country dataset like this one — the decisive finding of this project was
that address similarity remains reliable even when name similarity breaks down
under script transliteration, and building both blocking and feature engineering
around that finding (rather than reaching for a heavier multilingual embedding
model) produced a precision-weighted, MIT/Apache-license-clean pipeline that
validated at F_0.5 ≈ 0.97 on real held-out data.

---

## Appendix

### A. Code Artefacts

Runnable pipeline under `code/business_entity_resolution/`:

- `src/normalize.py` — legal-suffix stripping (US/India/France terms),
  diacritic folding, and script-general transliteration
- `src/preprocess.py` — streams each raw source TSV once into normalized Parquet
  (bounded memory regardless of file size)
- `src/blocking.py` — DuckDB SQL token-blocking (name + address, country-hard-
  filtered, country-partitioned execution, document-frequency and per-entity
  token-count caps for tractability at full scale)
- `src/features.py` — rapidfuzz + TF-IDF pairwise features, computed in chunks
- `src/matching.py` — LightGBM training, scoring, and F_0.5-driven threshold
  tuning
- `src/scoring.py` — local F_0.5 macro-average implementation matching the
  official formula exactly
- `src/pipeline.py` — end-to-end orchestration; entry point:
  `python -m src.pipeline --processed-dir dataset/processed --train-gt
  dataset/train/train_ground_truth.tsv --test-s1 dataset/test/test_source1.tsv
  --out-dir output`

See `code/business_entity_resolution/README.md` for full setup and run
instructions.

### B. Additional Results

[Fill in with full-scale blocking recall / reduction ratio and any additional
error-analysis charts once the complete run finishes.]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
