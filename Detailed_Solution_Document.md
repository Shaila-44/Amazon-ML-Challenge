# Business Entity Resolution — Detailed Solution Document

**Amazon ML Challenge 2026 · Team Solaris**
Team: Abdul Rahman RSA, Shaila Neelofar K, Pravin Lenin Naidu, Shamanth S
Last updated: 2026-09-25 (challenge window: 25–27 Sept 2026 IST)

This document is the expanded internal write-up behind the required
[`Documentation_template.md`](Documentation_template.md). That file is the
concise, submission-ready version; this one goes into the reasoning, the
data evidence, the rejected alternatives, and the current run status, for
the team's own reference and for anyone auditing the approach.

---

## 1. The Problem

We are given business records from **three independent data sources** that
describe the same underlying set of real-world businesses, but with no
shared identifier. Source 1 is a deduplicated reference list; Source 2 and
Source 3 are noisy, overlapping, partially-redundant listings. For every
Source 1 entity we must find **all** matching records in Source 2 and
Source 3 — zero, one, or many.

Each record has only three usable fields: `business_name`,
`business_address`, `country`. Both name and address are corrupted by
realistic noise: abbreviations and legal-suffix inconsistencies (`Pvt` vs
`Private`, `Corp` vs `Corporation`), punctuation (`&` vs `and`), word
reordering, typos, transliteration, missing address components (no PIN
code/state), and landmark-based addresses ("Near SBI ATM"). Training data
covers **US** and **India**; the **test** set adds **France**, unseen in
training, so nothing in the pipeline may hard-code a fixed country set.

**Scale** is a first-order constraint, not an afterthought:

| Split | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,617 | 5,285,604 |
| Test | 1,732,545 | 4,887,274 | 5,082,317 |

A naive all-pairs comparison is ~10¹³ pairs for train alone — infeasible.
This is why the challenge itself flags **blocking/candidate-generation
recall as the ceiling on final score**, and why we treated it as the
central design problem rather than a preprocessing detail.

**Evaluation** is macro-averaged **F₀.₅** (β = 0.5) per Source 1 entity,
which weights precision **2×** over recall — a false merge is punished
much harder than a missed match. Singletons (entities with no true match)
score 1.0 for a correct empty prediction and 0.0 for any false merge, so
correctly saying "no match" is worth real credit, not a shortcut around
the problem.

**Hard constraints from the problem statement:**
- No external data/API lookups (disqualification if detected) — the
  pipeline uses only the provided training data.
- Final model must be **MIT/Apache-2.0 licensed and ≤ 8B parameters**.
- Output format is strict: exactly one row per Source 1 test entity in
  `matching_results.tsv`, no duplicate/self/out-of-set IDs; a
  `candidate_pairs.tsv` showing the exact candidate set fed to the model.

---

## 2. Approach at a Glance

Classical (non-neural) pipeline, in five stages:

```
raw TSV  →  normalize + transliterate  →  DuckDB token-blocking
         (name AND address, country-hard-filtered)
                     ↓
          candidate pairs (≤ top-50 per S1 entity)
                     ↓
   rapidfuzz + TF-IDF pairwise features (12 features)
                     ↓
        LightGBM binary classifier (match / no-match)
                     ↓
   threshold tuned directly against macro F_0.5  →  matching_results.tsv
```

We deliberately did **not** reach for a neural/embedding approach. The
reasoning is in §4 below — it follows directly from an EDA finding, not
from a default preference for simpler models.

---

## 3. What the Data Actually Showed (EDA)

We ran EDA against the **full** provided files in DuckDB (not a sample) to
ground every design decision in real numbers rather than assumption:

- **Ground-truth exclusivity is clean:** 0 Source 2/3 records match more
  than one Source 1 entity, and **0 cross-country true matches** anywhere
  in the training ground truth. This turns "country must match" into a
  zero-cost, zero-recall-risk hard filter for blocking — confirmed from
  data, not assumed from the problem description.
- **Singleton rate:** 5.59% of Source 1 entities have no true match at
  all; median 3 matches per matched entity.
- **Name alone is not discriminative enough:** 38.3% of Source 1 entities
  share an *exact* business name with another Source 1 entity. Name must
  be combined with address to disambiguate.
- **Name tokens can vanish entirely:** 15.8% of true matched pairs share
  **zero** name tokens even after normalization.
- **Train orphan rate** (Source 2/3 records with no true Source 1 match)
  is ~26% for both sources — the model needs to be good at saying "no
  match," not just at ranking candidates.
- **The decisive finding — script transliteration:** India business names
  appear directly in **Devanagari, Tamil, and Gujarati** script in the raw
  data (confirmed by inspecting real samples, not assumed to be Hindi-only).
  For those pairs:
  - Name similarity after transliteration: **~48** average fuzzy ratio —
    barely above the **~38** noise floor measured for unrelated name pairs.
  - Address similarity for the **same true-matched pairs**: **~96** average
    `token_set_ratio` — address stays in Latin script/numerals even when
    the business name doesn't.

This last point is the crux of the whole design: **address is not a
fallback signal for when name fails — it is an independent, first-class
channel that keeps working exactly where name breaks down.** Every
downstream decision (blocking on two channels, keeping address features
even though they duplicate the name feature set, not leaning on a
multilingual embedding model) traces back to this one measurement.

---

## 4. Model Chosen — and Why

### 4.1 What we chose

- **Candidate generation:** DuckDB SQL token-blocking, over **normalized
  name tokens and normalized address tokens together** (unioned, not
  name-first-with-address-fallback), under a hard country filter.
- **Matching model:** **LightGBM** (gradient-boosted decision trees),
  trained from scratch on 12 hand-engineered pairwise features, with the
  classification threshold **tuned directly against the real macro F₀.₅
  formula** rather than a generic metric.

### 4.2 Why LightGBM, and why not a neural/embedding model

Given the transliteration finding above, the obvious "modern" answer is a
multilingual sentence embedding model (e.g. a LaBSE/multilingual-E5-style
encoder) to get name similarity across scripts for free. We considered and
rejected this, for concrete reasons rather than convenience:

1. **The core failure mode isn't actually a name-embedding problem.**
   Address — not name — is what stays reliable across scripts, and address
   is already plain Latin text/numerals. A rule-based transliteration step
   (`indic_transliteration`, MIT-licensed, covering Devanagari, Tamil,
   Gujarati, Bengali, Kannada, Malayalam, Telugu, Gurmukhi, Oriya) plus
   classical string similarity closes most of the remaining gap on the name
   side without needing a learned multilingual representation at all.
2. **License risk.** The challenge requires the final model to be
   MIT/Apache-2.0 and ≤8B parameters. Most strong multilingual embedding
   checkpoints carry their own (sometimes ambiguous, sometimes
   non-commercial) license terms as *pretrained weights*, which is a murkier
   compliance question than "we trained a small GBDT from scratch on our
   own features." LightGBM itself is MIT-licensed, and because we train it
   from scratch on engineered features rather than redistributing anyone's
   weights, there is no pretrained-model license question to resolve at
   all.
3. **Infrastructure fit.** At 2.2M×10.3M scale, running a transformer
   encoder over every candidate pair (tens of millions of pairs) needs GPU
   throughput we didn't want to make a hard dependency of a pipeline that
   otherwise runs on CPU (DuckDB + LightGBM), especially under a 3-day
   challenge window.
4. **F₀.₅ rewards calibrated, tunable precision — which a GBDT on
   interpretable features gives us directly.** We can sweep the decision
   threshold against the *exact* evaluation formula (including singleton
   handling) and land on a threshold that is legible and re-tunable
   (0.95 — see §6), rather than trusting an embedding-similarity cutoff
   with no analogous calibration step.

This is a considered trade-off, not an avoidance of "harder" methods: we
picked the model whose failure modes we could measure and whose license
posture is unambiguous, once the EDA showed *which* channel actually needed
to carry the multilingual burden.

### 4.3 Feature set (12 features per candidate pair)

| Group | Features |
|---|---|
| Name | Levenshtein ratio, token-sort ratio, token-set ratio, partial ratio (`rapidfuzz`), Jaccard token overlap, TF-IDF cosine over character 2–4-grams |
| Address | ratio, token-set ratio, partial ratio, Jaccard token overlap |
| Structural | `shared_tokens` (carried from blocking), `name_len_diff` |

Notes on choices made here:
- **Character n-gram TF-IDF**, not word-level, for name — robust to typos
  and to transliteration noise where whole tokens don't line up but
  substrings do.
- **Address TF-IDF was tried and dropped** — it did not add signal beyond
  the ratio-based address features, so it was left out to keep the feature
  set lean (fewer features → less overfitting risk on a bounded training
  sample, per §5.3).
- **Country is deliberately not a feature** — blocking already hard-filters
  to same-country candidates (0 cross-country true matches), so it would be
  a constant column carrying zero signal.

---

## 5. Candidate Generation (Blocking) — Detail

Implemented in [`src/blocking.py`](code/business_entity_resolution/src/blocking.py)
as DuckDB SQL, not a Python loop, because a Python-side comparison loop
cannot approach DuckDB's join throughput at this row count.

1. **Tokenize** normalized name and address into token sets per record
   (done once, upstream, in `preprocess.py`).
2. **Hard country filter.** Zero-cost per the EDA finding — processing is
   also *partitioned* by country (one query per country, unioned at the
   end) because a single combined query across all countries hit DuckDB's
   memory limit at only ~1/6 scale (350k S1 entities); partitioning bounds
   peak memory to the largest single country's data instead of the sum of
   all countries.
3. **Drop overly common tokens** before the join — both a percentage cutoff
   (2% of the candidate-pool group) *and* an absolute cap (1,500
   occurrences), because at 10M+ records, 2% alone is still ~200,000
   occurrences for one token — a percentage-only cutoff stops being
   meaningful once the corpus is large enough.
4. **Restrict each Source 1 entity to its 8 rarest tokens** (by candidate-
   pool document frequency) before the join. This is the single change that
   makes the join tractable at full scale: without it, join cost scales
   with shared vocabulary size; with it, cost scales roughly linearly with
   entity count.
5. **Rank and cap** candidates per Source 1 entity by shared-token count,
   keep the **top 50**.

**Recall validation** (measured against real ground-truth pairs on a
held-out sample built from actual training data, not a synthetic proxy):

| top_k | Recall |
|---|---|
| 50 | 96.75% |
| 100 | 97.3% |
| 200 | 98.1% |

The marginal recall gain from 50→100→200 did not justify a 2×/4× growth in
candidate volume (and downstream feature-computation cost) at full scale,
so **top_k=50** was kept as the production setting.

---

## 6. Matching Model — Detail

Implemented in [`src/matching.py`](code/business_entity_resolution/src/matching.py).

- **Training sample construction:** the ~110M full train candidate set is
  never featurized wholesale. Labels are attached via a SQL join against
  ground truth, then we keep **all positives + a capped random sample of
  negatives** (4,000,000) *before* running the expensive per-pair
  rapidfuzz/TF-IDF feature computation — this is what bounds featurization
  to a few million rows instead of ~110M.
- **Split:** train/validation split **by Source 1 entity** (not by row) to
  avoid leakage across the split.
- **Model:** `LGBMClassifier`, 300 trees, 31 leaves, learning rate 0.05,
  `is_unbalance=True` (candidate pairs are heavily skewed toward
  non-matches after blocking).
- **Threshold tuning:** thresholds swept over `[0.1, 0.95]` and scored with
  the exact macro F₀.₅ implementation in
  [`src/scoring.py`](code/business_entity_resolution/src/scoring.py)
  (including singleton handling — entities with no candidate above
  threshold correctly count as a predicted-empty row). The tuned threshold
  lands at **0.95**, consistent with F₀.₅'s 2× precision weighting: the
  model would rather miss a true match than risk a false merge.

---

## 7. What We Have Done So Far

**Pipeline built and validated end-to-end** on real samples drawn directly
from the provided data (up to ~350k Source 1 entities, with genuine
embedded ground-truth pairs, not synthetic labels):

- Blocking recall: **96.75%** at top_k=50 (measured, not assumed).
- Held-out validation **macro F₀.₅ = 0.9655–0.9743** across runs.
- Output format confirmed **PASS** by the organizers' own
  `utils/validate_submission.py`, including `--check-ids`.
- Code is organized as a self-contained, reproducible package under
  [`code/business_entity_resolution/`](code/business_entity_resolution)
  with `src/` (all pipeline logic), notebooks mirroring each stage for
  exploratory work, a `README.md` with exact run instructions, and a
  pinned `requirements.txt`.
- Submission packaging is scripted
  ([`package_submission.sh`](package_submission.sh)) to assemble the
  exact zip structure the challenge requires
  (`output/`, `code/business_entity_resolution/`,
  `Documentation_template.md`) and refuses to run if any required file is
  missing or empty.
- Engineering hardening done in response to real full-scale failures, not
  anticipated in advance:
  - DuckDB memory limit is now set to ~75% of physical RAM **at runtime**
    (`os.sysconf`), rather than hand-edited per machine — a mismatch here
    is exactly what caused one full-scale attempt to OOM at 6GB after the
    pipeline had been validated separately at 48GB on the same instance.
  - The DuckDB connection is **closed and reopened between heavy stages**
    (train blocking → sampling → test blocking → aggregation). A single
    long-lived connection accumulated buffer-pool state across stages and
    caused a hard C++-level OOM crash during `candidate_pairs.tsv`
    aggregation on the real full-scale run — an operation that is cheap on
    its own but was hit right after a memory-fragmenting join. Reconnecting
    costs a few seconds and guarantees a clean slate per stage.
  - Model + tuned threshold are **checkpointed to disk** after the
    train-side stages, with a `--resume` flag, because train-side work
    (blocking, sampling, featurization, training, tuning) takes the better
    part of an hour at full scale and previously only existed in memory —
    a crash anywhere in the (longer) test-side stages meant redoing all of
    it.
- **Infrastructure for the full-scale run:** the 8GB local development
  machine is enough to validate correctness at meaningful sample scale but
  not to run the full 2.2M/5.0M/5.3M (train) + 1.7M/4.9M/5.1M (test) row
  set. Preprocessed Parquet (~2.5GB across 7 files) has been transferred to
  an EC2 `r5.2xlarge` (8 vCPU / 64GB RAM) via a retrying rsync script
  ([`.transfer_retry.sh`](.transfer_retry.sh)) — transfer completed
  successfully.

## 8. What's Left / Next Steps

1. **Complete and monitor the full-scale run on EC2.** This is the
   immediate next action — kick off `src.pipeline` against the transferred
   `dataset/processed/` on the `r5.2xlarge` instance, watching for the
   OOM/memory-fragmentation failure mode already hardened against in §7,
   and using `--resume` if any stage needs to be restarted after a
   train-side checkpoint exists.
2. **Pull back `output/matching_results.tsv` and `output/candidate_pairs.tsv`**
   from the full-scale run and validate locally with
   `utils/validate_submission.py --check-ids` before any leaderboard
   upload.
3. **Submit to the leaderboard** (up to 5 submissions/day allowed across
   the 25–27 Sept window) and record the resulting public score alongside
   the local held-out validation score, to sanity-check that sample-scale
   validation (F₀.₅ ≈ 0.97) generalizes to the full test set — including
   the unseen France country slice, which is the main generalization risk
   since the pipeline's country-handling was designed to be open-set but
   has not yet been *measured* against a real out-of-training-distribution
   country.
4. **Backfill the "[fill in]" placeholders in `Documentation_template.md`**
   (§3 candidate pair count, §5 full-scale F₀.₅, Appendix B blocking
   recall/reduction ratio) with numbers from the completed full-scale run,
   and record the actual submission date.
5. **Error analysis on real full-scale output** — the current false
   positive/negative patterns (§"Results & Error Analysis" in the
   template) are drawn from sample-scale validation; re-check them against
   the full run, particularly for France, where no training examples exist
   to have shaped the threshold or feature distributions.
6. **Package and submit the final zip** via `package_submission.sh` once
   `output/` is populated by the full-scale run and the documentation is
   updated, then keep a version-history record of each leaderboard
   submission per the challenge's own instructions.
7. **Open compliance question to raise via the challenge's Google Form (if
   time allows):** confirm that "LightGBM trained from scratch on
   engineered features" unambiguously satisfies the "final model must be
   MIT/Apache-2.0, ≤8B parameters" constraint, since that constraint reads
   as aimed at pretrained/foundation models and our classifier has no
   pretrained-weights component at all — worth a explicit confirmation
   rather than an assumption, given disqualification is on the table for
   any rules ambiguity.

---

## Appendix: Code Map

| File | Role |
|---|---|
| [`src/normalize.py`](code/business_entity_resolution/src/normalize.py) | Legal-suffix stripping (US/India/France), diacritic folding, script-general transliteration |
| [`src/preprocess.py`](code/business_entity_resolution/src/preprocess.py) | Streams each raw source TSV once into normalized Parquet (bounded memory regardless of file size) |
| [`src/blocking.py`](code/business_entity_resolution/src/blocking.py) | DuckDB SQL token-blocking, country-partitioned, doc-frequency-capped |
| [`src/features.py`](code/business_entity_resolution/src/features.py) | rapidfuzz + TF-IDF pairwise features, computed in chunks |
| [`src/matching.py`](code/business_entity_resolution/src/matching.py) | LightGBM training, scoring, F₀.₅-driven threshold tuning |
| [`src/scoring.py`](code/business_entity_resolution/src/scoring.py) | Local macro F₀.₅ implementation matching the official formula exactly |
| [`src/pipeline.py`](code/business_entity_resolution/src/pipeline.py) | End-to-end orchestration, checkpointing, resume support |
| [`utils/validate_submission.py`](utils/validate_submission.py) | Organizer-provided output-format validator |
| [`package_submission.sh`](package_submission.sh) | Assembles the final submission zip |
