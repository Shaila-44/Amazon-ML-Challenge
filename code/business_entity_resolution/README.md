# Business Entity Resolution — Amazon ML Challenge 2026

Matches Source 2 / Source 3 business records to Source 1 reference entities
(name + address entity resolution, precision-weighted F_0.5 scoring).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Place challenge files under `dataset/` (not committed — see `.gitignore`):

```
dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
```

## Run end-to-end

Preprocessing is a separate, one-time step (streams each raw file once into
normalized Parquet under `dataset/processed/` — decouples the per-row Python
normalization cost from the set-based DuckDB blocking joins that consume it):

```bash
for split in train test; do
  for src in source1 source2 source3; do
    python3 -m src.preprocess \
      --in  dataset/$split/${split}_${src}.tsv \
      --out dataset/processed/${split}_${src}.parquet
  done
done

python3 -m src.pipeline \
  --processed-dir dataset/processed \
  --train-gt dataset/train/train_ground_truth.tsv \
  --test-s1 dataset/test/test_source1.tsv \
  --out-dir output

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

## Pipeline stages (`src/`)

1. `normalize.py` — legal-suffix stripping (US/India/France), diacritic
   folding, and script-general transliteration (Devanagari/Tamil/Gujarati/
   etc. via `indic_transliteration`, MIT license) so non-Latin business names
   get a Latin approximation to compare against
2. `preprocess.py` — streams a raw source TSV once, normalizes, writes
   Parquet with token lists (bounded memory regardless of file size)
3. `blocking.py` — DuckDB SQL token-blocking on **both** normalized name and
   address tokens (not name alone — address stays in Latin script/numerals
   even when the name is transliterated), country-filtered and
   country-partitioned, document-frequency and per-entity token-count capped
   so join cost stays tractable at full scale, top-k capped per Source-1 entity
4. `features.py` — rapidfuzz + char-n-gram TF-IDF cosine similarity features,
   computed in chunks so peak memory stays bounded at full scale
5. `matching.py` — LightGBM classifier (MIT-licensed, trained from scratch —
   no pretrained-model license question) + threshold tuned against the real
   F_0.5 formula, not a generic metric
6. `scoring.py` — local F_0.5 macro-average, matching the official formula
7. `pipeline.py` — orchestrates 1-5 into `output/matching_results.tsv` +
   `output/candidate_pairs.tsv`, streaming throughout (candidates and
   features are never fully materialized in memory, since the real dataset
   produces tens of millions of candidate pairs)

Notebooks under `notebooks/` mirror these stages for exploratory work;
`src/` holds the reusable, reproducible logic the notebooks call into.

## Status

Pipeline validated end-to-end on real samples drawn from the actual dataset
(up to ~350k Source-1 entities, with genuine embedded ground-truth pairs):
held-out validation **F_0.5 = 0.9655-0.9743** across runs, blocking recall
**96.75%** at top_k=50, output format confirmed **PASS** by the organizers'
own `utils/validate_submission.py` including `--check-ids`. Full-scale run
(2.2M/5.0M/5.3M train, 1.7M/4.9M/5.1M test rows) is in progress on an EC2
`r5.2xlarge` (8 vCPU / 64GB RAM) — the local development machine's 8GB RAM
is enough to validate the pipeline's correctness at meaningful sample scale,
but not to run the full candidate/feature set.
