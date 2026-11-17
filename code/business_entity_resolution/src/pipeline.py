"""End-to-end orchestration: preprocessed data -> blocking -> features ->
matching -> output files.

Validation design (each point fixes a measured failure of the first
full-scale run, whose local validation said ~0.98 while the leaderboard said
0.53):
  - A fixed entity-level holdout is chosen first (deterministic hash) and is
    excluded from everything fitted: the classifier's training sample and the
    TF-IDF vocabulary sample.
  - Negatives are subsampled for TRAINING only. Validation keeps every
    candidate of the held-out entities; tuning on subsampled negatives shifts
    the class prior and picked a threshold (0.65) far too permissive.
  - Validation F_0.5 is scored against the COMPLETE ground truth, so true
    matches that blocking never surfaced count as misses.

Every run writes an audit trail to --artifacts-dir: run configuration and
seeds, candidate sanity checks, full-scale blocking recall by country x
source x K, training-set composition, feature distributions and
importances, feature/candidate alignment checks, the full validation
threshold sweep, a K decomposition, and a test-side prediction profile.

Memory / runtime model (train ~2.2M S1, test ~1.7M S1, ~5M S2 and S3 each):
  - Candidate generation stays inside DuckDB and is written straight to
    Parquet, with a per-entity rank so smaller K can be cut later.
  - Feature computation runs in parallel worker processes over hash-sharded
    Parquet (`parallel_features`); test pairs are featurized and scored once
    and the scores kept on disk, so re-thresholding never re-featurizes.

Run via:
    python -m src.pipeline \
        --processed-dir dataset/processed \
        --train-gt dataset/train/train_ground_truth.tsv \
        --test-s1 dataset/test/test_source1.tsv \
        --out-dir output --artifacts-dir artifacts
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import inspect
import json
import os
import pickle
import shutil
import time

import duckdb
import numpy as np
import pandas as pd

from . import blocking, features, matching
from .features import FEATURE_COLUMNS
from .parallel_features import featurize_parallel, verify_alignment

VAL_SEED = "solaris-val-v2"
NEG_SEED = "solaris-neg-v2"
TFIDF_SEED = "solaris-tfidf-v2"
K_REPORT = (50, 100, 200)
_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)


def _dump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _default_memory_limit_gb() -> int:
    """~75% of physical RAM, so the same command works on an 8GB laptop or a
    64GB EC2 box (a hardcoded 6GB once OOM-crashed a full run on the latter)."""
    total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return max(2, int(total_bytes / (1024**3) * 0.75))


def _connect(limit_gb: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{limit_gb}GB'")
    return con


def _load_gt(con: duckdb.DuckDBPyConnection, train_gt_path: str) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE gt_pairs AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
        FROM read_csv('{train_gt_path}', delim='\t', header=true, quote='"', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    """)


def _check_candidates(
    con: duckdb.DuckDBPyConnection, cands: str, s1_parquet: str, top_k: int, expect_s1: int | None, name: str
) -> dict:
    """Fail fast before any feature computation if the candidate set is not
    what K should produce (duplicates, rank gaps, over-cap entities, orphan
    ids, wrong S1 universe)."""
    n_s1 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{s1_parquet}')").fetchone()[0]
    ents, pairs, max_n, dup_ents, gap_ents, median_n, frac_cap = con.execute(f"""
        WITH per AS (
            SELECT source1_entity_id, COUNT(*) AS n, COUNT(DISTINCT candidate_entity_id) AS nd,
                   MIN(candidate_rank) AS mn, MAX(candidate_rank) AS mx
            FROM read_parquet('{cands}') GROUP BY 1)
        SELECT COUNT(*), SUM(n), MAX(n), SUM((n != nd)::INT), SUM((mn != 1 OR mx != n)::INT),
               quantile_cont(n, 0.5), AVG((n = {top_k})::INT)
        FROM per
    """).fetchone()
    orphans = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{cands}') c
        LEFT JOIN read_parquet('{s1_parquet}') s ON s.entity_id = c.source1_entity_id
        WHERE s.entity_id IS NULL
    """).fetchone()[0]
    by_src = dict(con.execute(f"""SELECT left(candidate_entity_id, 2), COUNT(*) FROM read_parquet('{cands}')
                                  GROUP BY 1 ORDER BY 1""").fetchall())
    stats = dict(name=name, s1_entities=int(n_s1), expected_s1_entities=expect_s1, top_k=top_k,
                 entities_with_candidates=int(ents), pairs=int(pairs), max_candidates_per_entity=int(max_n),
                 median_candidates_per_entity=float(median_n), frac_entities_at_cap=float(frac_cap),
                 entities_with_duplicate_pairs=int(dup_ents), entities_with_rank_gaps=int(gap_ents),
                 orphan_candidate_rows=int(orphans), candidates_by_source=by_src)
    problems = []
    if expect_s1 is not None and n_s1 != expect_s1:
        problems.append(f"S1 entity count {n_s1} != expected {expect_s1}")
    if max_n > top_k or pairs > top_k * n_s1:
        problems.append("candidate count exceeds K")
    if dup_ents or gap_ents or orphans:
        problems.append("duplicate pairs / rank gaps / orphan ids")
    stats["passed"] = not problems
    if problems:
        raise RuntimeError(f"candidate sanity check FAILED ({name}): {problems} {stats}")
    return stats


def _blocking_recall(con: duckdb.DuckDBPyConnection, cands: str, s1_parquet: str, ks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair recall by country x source at each K (via candidate_rank), and
    per-K entity-level loss: share of entities whose true matches were ALL
    blocked away, and the perfect-classifier F_0.5 ceiling."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE truth_ranked AS
        SELECT g.source1_entity_id, left(g.matched_entity_id, 2) AS src, s.country, c.candidate_rank
        FROM gt_pairs g
        JOIN read_parquet('{s1_parquet}') s ON s.entity_id = g.source1_entity_id
        LEFT JOIN read_parquet('{cands}') c
          ON c.source1_entity_id = g.source1_entity_id AND c.candidate_entity_id = g.matched_entity_id
    """)
    cols = ", ".join(f"AVG(COALESCE(candidate_rank <= {k}, false)::INT) AS recall_k{k}" for k in ks)
    pair = con.sql(f"""
        SELECT COALESCE(country, 'ALL') AS country, COALESCE(src, 'ALL') AS src, COUNT(*) AS n_true, {cols}
        FROM truth_ranked GROUP BY GROUPING SETS ((country, src), (country), (src), ())
        ORDER BY 1, 2
    """).df()
    rows = []
    for k in ks:
        r = con.execute(f"""
            WITH e AS (
                SELECT s.entity_id, COUNT(t.source1_entity_id) AS n_truth,
                       SUM(COALESCE(t.candidate_rank <= {k}, false)::INT) AS found
                FROM read_parquet('{s1_parquet}') s
                LEFT JOIN truth_ranked t ON t.source1_entity_id = s.entity_id
                GROUP BY 1)
            SELECT AVG(CASE WHEN n_truth > 0 AND found = 0 THEN 1 ELSE 0 END),
                   AVG(CASE WHEN n_truth = 0 THEN 1.0 WHEN found = 0 THEN 0.0
                            ELSE 1.25 * (found * 1.0 / n_truth) / (0.25 + found * 1.0 / n_truth) END)
            FROM e
        """).fetchone()
        rows.append(dict(K=k, frac_entities_all_true_matches_missed=r[0], perfect_classifier_ceiling=r[1]))
    return pair, pd.DataFrame(rows)


def _ceiling(found: pd.Series, n_truth: pd.Series) -> float:
    r = found / n_truth.where(n_truth > 0, 1)
    c = pd.Series(1.0, index=n_truth.index).where(n_truth == 0, (1.25 * r / (0.25 + r)).where(r > 0, 0.0))
    return float(c.mean())


def _write_candidate_pairs(
    con: duckdb.DuckDBPyConnection, candidates_parquet: str, required_ids_path: str, out_path: str,
    work_dir: str, n_shards: int = 16,
) -> None:
    """candidate_pairs.tsv without a single huge GROUP BY: a one-shot
    `string_agg` over ~85-110M rows OOM-crashed twice at full scale. Aggregate
    one hash shard at a time into Parquet, then do one ordered join against
    the required-id list (1.7M rows)."""
    agg_dir = os.path.join(work_dir, "_candidate_agg_tmp")
    shutil.rmtree(agg_dir, ignore_errors=True)
    os.makedirs(agg_dir)
    for s in range(n_shards):
        con.execute(f"""
            COPY (
                SELECT source1_entity_id,
                       string_agg(candidate_entity_id, ',' ORDER BY candidate_entity_id) AS candidate_entity_ids
                FROM read_parquet('{candidates_parquet}')
                WHERE hash(source1_entity_id) % {n_shards} = {s}
                GROUP BY 1
            ) TO '{agg_dir}/part-{s:02d}.parquet' (FORMAT PARQUET)
        """)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT r.entity_id AS source1_entity_id, a.candidate_entity_ids
            FROM read_csv('{required_ids_path}', delim='\t', header=true, quote='"', all_varchar=true) r
            LEFT JOIN read_parquet('{agg_dir}/*.parquet') a ON a.source1_entity_id = r.entity_id
            ORDER BY r.entity_id
        ) TO '{out_path}' (DELIMITER '\t', HEADER true, NULL '')
    """)
    shutil.rmtree(agg_dir)


def _write_matching_results(
    con: duckdb.DuckDBPyConnection, scored_dir: str, threshold: float, required_ids_path: str, out_path: str
) -> None:
    """Every required Source-1 entity gets exactly one row; entities with no
    pair above threshold get an empty (NULL -> '') match list."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT r.entity_id AS source1_entity_id, m.matched_entity_ids
            FROM read_csv('{required_ids_path}', delim='\t', header=true, quote='"', all_varchar=true) r
            LEFT JOIN (
                SELECT source1_entity_id,
                       string_agg(candidate_entity_id, ',' ORDER BY candidate_entity_id) AS matched_entity_ids
                FROM read_parquet('{scored_dir}/*.parquet')
                WHERE score >= {threshold!r}
                GROUP BY 1
            ) m ON m.source1_entity_id = r.entity_id
            ORDER BY r.entity_id
        ) TO '{out_path}' (DELIMITER '\t', HEADER true, NULL '')
    """)


def _test_profile(con: duckdb.DuckDBPyConnection, scored_dir: str, n_s1: int, thresholds) -> pd.DataFrame:
    """Prediction profile of the test set at each threshold (no labels):
    what a submission at that threshold would contain."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hi AS
                    SELECT source1_entity_id, candidate_entity_id, score FROM read_parquet('{scored_dir}/*.parquet')
                    WHERE score >= {float(min(thresholds))!r}""")
    rows = []
    for t in map(float, thresholds):
        pairs, ents, s2, s3 = con.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT source1_entity_id),
                   SUM(starts_with(candidate_entity_id, 'S2')::INT), SUM(starts_with(candidate_entity_id, 'S3')::INT)
            FROM hi WHERE score >= {t!r}""").fetchone()
        dup = con.execute(f"""SELECT COUNT(*) FROM (SELECT candidate_entity_id FROM hi WHERE score >= {t!r}
                              GROUP BY 1 HAVING COUNT(*) > 1)""").fetchone()[0]
        rows.append(dict(threshold=t, pairs=pairs, frac_entities_nonempty=ents / n_s1,
                         mean_pred_per_entity_if_any=pairs / max(ents, 1),
                         records_claimed_by_multiple_s1=dup, n_pred_S2=s2 or 0, n_pred_S3=s3 or 0))
    return pd.DataFrame(rows)


def _k_decomposition(va: pd.DataFrame, score_col: str, n_truth: pd.Series, ks, model_name: str) -> list[dict]:
    rows = []
    for k in ks:
        sub = va[va["candidate_rank"] <= k]
        pairs = sub[["source1_entity_id", "candidate_entity_id", "label"]].assign(score=sub[score_col])
        t, f, _ = matching.tune_threshold_full(pairs, n_truth)
        found = sub[sub["label"]].groupby("source1_entity_id").size().reindex(n_truth.index, fill_value=0)
        f096 = matching.per_entity_f05(pairs[pairs["score"] >= 0.96], n_truth).mean()
        rows.append(dict(model=model_name, K=k, pair_recall=float(sub["label"].sum() / n_truth.sum()),
                         perfect_classifier_ceiling=_ceiling(found, n_truth),
                         f05_at_0_96=float(f096), tuned_threshold=t, f05_at_tuned=f))
    return rows


def _train_and_tune(con, art, train_s, train_gt_path, vec_path, top_k, max_train_negatives,
                    limit, workers, expect_train_s1, baseline_model, baseline_vec):
    """block -> sanity checks -> blocking recall -> sample training pairs ->
    featurize (+ alignment checks) -> train -> tune threshold on the
    all-candidate holdout -> reports. Returns (model_path, threshold, con)."""
    train_s1 = train_s[0]
    cands = f"{art}/train_candidates.parquet"
    if not os.path.exists(cands):
        blocking_stats = []
        blocking.generate_candidates(con, *train_s, top_k=top_k, stats=blocking_stats).write_parquet(cands)
        _dump(blocking_stats, f"{art}/blocking_stats_train.json")
        con.close()
        con = _connect(limit)
        _load_gt(con, train_gt_path)
        con.execute(f"CREATE OR REPLACE TABLE val_ents AS SELECT * FROM read_parquet('{art}/val_entities.parquet')")
    checks = _check_candidates(con, cands, train_s1, top_k, expect_train_s1, "train")
    _dump(checks, f"{art}/candidate_checks_train.json")
    _log(f"train candidate checks passed: {json.dumps(checks)}")

    ks = [k for k in K_REPORT if k <= top_k]
    pair_recall, entity_loss = _blocking_recall(con, cands, train_s1, ks)
    pair_recall.to_csv(f"{art}/blocking_recall_train.csv", index=False)
    entity_loss.to_csv(f"{art}/blocking_entity_loss_train.csv", index=False)
    _log("full-scale train blocking recall:\n" + pair_recall.to_string(index=False) + "\n" + entity_loss.to_string(index=False))

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW labeled AS
        SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens, c.candidate_rank,
               (g.matched_entity_id IS NOT NULL) AS label
        FROM read_parquet('{cands}') c
        LEFT JOIN gt_pairs g
          ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.candidate_entity_id
    """)
    sample = f"{art}/train_sample.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM labeled WHERE label AND source1_entity_id NOT IN (SELECT entity_id FROM val_ents)
            UNION ALL
            SELECT * FROM (
                SELECT * FROM labeled
                WHERE NOT label AND source1_entity_id NOT IN (SELECT entity_id FROM val_ents)
                ORDER BY hash(source1_entity_id || '|' || candidate_entity_id || '{NEG_SEED}')
                LIMIT {max_train_negatives})
        ) TO '{sample}' (FORMAT PARQUET)
    """)
    val_cands = f"{art}/val_candidates.parquet"
    con.execute(f"""COPY (SELECT l.* FROM labeled l JOIN val_ents v ON v.entity_id = l.source1_entity_id)
                    TO '{val_cands}' (FORMAT PARQUET)""")
    leak = con.execute(f"""SELECT COUNT(*) FROM read_parquet('{sample}')
                           WHERE source1_entity_id IN (SELECT entity_id FROM val_ents)""").fetchone()[0]
    if leak:
        raise RuntimeError(f"{leak} training rows belong to holdout entities")
    n_pos, n_neg = (con.execute(f"SELECT COUNT(*) FROM read_parquet('{sample}') WHERE label = {b}").fetchone()[0]
                    for b in ("true", "false"))
    n_val_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{val_cands}')").fetchone()[0]
    _log(f"train sample: {n_pos} positives, {n_neg} negatives; holdout: {n_val_pairs} candidate pairs (all kept)")

    alignment = {}
    featurize_parallel(con, f"read_parquet('{sample}')", *train_s, f"{art}/train_features", vec_path, workers=workers)
    alignment["train"] = verify_alignment(con, f"read_parquet('{sample}')", f"{art}/train_features", *train_s, vec_path)
    featurize_parallel(con, f"read_parquet('{val_cands}')", *train_s, f"{art}/val_features", vec_path, workers=workers)
    alignment["val"] = verify_alignment(con, f"read_parquet('{val_cands}')", f"{art}/val_features", *train_s, vec_path)
    _dump(alignment, f"{art}/alignment_checks.json")
    _log(f"train + holdout features done, alignment verified: {json.dumps(alignment)}")

    tr = con.sql(f"""
        SELECT f.*, s.label FROM read_parquet('{art}/train_features/*.parquet') f
        JOIN read_parquet('{sample}') s USING (source1_entity_id, candidate_entity_id)
    """).df()
    model = matching.train_matcher(tr, tr["label"].astype(int))
    model_path = f"{art}/model.txt"
    model.booster_.save_model(model_path)
    booster = model.booster_
    _dump(dict(
        positives=int(n_pos), sampled_negatives=int(n_neg), negatives_per_positive=n_neg / max(n_pos, 1),
        negative_sampling=f"deterministic hash order, seed '{NEG_SEED}', cap {max_train_negatives}",
        model_params=model.get_params(),
        feature_importance_gain=dict(zip(FEATURE_COLUMNS, booster.feature_importance("gain").tolist())),
        feature_importance_split=dict(zip(FEATURE_COLUMNS, booster.feature_importance("split").tolist())),
    ), f"{art}/training_report.json")
    tr.groupby("label")[FEATURE_COLUMNS].describe().T.to_csv(f"{art}/feature_distributions_train.csv")
    del tr
    _log("model trained")

    va = con.sql(f"""
        SELECT f.*, v.label, v.candidate_rank FROM read_parquet('{art}/val_features/*.parquet') f
        JOIN read_parquet('{val_cands}') v USING (source1_entity_id, candidate_entity_id)
    """).df()
    va["score"] = matching.score_pairs(model, va)
    va.groupby("label")[FEATURE_COLUMNS].describe().T.to_csv(f"{art}/feature_distributions_val.csv")
    n_truth = con.sql("""
        SELECT v.entity_id, COUNT(g.matched_entity_id) AS n
        FROM val_ents v LEFT JOIN gt_pairs g ON g.source1_entity_id = v.entity_id
        GROUP BY 1
    """).df().set_index("entity_id")["n"]
    threshold, f05, sweep = matching.tune_threshold_full(
        va[["source1_entity_id", "candidate_entity_id", "score", "label"]], n_truth)
    sweep["selected"] = np.isclose(sweep["threshold"], threshold)
    sweep.to_csv(f"{art}/threshold_sweep_val.csv", index=False)

    decomp = _k_decomposition(va, "score", n_truth, ks, "v3_retrained")
    if baseline_model and baseline_vec:
        base_dir = f"{art}/val_features_baseline"
        featurize_parallel(con, f"read_parquet('{val_cands}')", *train_s, base_dir, baseline_vec,
                           model_path=baseline_model, workers=workers)
        base = con.sql(f"SELECT source1_entity_id, candidate_entity_id, score AS score_baseline FROM read_parquet('{base_dir}/*.parquet')").df()
        va = va.merge(base, on=["source1_entity_id", "candidate_entity_id"], how="left")
        decomp += _k_decomposition(va, "score_baseline", n_truth, ks, "v1_model_(old_vec)")
    pd.DataFrame(decomp).to_csv(f"{art}/val_k_decomposition.csv", index=False)

    per_entity = matching.per_entity_f05(va[va["score"] >= threshold], n_truth)
    countries = con.sql("SELECT entity_id, country FROM val_ents").df().set_index("entity_id")["country"]
    found = va[va["label"]].groupby("source1_entity_id").size().reindex(n_truth.index, fill_value=0)
    summary = dict(
        threshold=threshold, holdout_f05=f05,
        holdout_f05_by_country=per_entity.groupby(countries).mean().round(4).to_dict(),
        holdout_blocking_pair_recall=float(va["label"].sum() / n_truth.sum()),
        holdout_perfect_classifier_ceiling=_ceiling(found, n_truth),
        holdout_entities=int(len(n_truth)), holdout_pairs=int(len(va)), top_k=top_k,
        k_decomposition=decomp,
    )
    with open(f"{art}/threshold.txt", "w") as f:
        f.write(repr(threshold))
    va.drop(columns=FEATURE_COLUMNS).to_parquet(f"{art}/val_scored.parquet")
    _dump(summary, f"{art}/validation_summary.json")
    _log(f"VALIDATION (all candidates, complete ground truth): {json.dumps(summary, default=str)}")
    return model_path, threshold, con


def run(
    processed_dir: str,
    train_gt_path: str,
    test_s1_raw_path: str,
    out_dir: str,
    artifacts_dir: str,
    top_k: int = 200,
    max_train_negatives: int = 8_000_000,
    n_val_entities: int = 50_000,
    memory_limit_gb: int | None = None,
    workers: int | None = None,
    resume: bool = False,
    expect_train_s1: int | None = None,
    expect_test_s1: int | None = None,
    baseline_model: str | None = None,
    baseline_vec: str | None = None,
    run_id: str | None = None,
) -> None:
    limit = memory_limit_gb or _default_memory_limit_gb()
    con = _connect(limit)
    art = artifacts_dir
    os.makedirs(art, exist_ok=True)
    train_s = tuple(f"{processed_dir}/train_source{i}.parquet" for i in (1, 2, 3))
    test_s = tuple(f"{processed_dir}/test_source{i}.parquet" for i in (1, 2, 3))
    vec_path, model_path, thr_path = f"{art}/vec.pkl", f"{art}/model.txt", f"{art}/threshold.txt"
    src_dir = os.path.dirname(os.path.abspath(__file__))
    config = dict(
        run_id=run_id, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        processed_dir=processed_dir, train_gt=train_gt_path, test_s1=test_s1_raw_path,
        out_dir=out_dir, artifacts_dir=art, top_k=top_k, max_train_negatives=max_train_negatives,
        n_val_entities=n_val_entities, memory_limit_gb=limit, workers=workers or os.cpu_count(),
        seeds=dict(holdout=VAL_SEED, negatives=NEG_SEED, tfidf_sample=TFIDF_SEED, lightgbm=42),
        blocking_config={k: v.default for k, v in inspect.signature(blocking.generate_candidates).parameters.items()
                         if v.default is not inspect.Parameter.empty and k != "stats"} | dict(
            top_k=top_k, token_selection="rarest tokens that survive the S2/S3 doc-frequency filter",
            ranking="idf_same DESC, idf_all DESC, hash(candidate_entity_id)"),
        baseline_model=baseline_model, baseline_vec=baseline_vec,
        code_md5={os.path.basename(p): _md5(p) for p in sorted(glob.glob(f"{src_dir}/*.py"))},
    )
    _dump(config, f"{art}/run_config.json")
    _log(f"run config: {json.dumps(config, default=str)}")

    _load_gt(con, train_gt_path)
    if resume and all(os.path.exists(p) for p in (vec_path, model_path, thr_path)):
        with open(thr_path) as f:
            threshold = float(f.read())
        _log(f"resuming from checkpoint: threshold={threshold}")
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE val_ents AS
            SELECT entity_id, country FROM read_parquet('{train_s[0]}')
            ORDER BY hash(entity_id || '{VAL_SEED}') LIMIT {n_val_entities}
        """)
        con.execute(f"COPY val_ents TO '{art}/val_entities.parquet' (FORMAT PARQUET)")
        # TF-IDF vocabulary: unsupervised, but holdout S1 names and the S2
        # records matched to holdout entities are still kept out of it.
        names = con.execute(f"""
            SELECT name_norm FROM read_parquet('{train_s[0]}')
            WHERE entity_id NOT IN (SELECT entity_id FROM val_ents)
            ORDER BY hash(entity_id || '{TFIDF_SEED}') LIMIT 300000
        """).df()["name_norm"].tolist()
        names += con.execute(f"""
            SELECT name_norm FROM read_parquet('{train_s[1]}')
            WHERE entity_id NOT IN (SELECT g.matched_entity_id FROM gt_pairs g
                                    JOIN val_ents v ON v.entity_id = g.source1_entity_id)
            ORDER BY hash(entity_id || '{TFIDF_SEED}') LIMIT 300000
        """).df()["name_norm"].tolist()
        vec = features.fit_tfidf(names)
        with open(vec_path, "wb") as f:
            pickle.dump(vec, f)
        _log(f"tfidf fit on {len(names)} names (holdout excluded): vocab size={len(vec.vocabulary_)}")
        model_path, threshold, con = _train_and_tune(
            con, art, train_s, train_gt_path, vec_path, top_k, max_train_negatives,
            limit, workers, expect_train_s1, baseline_model, baseline_vec,
        )

    test_cands = f"{art}/test_candidates.parquet"
    if not (resume and os.path.exists(test_cands)):
        blocking_stats = []
        blocking.generate_candidates(con, *test_s, top_k=top_k, stats=blocking_stats).write_parquet(test_cands)
        _dump(blocking_stats, f"{art}/blocking_stats_test.json")
        con.close()
        con = _connect(limit)
    checks = _check_candidates(con, test_cands, test_s[0], top_k, expect_test_s1, "test")
    _dump(checks, f"{art}/candidate_checks_test.json")
    _log(f"test candidate checks passed: {json.dumps(checks)}")
    n_test = checks["pairs"]

    _write_candidate_pairs(con, test_cands, test_s1_raw_path, f"{out_dir}/candidate_pairs.tsv", art)
    _log(f"wrote {out_dir}/candidate_pairs.tsv")

    scored = f"{art}/test_scored"
    have = (con.execute(f"SELECT COUNT(*) FROM read_parquet('{scored}/*.parquet')").fetchone()[0]
            if resume and os.path.isdir(scored) and os.listdir(scored) else -1)
    if have != n_test:
        n_passes = max(1, -(-n_test // 80_000_000))  # bounds pre-joined temp shards to ~6-7GB per pass
        _log(f"featurizing {n_test} test pairs in {n_passes} pass(es)")
        featurize_parallel(con, f"read_parquet('{test_cands}')", *test_s, scored, vec_path,
                           model_path=model_path, workers=workers, n_passes=n_passes)
    test_alignment = verify_alignment(con, f"read_parquet('{test_cands}')", scored, *test_s, vec_path)
    _dump(test_alignment, f"{art}/alignment_checks_test.json")
    _log(f"test pairs featurized + scored, alignment verified: {json.dumps(test_alignment)}")

    out_tsv = f"{out_dir}/matching_results.tsv"
    _write_matching_results(con, scored, threshold, test_s1_raw_path, out_tsv)
    _log(f"wrote {out_tsv} (threshold={threshold})")

    grid = sorted({float(x) for x in np.round(np.concatenate([np.arange(0.90, 0.995, 0.01), [0.65, 0.995, threshold]]), 4)})
    profile = _test_profile(con, scored, checks["s1_entities"], grid)
    profile["selected"] = np.isclose(profile["threshold"], threshold)
    profile.to_csv(f"{art}/test_threshold_profile.csv", index=False)
    breakdown = con.execute(f"""
        SELECT s1.country, COUNT(*) AS n_entities,
               AVG((m.matched_entity_ids IS NOT NULL AND m.matched_entity_ids != '')::INT) AS frac_nonempty,
               AVG(len(string_split(m.matched_entity_ids, ','))) AS mean_matches_if_any
        FROM read_parquet('{test_s[0]}') s1
        LEFT JOIN read_csv('{out_tsv}', delim='\t', header=true, quote='"', all_varchar=true) m
          ON m.source1_entity_id = s1.entity_id
        GROUP BY 1 ORDER BY 2 DESC
    """).df()
    _dump(dict(
        run_id=run_id, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        threshold=threshold, matching_results=out_tsv, matching_results_md5=_md5(out_tsv),
        candidate_pairs=f"{out_dir}/candidate_pairs.tsv", test_candidates=test_cands, test_scored_dir=scored,
        model=model_path, model_md5=_md5(model_path), vec_md5=_md5(vec_path),
        selected_threshold_profile=profile[profile["selected"]].to_dict("records"),
        by_country=breakdown.to_dict("records"), leaderboard_score=None,
    ), f"{art}/run_output_record.json")
    _log("test profile at selected threshold: " + profile[profile["selected"]].to_string(index=False))
    print("\n=== TEST PREDICTIONS BY COUNTRY (descriptive, no ground truth) ===")
    print(breakdown.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--train-gt", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--test-s1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-train-negatives", type=int, default=8_000_000)
    parser.add_argument("--val-entities", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=None, help="feature worker processes (default: all cores)")
    parser.add_argument("--memory-limit-gb", type=int, default=None,
                        help="DuckDB memory limit; defaults to ~75%% of physical RAM")
    parser.add_argument("--resume", action="store_true",
                        help="reuse model/threshold/candidates/scores already in --artifacts-dir")
    parser.add_argument("--expect-train-s1", type=int, default=None, help="fail if train S1 entity count differs")
    parser.add_argument("--expect-test-s1", type=int, default=None, help="fail if test S1 entity count differs")
    parser.add_argument("--baseline-model", default=None, help="previous model, scored on the same holdout candidates")
    parser.add_argument("--baseline-vec", default=None, help="TF-IDF vectorizer that goes with --baseline-model")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run(
        args.processed_dir, args.train_gt, args.test_s1, args.out_dir, args.artifacts_dir,
        top_k=args.top_k, max_train_negatives=args.max_train_negatives, n_val_entities=args.val_entities,
        memory_limit_gb=args.memory_limit_gb, workers=args.workers, resume=args.resume,
        expect_train_s1=args.expect_train_s1, expect_test_s1=args.expect_test_s1,
        baseline_model=args.baseline_model, baseline_vec=args.baseline_vec, run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
