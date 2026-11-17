"""Parallel feature computation (and optional scoring) for large candidate sets.

DuckDB does the multithreaded join once and writes hash-sharded Parquet;
each worker process reads its own shard, featurizes (and scores, if a model
is given) and writes an output part. Nothing large is pickled between
processes -- an earlier design that shipped DataFrames through the pool left
workers idle at ~18% CPU.

Workers use the "spawn" start method, so this is safe to call from a process
that already holds a DuckDB connection (forking a process with live DuckDB
threads is not).
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import pickle
import shutil

from .features import FEATURE_COLUMNS, compute_features_for_chunk

_VEC = None
_BOOSTER = None


def _init(vec_path: str, model_path: str | None) -> None:
    global _VEC, _BOOSTER
    with open(vec_path, "rb") as f:
        _VEC = pickle.load(f)
    if model_path:
        import lightgbm as lgb
        _BOOSTER = lgb.Booster(model_file=model_path)


def _work(job: tuple[str, str]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    in_path, out_path = job
    writer, n = None, 0
    for batch in pq.ParquetFile(in_path).iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        if chunk.empty:
            continue
        feats = compute_features_for_chunk(chunk, _VEC)
        if _BOOSTER is not None:
            feats["score"] = _BOOSTER.predict(feats[FEATURE_COLUMNS])
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        n += len(feats)
    if writer is not None:
        writer.close()
    return n


def _joined_sql(candidates_sql: str, s1_parquet: str, s2_parquet: str, s3_parquet: str, extra: str = "") -> str:
    return f"""
        SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens,
               s1.name_norm AS name_norm_1, s23.name_norm AS name_norm_2,
               s1.addr_norm AS addr_norm_1, s23.addr_norm AS addr_norm_2,
               s1.name_tokens AS name_tokens_1, s23.name_tokens AS name_tokens_2,
               s1.addr_tokens AS addr_tokens_1, s23.addr_tokens AS addr_tokens_2{extra}
        FROM {candidates_sql} c
        JOIN read_parquet('{s1_parquet}') s1 ON s1.entity_id = c.source1_entity_id
        JOIN (SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{s2_parquet}')
              UNION ALL
              SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{s3_parquet}')) s23
          ON s23.entity_id = c.candidate_entity_id"""


def verify_alignment(
    con, candidates_sql: str, feats_dir: str, s1_parquet: str, s2_parquet: str, s3_parquet: str,
    vec_path: str, n_sample: int = 2000,
) -> dict:
    """Check that sharded/parallel feature output is exactly the candidate set:
    same row count, same multiset of (source1, candidate) keys and
    shared_tokens (order-independent hash checksums), and -- for a
    deterministic sample of pairs -- features identical to a single-process
    recomputation. Raises on any mismatch."""
    import numpy as np

    feats = f"read_parquet('{feats_dir}/*.parquet')"
    key = "hash(source1_entity_id || '|' || candidate_entity_id)::HUGEINT"
    key_st = "hash(source1_entity_id || '|' || candidate_entity_id || '|' || shared_tokens)::HUGEINT"
    c_n, c_k, c_s = con.execute(f"SELECT COUNT(*), SUM({key}), SUM({key_st}) FROM {candidates_sql}").fetchone()
    f_n, f_k, f_s = con.execute(f"SELECT COUNT(*), SUM({key}), SUM({key_st}) FROM {feats}").fetchone()
    sample = con.sql(
        _joined_sql(candidates_sql, s1_parquet, s2_parquet, s3_parquet)
        + f" ORDER BY hash(c.source1_entity_id || '|' || c.candidate_entity_id) LIMIT {n_sample}"
    ).df()
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    ref = compute_features_for_chunk(sample, vec)
    got = con.execute(f"""
        SELECT f.* FROM {feats} f
        JOIN (SELECT UNNEST(?) AS source1_entity_id, UNNEST(?) AS candidate_entity_id) k
        USING (source1_entity_id, candidate_entity_id)
    """, [ref["source1_entity_id"].tolist(), ref["candidate_entity_id"].tolist()]).df()
    merged = ref.merge(got, on=["source1_entity_id", "candidate_entity_id"], suffixes=("_ref", "_par"))
    max_diff = max(float(np.nanmax(np.abs(merged[f"{c}_ref"].astype(float) - merged[f"{c}_par"].astype(float))))
                   for c in FEATURE_COLUMNS)
    result = dict(candidates=int(c_n), feature_rows=int(f_n), key_checksum_match=c_k == f_k,
                  shared_tokens_checksum_match=c_s == f_s, spot_checked=int(len(merged)),
                  spot_check_max_abs_diff=max_diff)
    ok = (c_n == f_n and c_k == f_k and c_s == f_s and len(merged) == len(ref) and max_diff <= 1e-9)
    if not ok:
        raise RuntimeError(f"feature/candidate alignment check FAILED: {result}")
    return result


def featurize_parallel(
    con,
    candidates_sql: str,
    s1_parquet: str,
    s2_parquet: str,
    s3_parquet: str,
    out_dir: str,
    vec_path: str,
    model_path: str | None = None,
    workers: int | None = None,
    n_shards: int = 32,
    n_passes: int = 1,
) -> int:
    """Featurize every pair in `candidates_sql` (a FROM-clause source with
    source1_entity_id, candidate_entity_id, shared_tokens) into Parquet parts
    under `out_dir`; adds a `score` column when `model_path` is given.
    Returns the number of pairs written.

    `n_passes` splits the work by a salted S1 hash so only one pass's
    pre-joined shards (names + token lists, ~25GB for 310M pairs in one go)
    sit on disk at a time. Features are computed per row, so the output rows
    do not depend on the number of passes or shards."""
    tmp_dir = out_dir.rstrip("/") + "_shards_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    ctx = mp.get_context("spawn")
    total = 0
    with ctx.Pool(workers or os.cpu_count(), initializer=_init, initargs=(vec_path, model_path)) as pool:
        for p in range(n_passes):
            source = (candidates_sql if n_passes == 1 else
                      f"(SELECT * FROM {candidates_sql} WHERE hash(source1_entity_id || '#pass') % {n_passes} = {p})")
            joined = _joined_sql(source, s1_parquet, s2_parquet, s3_parquet,
                                 extra=f",\n               hash(c.source1_entity_id) % {n_shards} AS shard")
            con.execute(f"COPY ({joined}) TO '{tmp_dir}' (FORMAT PARQUET, PARTITION_BY (shard), COMPRESSION zstd)")
            files = sorted(glob.glob(f"{tmp_dir}/**/*.parquet", recursive=True))
            jobs = [(f, os.path.join(out_dir, f"part-p{p}-{i:03d}.parquet")) for i, f in enumerate(files)]
            for i, n in enumerate(pool.imap_unordered(_work, jobs), 1):
                total += n
                print(f"  pass {p + 1}/{n_passes}: featurized shard {i}/{len(jobs)} ({total} pairs total)", flush=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return total
