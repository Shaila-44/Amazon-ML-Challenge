"""Parallel featurize + score for large candidate sets.

DuckDB does the (multithreaded) join once and writes hash-sharded Parquet;
each worker process then reads its own shard, featurizes, scores and writes
an output part. No DataFrames are pickled between processes -- that was the
bottleneck of the previous design (workers idle at ~18% CPU).

Create the pool BEFORE importing duckdb in the parent (fork safety).
"""
import glob
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "/home/ubuntu/amazon/code/business_entity_resolution")
from src.features import FEATURE_COLUMNS, compute_features_for_chunk

_VEC = None
_BOOST = None


def _init(vec_path, model_path):
    global _VEC, _BOOST
    import lightgbm as lgb
    with open(vec_path, "rb") as f:
        _VEC = pickle.load(f)
    _BOOST = lgb.Booster(model_file=model_path)


def _work(job):
    in_path, out_path = job
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer, n = None, 0
    for b in pq.ParquetFile(in_path).iter_batches(batch_size=100_000):
        c = b.to_pandas()
        if not len(c):
            continue
        f = compute_features_for_chunk(c, _VEC)
        f["score"] = _BOOST.predict(f[FEATURE_COLUMNS])
        t = pa.Table.from_pandas(f, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
        writer.write_table(t)
        n += len(f)
    if writer is not None:
        writer.close()
    return n


def make_pool(vec_path, model_path, n=8):
    return Pool(n, initializer=_init, initargs=(vec_path, model_path))


def score_parallel(pool, cands_source_sql, s1, s2, s3, out_dir, tmp_dir, n_shards=32, mem_gb=40):
    import duckdb
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem_gb}GB'")
    con.execute(f"""
        COPY (
            SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens,
                   s1.name_norm AS name_norm_1, s23.name_norm AS name_norm_2,
                   s1.addr_norm AS addr_norm_1, s23.addr_norm AS addr_norm_2,
                   s1.name_tokens AS name_tokens_1, s23.name_tokens AS name_tokens_2,
                   s1.addr_tokens AS addr_tokens_1, s23.addr_tokens AS addr_tokens_2,
                   hash(c.source1_entity_id) % {n_shards} AS shard
            FROM {cands_source_sql} c
            JOIN read_parquet('{s1}') s1 ON s1.entity_id = c.source1_entity_id
            JOIN (SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{s2}')
                  UNION ALL
                  SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{s3}')) s23
              ON s23.entity_id = c.candidate_entity_id
        ) TO '{tmp_dir}' (FORMAT PARQUET, PARTITION_BY (shard))
    """)
    con.close()
    files = sorted(glob.glob(f"{tmp_dir}/**/*.parquet", recursive=True))
    print(f"  [{time.time()-t0:.0f}s] joined + sharded into {len(files)} files", flush=True)
    jobs = [(f, os.path.join(out_dir, f"part-{i:03d}.parquet")) for i, f in enumerate(files)]
    total = 0
    for i, n in enumerate(pool.imap_unordered(_work, jobs), 1):
        total += n
        print(f"  [{time.time()-t0:.0f}s] shard {i}/{len(jobs)} done, {total} pairs scored", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return total
