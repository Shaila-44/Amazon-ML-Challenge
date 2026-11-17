"""Re-score every test candidate pair of the submitted run with the
checkpointed model; keep features + scores as a read-only artifact so
threshold / dedup variants are cheap post-processing."""
import glob
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/amazon")
from par_score import make_pool, score_parallel

P = "/home/ubuntu/amazon/dataset/processed"
A = "/home/ubuntu/amazon/artifacts"
CANDS = f"{A}/test_candidates_v1.parquet"  # exact candidate set of the submitted run
OUT = f"{A}/test_scored_v1"
EXPECTED_PAIRS = 85_150_507

if __name__ == "__main__":
    pool = make_pool(f"{P}/_vec_checkpoint.pkl", f"{P}/_model_checkpoint.txt")
    t0 = time.time()
    n = score_parallel(pool, f"read_parquet('{CANDS}')",
                       f"{P}/test_source1.parquet", f"{P}/test_source2.parquet", f"{P}/test_source3.parquet",
                       OUT, "/home/ubuntu/amazon/.tmp_rescore_shards")
    pool.close()
    for f in glob.glob(f"{OUT}/*.parquet"):
        os.chmod(f, 0o444)
    os.chmod(OUT, 0o555)
    status = "OK" if n == EXPECTED_PAIRS else f"MISMATCH (expected {EXPECTED_PAIRS})"
    print(f"[{time.time()-t0:.0f}s] done: {n} scored pairs -> {OUT} (read-only) {status}", flush=True)
