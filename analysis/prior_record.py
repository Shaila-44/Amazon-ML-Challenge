"""Provenance of submissions #1/#2 from the ledger + immutable artifacts only (read-only on all of them)."""
import datetime as dt, hashlib, json, os, re
import duckdb

H = "/home/ubuntu/amazon"
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def mtime(p): return dt.datetime.fromtimestamp(os.path.getmtime(p), dt.timezone.utc).isoformat()

ledger = [json.loads(l) for l in open(f"{H}/artifacts/submission_ledger.jsonl")]
cands = f"{H}/artifacts/test_candidates_v1.parquet"
n_pairs, n_ent, k_max = duckdb.sql(f"""SELECT SUM(n), COUNT(*), MAX(n) FROM (
    SELECT source1_entity_id, COUNT(*) n FROM read_parquet('{cands}') GROUP BY 1)""").fetchone()
blk = open(f"{H}/code/src_v1_backup_0121/blocking.py").read()
sig = re.search(r"def generate_candidates\((.*?)\) ->", blk, re.S).group(1)
order = re.search(r"ORDER BY (shared_tokens DESC, candidate_entity_id)", blk)
model = f"{H}/dataset/processed/_model_checkpoint.txt"
vec = f"{H}/dataset/processed/_vec_checkpoint.pkl"
thr = open(f"{H}/dataset/processed/_threshold_checkpoint.txt").read().strip()
common = dict(
    candidate_generation=dict(code=f"{H}/code/src_v1_backup_0121/blocking.py", code_md5=md5(f"{H}/code/src_v1_backup_0121/blocking.py"),
                              generate_candidates_signature=" ".join(sig.split()),
                              ranking=order.group(1) if order else "NOT FOUND",
                              token_selection="8 rarest S1 tokens ranked by COALESCE(s23_df, 0)"),
    K=int(k_max), test_candidates=cands, test_candidates_md5=md5(cands),
    test_candidate_pairs=int(n_pairs), test_entities_with_candidates=int(n_ent),
    model=model, model_md5=md5(model), model_mtime_utc=mtime(model),
    vectorizer=vec, vectorizer_md5=md5(vec), model_run_threshold_checkpoint=thr,
    test_scores=f"{H}/artifacts/test_scored_v1", model_run_id="not recorded (no run id existed for v1)",
)
subs = []
for num, thr_v, out, how in [(1, 0.65, f"{H}/output/matching_results.tsv", "original pipeline run (src.pipeline --resume)"),
                             (2, 0.96, f"{H}/output_v2/matching_results.tsv", "make_submission.py on test_scored_v1")]:
    led = [r for r in ledger if r["threshold"] == thr_v and not r["dedup"]]
    subs.append(dict(submission_number=num, artifact_version=f"v{num}", produced_by=how, threshold=thr_v, dedup=False,
                     matching_results=out, matching_results_md5=md5(out), matching_results_mtime_utc=mtime(out),
                     leaderboard_score=led[0]["leaderboard_score"] if led else None,
                     submission_timestamp="not recorded in ledger", submission_status="scored" if led and led[0]["leaderboard_score"] else "unknown",
                     ledger_entry=led[0] if led else None, **common))
rec = dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
           source="artifacts/submission_ledger.jsonl + immutable artifacts (read-only)",
           note=("v1 rebuild via make_submission.py at 0.65 matched the originally submitted file row-for-row (identical rows 1.000000). "
                 "Ledger fields leaderboard_score/label for both entries were filled in on 2026-09-26 from scores reported by the user; "
                 "the ledger is append-only from then on."),
           submissions=subs)
os.makedirs(f"{H}/artifacts_v3", exist_ok=True)
json.dump(rec, open(f"{H}/artifacts_v3/prior_submissions.json", "w"), indent=1)
print(json.dumps({k: v for k, v in subs[0].items() if k != "ledger_entry"}, indent=1))
print("#2:", subs[1]["threshold"], subs[1]["leaderboard_score"], subs[1]["matching_results_md5"])
