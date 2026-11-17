"""Build matching_results.tsv from saved test scores.

    python make_submission.py --threshold 0.65 --out /tmp/check.tsv --compare ~/amazon/output/matching_results.tsv
    python make_submission.py --threshold 0.93 --out ~/amazon/output_v2/matching_results.tsv
    python make_submission.py --threshold 0.93 --dedup --out ~/amazon/output_v3/matching_results.tsv

--dedup: each S2/S3 record goes only to the S1 entity with its highest score
(ground truth is a strict partition: every record has at most one S1 owner).
"""
import argparse
import json
import os

import duckdb

P = "/home/ubuntu/amazon/dataset/processed"
REQUIRED = "/home/ubuntu/amazon/dataset_meta/test_source1.tsv"

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, required=True)
ap.add_argument("--dedup", action="store_true")
ap.add_argument("--out", required=True)
ap.add_argument("--compare", default=None, help="existing matching_results.tsv to diff against")
a = ap.parse_args()

con = duckdb.connect()
con.execute("SET memory_limit='40GB'")
dedup = "QUALIFY score >= MAX(score) OVER (PARTITION BY candidate_entity_id)" if a.dedup else ""
con.execute(f"""
    CREATE TABLE sel AS
    SELECT source1_entity_id, candidate_entity_id, score
    FROM read_parquet('/home/ubuntu/amazon/artifacts/test_scored_v1/*.parquet')
    WHERE score >= {a.threshold}
    {dedup}
""")
con.execute("""
    CREATE TABLE m AS
    SELECT source1_entity_id, string_agg(candidate_entity_id, ',' ORDER BY candidate_entity_id) AS matched_entity_ids,
           COUNT(*) AS n
    FROM sel GROUP BY 1
""")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
con.execute(f"""
    COPY (
        SELECT r.entity_id AS source1_entity_id, m.matched_entity_ids
        FROM read_csv('{REQUIRED}', delim='\t', header=true, quote='"', all_varchar=true) r
        LEFT JOIN m ON m.source1_entity_id = r.entity_id
        ORDER BY r.entity_id
    ) TO '{a.out}' (DELIMITER '\t', HEADER true, NULL '')
""")
n_req = con.execute(f"SELECT COUNT(*) FROM read_csv('{REQUIRED}', delim='\t', header=true, quote='\"', all_varchar=true)").fetchone()[0]
n_any, mean_n = con.execute("SELECT COUNT(*), AVG(n) FROM m").fetchone()
dup = con.execute("SELECT COUNT(*) FROM (SELECT candidate_entity_id FROM sel GROUP BY 1 HAVING COUNT(*) > 1)").fetchone()[0]
print(f"threshold={a.threshold} dedup={a.dedup}: {con.execute('SELECT COUNT(*) FROM sel').fetchone()[0]} pairs, "
      f"{n_any}/{n_req} entities non-empty ({n_any/n_req:.4f}), mean matches if any={mean_n:.3f}, "
      f"records claimed by >1 S1: {dup}")
con.execute(f"""
    CREATE TABLE prof AS
    SELECT r.entity_id, t1.country, COALESCE(m.n, 0) AS n
    FROM read_csv('{REQUIRED}', delim='\t', header=true, quote='"', all_varchar=true) r
    JOIN read_parquet('{P}/test_source1.parquet') t1 ON t1.entity_id = r.entity_id
    LEFT JOIN m ON m.source1_entity_id = r.entity_id
""")
hist = dict(con.execute("""
    SELECT CASE WHEN n >= 10 THEN '10+' ELSE n::VARCHAR END, ROUND(COUNT(*)*1.0/SUM(COUNT(*)) OVER (), 4)
    FROM prof GROUP BY 1 ORDER BY MIN(n)""").fetchall())
by_country = {c: dict(frac_nonempty=round(f, 4), mean_n_if_any=round(mn or 0, 3)) for c, f, mn in con.execute("""
    SELECT country, AVG((n > 0)::INT), AVG(n) FILTER (WHERE n > 0) FROM prof GROUP BY 1 ORDER BY 1""").fetchall()}
by_src = dict(con.execute("SELECT left(candidate_entity_id, 2), COUNT(*) FROM sel GROUP BY 1 ORDER BY 1").fetchall())
record = dict(out=a.out, threshold=a.threshold, dedup=a.dedup,
              pairs=con.execute("SELECT COUNT(*) FROM sel").fetchone()[0],
              frac_nonempty=round(n_any / n_req, 4), mean_n_if_any=round(mean_n, 3),
              records_claimed_by_multiple_s1=dup, n_matches_hist=hist, by_country=by_country, by_source=by_src,
              leaderboard_score=None)
print(json.dumps(record, indent=1))
with open("/home/ubuntu/amazon/artifacts/submission_ledger.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
if a.compare:
    same = con.execute(f"""
        SELECT AVG((COALESCE(x.matched_entity_ids,'') = COALESCE(y.matched_entity_ids,''))::INT)
        FROM read_csv('{a.out}', delim='\t', header=true, quote='"', all_varchar=true) x
        JOIN read_csv('{a.compare}', delim='\t', header=true, quote='"', all_varchar=true) y USING (source1_entity_id)
    """).fetchone()[0]
    print(f"identical rows vs {a.compare}: {same:.6f}")
