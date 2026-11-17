"""Equivalence check: bucketed blocking vs the candidates the unbucketed
implementation produced in the smoke-guardrails run (read-only reference)."""
import json
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/amazon/code/business_entity_resolution")
import duckdb
from src import blocking

S = "/home/ubuntu/amazon/smoke"
OUT = "/home/ubuntu/amazon/bucket_equivalence"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit='46GB'")
con.execute(f"""CREATE TABLE gt_pairs AS
    SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
    FROM read_csv('{GT}', delim='\t', header=true, quote='"', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
COLS = "source1_entity_id, candidate_entity_id, shared_tokens, candidate_rank"


def profile(path):
    n, ents, chk = con.execute(f"""SELECT COUNT(*), COUNT(DISTINCT source1_entity_id),
        SUM(hash({COLS.replace(', ', " || '|' || ")})::HUGEINT) FROM read_parquet('{path}')""").fetchone()
    src = dict(con.execute(f"SELECT left(candidate_entity_id,2), COUNT(*) FROM read_parquet('{path}') GROUP BY 1 ORDER BY 1").fetchall())
    return dict(rows=n, entities=ents, checksum=str(chk), by_source=src)


def diff(a, b):
    only_a = con.execute(f"SELECT COUNT(*) FROM (SELECT {COLS} FROM read_parquet('{a}') EXCEPT ALL SELECT {COLS} FROM read_parquet('{b}'))").fetchone()[0]
    only_b = con.execute(f"SELECT COUNT(*) FROM (SELECT {COLS} FROM read_parquet('{b}') EXCEPT ALL SELECT {COLS} FROM read_parquet('{a}'))").fetchone()[0]
    pair_only_a = con.execute(f"""SELECT COUNT(*) FROM (SELECT source1_entity_id, candidate_entity_id FROM read_parquet('{a}')
                                  EXCEPT ALL SELECT source1_entity_id, candidate_entity_id FROM read_parquet('{b}'))""").fetchone()[0]
    ents = con.execute(f"""SELECT COUNT(DISTINCT source1_entity_id) FROM (
        (SELECT {COLS} FROM read_parquet('{a}') EXCEPT ALL SELECT {COLS} FROM read_parquet('{b}'))
        UNION ALL (SELECT {COLS} FROM read_parquet('{b}') EXCEPT ALL SELECT {COLS} FROM read_parquet('{a}')))""").fetchone()[0]
    return dict(rows_only_in_first=only_a, rows_only_in_second=only_b,
                candidate_pairs_only_in_first=pair_only_a, entities_affected=ents,
                identical=(only_a == 0 and only_b == 0))


def recall(path, s1):
    return con.sql(f"""
        WITH t AS (SELECT left(g.matched_entity_id,2) src, s.country, c.candidate_rank
                   FROM gt_pairs g JOIN read_parquet('{s1}') s ON s.entity_id = g.source1_entity_id
                   LEFT JOIN read_parquet('{path}') c ON c.source1_entity_id = g.source1_entity_id AND c.candidate_entity_id = g.matched_entity_id)
        SELECT COALESCE(country,'ALL') country, COALESCE(src,'ALL') src, COUNT(*) n_true,
               AVG(COALESCE(candidate_rank<=50,false)::INT) k50, AVG(COALESCE(candidate_rank<=100,false)::INT) k100,
               AVG(COALESCE(candidate_rank<=200,false)::INT) k200
        FROM t GROUP BY GROUPING SETS ((country,src),(country),(src),()) ORDER BY 1,2""").df()


report = {}
for split in ("train", "test"):
    s = tuple(f"{S}/processed/{split}_source{i}.parquet" for i in (1, 2, 3))
    ref = f"{S}/artifacts/{split}_candidates.parquet"
    r = dict(reference=ref, reference_profile=profile(ref), runs={})
    for name, nb in (("buckets1", 1), ("buckets8_run1", 8), ("buckets8_run2", 8)):
        path = f"{OUT}/{split}_{name}.parquet"
        stats, t0 = [], time.time()
        blocking.generate_candidates(con, *s, top_k=200, n_buckets=nb, stats=stats).write_parquet(path)
        r["runs"][name] = dict(n_buckets=nb, seconds=round(time.time() - t0, 1), profile=profile(path),
                               vs_reference=diff(ref, path), bucket_stats=stats)
        print(f"{split} {name}: {json.dumps(r['runs'][name]['vs_reference'])}  ({r['runs'][name]['seconds']}s)", flush=True)
    r["determinism_buckets8_run1_vs_run2"] = diff(f"{OUT}/{split}_buckets8_run1.parquet", f"{OUT}/{split}_buckets8_run2.parquet")
    print(f"{split} determinism run1 vs run2: {json.dumps(r['determinism_buckets8_run1_vs_run2'])}", flush=True)
    if split == "train":
        a, b = recall(ref, s[0]), recall(f"{OUT}/train_buckets8_run1.parquet", s[0])
        r["blocking_recall_reference"] = a.to_dict("records")
        r["blocking_recall_bucketed"] = b.to_dict("records")
        r["blocking_recall_identical"] = a.equals(b)
        print("train blocking recall reference vs bucketed identical:", a.equals(b))
        print(b.to_string(index=False))
    report[split] = r

json.dump(report, open(f"{OUT}/equivalence_report.json", "w"), indent=1, default=str)
for split in report:
    print(f"\n{split}: reference {report[split]['reference_profile']}")
    for name, run in report[split]["runs"].items():
        print(f"  {name}: {run['profile']}")
    for st in report[split]["runs"]["buckets8_run1"]["bucket_stats"]:
        print("   ", st)
