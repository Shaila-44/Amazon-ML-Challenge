"""Equivalence + reproducibility check for deterministic, bucketed blocking.

Required (all must hold, train and test smoke data):
  1. 8-bucket run1 == 8-bucket run2 (pairs AND ranks), and == 8-bucket at 3 threads
  2. 8-bucket == unbucketed (n_buckets=1)
  3. train blocking recall identical between bucketed and unbucketed
  4. S2/S3 counts identical between bucketed and unbucketed
  5. K behaviour: <=200 per entity, no duplicate pairs, contiguous ranks
Informational: differences vs the earlier (non-deterministic) smoke reference.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/amazon/code/business_entity_resolution")
import duckdb
from src import blocking

S = "/home/ubuntu/amazon/smoke"
OUT = "/home/ubuntu/amazon/bucket_equivalence_v2"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
K = 200
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
    max_n, dups, gaps = con.execute(f"""WITH p AS (SELECT source1_entity_id, COUNT(*) n, COUNT(DISTINCT candidate_entity_id) nd,
                                        MIN(candidate_rank) mn, MAX(candidate_rank) mx FROM read_parquet('{path}') GROUP BY 1)
                                        SELECT MAX(n), SUM((n != nd)::INT), SUM((mn != 1 OR mx != n)::INT) FROM p""").fetchone()
    return dict(rows=n, entities=ents, checksum=str(chk), by_source=src, max_per_entity=max_n,
                entities_with_dup_pairs=dups, entities_with_rank_gaps=gaps,
                k_behaviour_ok=(max_n <= K and dups == 0 and gaps == 0))


def diff(a, b):
    q = lambda x, y, c: con.execute(f"SELECT COUNT(*) FROM (SELECT {c} FROM read_parquet('{x}') EXCEPT ALL SELECT {c} FROM read_parquet('{y}'))").fetchone()[0]
    r = dict(rows_only_in_first=q(a, b, COLS), rows_only_in_second=q(b, a, COLS),
             pairs_only_in_first=q(a, b, "source1_entity_id, candidate_entity_id"),
             pairs_only_in_second=q(b, a, "source1_entity_id, candidate_entity_id"))
    r["identical"] = r["rows_only_in_first"] == 0 and r["rows_only_in_second"] == 0
    return r


def recall(path, s1):
    return con.sql(f"""
        WITH t AS (SELECT left(g.matched_entity_id,2) src, s.country, c.candidate_rank
                   FROM gt_pairs g JOIN read_parquet('{s1}') s ON s.entity_id = g.source1_entity_id
                   LEFT JOIN read_parquet('{path}') c ON c.source1_entity_id = g.source1_entity_id AND c.candidate_entity_id = g.matched_entity_id)
        SELECT COALESCE(country,'ALL') country, COALESCE(src,'ALL') src, COUNT(*) n_true,
               AVG(COALESCE(candidate_rank<=50,false)::INT) k50, AVG(COALESCE(candidate_rank<=100,false)::INT) k100,
               AVG(COALESCE(candidate_rank<=200,false)::INT) k200
        FROM t GROUP BY GROUPING SETS ((country,src),(country),(src),()) ORDER BY 1,2""").df()


report, checks = {}, {}
for split in ("train", "test"):
    s = tuple(f"{S}/processed/{split}_source{i}.parquet" for i in (1, 2, 3))
    ref = f"{S}/artifacts/{split}_candidates.parquet"
    r = dict(reference_profile=profile(ref), runs={})
    for name, nb, threads in (("unbucketed", 1, None), ("b8_run1", 8, None), ("b8_run2", 8, None), ("b8_threads3", 8, 3)):
        if threads:
            con.execute(f"SET threads = {threads}")
        path = f"{OUT}/{split}_{name}.parquet"
        stats, t0 = [], time.time()
        blocking.generate_candidates(con, *s, top_k=K, n_buckets=nb, stats=stats).write_parquet(path)
        if threads:
            con.execute("RESET threads")
        r["runs"][name] = dict(n_buckets=nb, threads=threads or "default", seconds=round(time.time() - t0, 1),
                               profile=profile(path), vs_reference_informational=diff(ref, path), bucket_stats=stats)
        print(f"{split} {name}: {r['runs'][name]['seconds']}s  profile={json.dumps({k: v for k, v in r['runs'][name]['profile'].items() if k != 'checksum'})}", flush=True)
    p = lambda n: f"{OUT}/{split}_{n}.parquet"
    r["b8_run1_vs_b8_run2"] = diff(p("b8_run1"), p("b8_run2"))
    r["b8_run1_vs_b8_threads3"] = diff(p("b8_run1"), p("b8_threads3"))
    r["b8_run1_vs_unbucketed"] = diff(p("b8_run1"), p("unbucketed"))
    checks[f"{split}: 1a 8-bucket run1 == run2"] = r["b8_run1_vs_b8_run2"]["identical"]
    checks[f"{split}: 1b 8-bucket 8 threads == 3 threads"] = r["b8_run1_vs_b8_threads3"]["identical"]
    checks[f"{split}: 2 bucketed == unbucketed"] = r["b8_run1_vs_unbucketed"]["identical"]
    checks[f"{split}: 4 S2/S3 counts bucketed == unbucketed"] = (
        r["runs"]["b8_run1"]["profile"]["by_source"] == r["runs"]["unbucketed"]["profile"]["by_source"])
    checks[f"{split}: 5 K behaviour (all runs)"] = all(x["profile"]["k_behaviour_ok"] for x in r["runs"].values())
    if split == "train":
        a, b, ref_rec = recall(p("unbucketed"), s[0]), recall(p("b8_run1"), s[0]), recall(ref, s[0])
        checks["train: 3 blocking recall bucketed == unbucketed"] = a.equals(b)
        r["recall_bucketed"] = b.to_dict("records")
        r["recall_delta_vs_reference_informational"] = (
            b.set_index(["country", "src"])[["k50", "k100", "k200"]] - ref_rec.set_index(["country", "src"])[["k50", "k100", "k200"]]
        ).reset_index().to_dict("records")
        print(b.to_string(index=False))
    report[split] = r

report["checks"] = checks
report["all_passed"] = all(checks.values())
json.dump(report, open(f"{OUT}/equivalence_report.json", "w"), indent=1, default=str)
print("\n===== REQUIRED CHECKS =====")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("ALL PASSED" if report["all_passed"] else "SOME CHECKS FAILED")
print("\n===== INFORMATIONAL: vs earlier non-deterministic smoke reference =====")
for split in ("train", "test"):
    print(split, "reference by_source:", report[split]["reference_profile"]["by_source"],
          "| b8_run1 by_source:", report[split]["runs"]["b8_run1"]["profile"]["by_source"],
          "| diff:", report[split]["runs"]["b8_run1"]["vs_reference_informational"])
print("train recall delta vs reference (k50,k100,k200):",
      [(d["country"], d["src"], round(d["k50"], 6), round(d["k100"], 6), round(d["k200"], 6)) for d in report["train"]["recall_delta_vs_reference_informational"]])
