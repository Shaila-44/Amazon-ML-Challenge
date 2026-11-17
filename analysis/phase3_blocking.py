"""Phase 3: which candidate-ranking / token-selection scheme recovers recall?

Exact emulation of blocking._generate_for_country for the 40k holdout
entities (doc-frequencies computed on the FULL country data, as production
does), without the top-k cap, for several token-selection variants. Then
recall@K by source under several ranking scores.

Token selection:
  T0  current: 8 rarest S1 tokens, ranked by COALESCE(s23_df, 0) -- tokens
      absent from S2/S3 count as "rarest" and can waste slots
  T1  8 rarest tokens among those that survive the S2/S3 filter
  T2  16 rarest tokens among those that survive the S2/S3 filter
Ranking:
  count_alpha  current production order (shared_tokens DESC, candidate id)
  count_hash   shared_tokens DESC, hash tie-break
  idf          sum of ln(N/df) over shared tokens
  idf_same     same, counting only name<->name and addr<->addr matches
"""
import time

import duckdb
import pandas as pd

pd.set_option("display.width", 250)
P = "/home/ubuntu/amazon/dataset/processed"
OUT = "/home/ubuntu/amazon/eval_out"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
S1, S2, S3 = f"{P}/train_source1.parquet", f"{P}/train_source2.parquet", f"{P}/train_source3.parquet"
MAX_DF_ABS, MAX_DF_FRAC, MIN_LEN = 1500, 0.02, 2

t0 = time.time()
con = duckdb.connect()
con.execute("SET memory_limit='44GB'")
con.execute(f"""
    CREATE TABLE gt_pairs AS
    SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
    FROM read_csv('{GT}', delim='\t', header=true, quote='"', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
con.execute(f"CREATE TABLE hold AS SELECT entity_id, country FROM read_parquet('{OUT}/holdout_entities.parquet')")
con.execute("""CREATE TABLE hold_truth AS SELECT g.source1_entity_id, g.matched_entity_id, left(g.matched_entity_id,2) AS src
               FROM gt_pairs g JOIN hold h ON h.entity_id = g.source1_entity_id""")


def tok_union(path, c):
    return f"""SELECT entity_id, unnest(list_distinct(name_tokens)) AS token, 'name' AS token_type
               FROM read_parquet('{path}') WHERE country = '{c}' AND len(name_tokens) > 0
               UNION ALL
               SELECT entity_id, unnest(list_distinct(addr_tokens)) AS token, 'addr' AS token_type
               FROM read_parquet('{path}') WHERE country = '{c}' AND len(addr_tokens) > 0"""


variants = {"T0": None, "T1": 8, "T2": 16}
for c in sorted(con.execute("SELECT DISTINCT country FROM hold").fetchdf().country):
    con.execute(f"CREATE OR REPLACE TEMP TABLE s23_raw AS {tok_union(S2, c)} UNION ALL {tok_union(S3, c)}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s23_df AS SELECT token_type, token, COUNT(DISTINCT entity_id) AS df
                    FROM s23_raw WHERE len(token) >= {MIN_LEN} GROUP BY 1, 2""")
    con.execute("CREATE OR REPLACE TEMP TABLE tot AS SELECT token_type, COUNT(DISTINCT entity_id) AS total FROM s23_raw GROUP BY 1")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s23_keep AS
                    SELECT d.token_type, d.token, d.df, ln(t.total * 1.0 / d.df) AS idf
                    FROM s23_df d JOIN tot t USING (token_type)
                    WHERE d.df <= t.total * {MAX_DF_FRAC} AND d.df <= {MAX_DF_ABS}""")
    con.execute("""CREATE OR REPLACE TEMP TABLE s23_tok AS
                   SELECT s.entity_id, s.token, s.token_type, k.idf FROM s23_raw s JOIN s23_keep k USING (token_type, token)""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE s1_raw AS {tok_union(S1, c)}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s1_capped AS
                    SELECT s.entity_id, s.token, s.token_type
                    FROM s1_raw s
                    JOIN (SELECT token_type, token FROM s1_raw WHERE len(token) >= {MIN_LEN}
                          GROUP BY 1, 2 HAVING COUNT(DISTINCT entity_id) <= {MAX_DF_ABS}) f USING (token_type, token)
                    JOIN hold h ON h.entity_id = s.entity_id
                    WHERE len(s.token) >= {MIN_LEN}""")
    print(f"[{time.time()-t0:.0f}s] {c}: token tables ready", flush=True)
    for v, k in variants.items():
        if v == "T0":
            sel = """SELECT entity_id, token, token_type FROM (
                       SELECT t.*, ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY COALESCE(d.df, 0) ASC) AS rk
                       FROM s1_capped t LEFT JOIN s23_df d USING (token_type, token)) WHERE rk <= 8"""
        else:
            sel = f"""SELECT entity_id, token, token_type FROM (
                       SELECT t.*, ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY k.df ASC, t.token) AS rk
                       FROM s1_capped t JOIN s23_keep k USING (token_type, token)) WHERE rk <= {k}"""
        con.execute(f"""
            CREATE OR REPLACE TABLE u_{v}_{c} AS
            SELECT s1.entity_id AS source1_entity_id, s23.entity_id AS candidate_entity_id,
                   COUNT(*) AS shared, SUM(s23.idf) AS idf,
                   SUM(CASE WHEN s1.token_type = s23.token_type THEN s23.idf ELSE 0 END) AS idf_same
            FROM ({sel}) s1 JOIN s23_tok s23 ON s1.token = s23.token
            GROUP BY 1, 2""")
        print(f"[{time.time()-t0:.0f}s] {c} {v}: {con.execute(f'SELECT COUNT(*) FROM u_{v}_{c}').fetchone()[0]} uncapped pairs", flush=True)
    for tname in ("s23_raw", "s23_df", "tot", "s23_keep", "s23_tok", "s1_raw", "s1_capped"):
        con.execute(f"DROP TABLE {tname}")

countries = sorted(con.execute("SELECT DISTINCT country FROM hold").fetchdf().country)
n_truth = dict(con.execute("SELECT src, COUNT(*) FROM hold_truth GROUP BY 1").fetchall())
print(f"\nholdout truth pairs by source: {n_truth}", flush=True)
rows = []
for v in variants:
    con.execute(f"""
        CREATE OR REPLACE TABLE u AS
        SELECT u.*, left(u.candidate_entity_id, 2) AS src, (g.matched_entity_id IS NOT NULL) AS label
        FROM ({' UNION ALL '.join(f'SELECT * FROM u_{v}_{c}' for c in countries)}) u
        LEFT JOIN gt_pairs g ON g.source1_entity_id = u.source1_entity_id AND g.matched_entity_id = u.candidate_entity_id""")
    orders = {
        "count_alpha": "shared DESC, candidate_entity_id",
        "count_hash": "shared DESC, hash(candidate_entity_id)",
        "idf": "idf DESC, hash(candidate_entity_id)",
        "idf_same": "idf_same DESC, idf DESC, hash(candidate_entity_id)",
    }
    for rname, order in orders.items():
        r = con.execute(f"""
            WITH rk AS (SELECT src, label, ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY {order}) AS rk FROM u)
            SELECT src,
                   SUM((label AND rk <= 50)::INT), SUM((label AND rk <= 100)::INT), SUM((label AND rk <= 200)::INT),
                   SUM(label::INT), SUM((rk <= 50)::INT)
            FROM rk GROUP BY src ORDER BY src""").fetchall()
        for src, r50, r100, r200, runc, n50 in r:
            rows.append(dict(tokens=v, ranking=rname, src=src,
                             recall_at50=r50 / n_truth[src], recall_at100=r100 / n_truth[src],
                             recall_at200=r200 / n_truth[src], recall_uncapped=runc / n_truth[src],
                             cand_slots_at50=n50))
    print(f"[{time.time()-t0:.0f}s] {v} rankings done", flush=True)

res = pd.DataFrame(rows)
print("\n===== RECALL BY TOKEN SELECTION x RANKING x SOURCE =====")
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
tot = res.groupby(["tokens", "ranking"]).apply(
    lambda d: pd.Series({k: (d[k] * d.src.map(n_truth)).sum() / sum(n_truth.values())
                         for k in ["recall_at50", "recall_at100", "recall_at200", "recall_uncapped"]})).reset_index()
print("\n===== OVERALL (S2+S3) =====")
print(tot.sort_values("recall_at50", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
res.to_csv(f"{OUT}/phase3_blocking_recall.csv", index=False)
print(f"[{time.time()-t0:.0f}s] done", flush=True)
