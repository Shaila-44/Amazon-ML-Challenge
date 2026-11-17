"""Phase 4: estimated holdout F_0.5 for the best blocking scheme at K=50/100/200.

Scheme: T1 token selection (8 rarest tokens that survive the S2/S3 filter)
+ ranking by same-field IDF sum (idf_same DESC, idf DESC, hash tie-break).
Top-200 per holdout entity is featurized once and scored with the CURRENT
checkpointed model (trained on the old candidate distribution, so these are
relative estimates across K, not final numbers).
"""
import sys
import time

sys.path.insert(0, "/home/ubuntu/amazon")
from par_score import make_pool, score_parallel

P = "/home/ubuntu/amazon/dataset/processed"
OUT = "/home/ubuntu/amazon/eval_out"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
S1, S2, S3 = f"{P}/train_source1.parquet", f"{P}/train_source2.parquet", f"{P}/train_source3.parquet"
MAX_DF_ABS, MAX_DF_FRAC, MIN_LEN, N_TOK, K_MAX = 1500, 0.02, 2, 8, 200
KS = (50, 100, 200)

if __name__ == "__main__":
    pool = make_pool(f"{P}/_vec_checkpoint.pkl", f"{P}/_model_checkpoint.txt")

    import duckdb
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 250)
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='44GB'")
    con.execute(f"""
        CREATE TABLE gt_pairs AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
        FROM read_csv('{GT}', delim='\t', header=true, quote='"', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
    con.execute(f"CREATE TABLE hold AS SELECT entity_id, country FROM read_parquet('{OUT}/holdout_entities.parquet')")

    def tok_union(path, c):
        return f"""SELECT entity_id, unnest(list_distinct(name_tokens)) AS token, 'name' AS token_type
                   FROM read_parquet('{path}') WHERE country = '{c}' AND len(name_tokens) > 0
                   UNION ALL
                   SELECT entity_id, unnest(list_distinct(addr_tokens)) AS token, 'addr' AS token_type
                   FROM read_parquet('{path}') WHERE country = '{c}' AND len(addr_tokens) > 0"""

    countries = sorted(con.execute("SELECT DISTINCT country FROM hold").fetchdf().country)
    for c in countries:
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
        con.execute(f"""CREATE OR REPLACE TEMP TABLE s1_sel AS
            SELECT entity_id, token, token_type FROM (
              SELECT t.*, ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY k.df ASC, t.token) AS rk
              FROM (SELECT s.entity_id, s.token, s.token_type
                    FROM s1_raw s
                    JOIN (SELECT token_type, token FROM s1_raw WHERE len(token) >= {MIN_LEN}
                          GROUP BY 1, 2 HAVING COUNT(DISTINCT entity_id) <= {MAX_DF_ABS}) f USING (token_type, token)
                    JOIN hold h ON h.entity_id = s.entity_id
                    WHERE len(s.token) >= {MIN_LEN}) t
              JOIN s23_keep k USING (token_type, token)) WHERE rk <= {N_TOK}""")
        con.execute(f"""
            CREATE OR REPLACE TABLE top_{c} AS
            SELECT source1_entity_id, candidate_entity_id, shared AS shared_tokens, rk FROM (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY source1_entity_id
                                           ORDER BY idf_same DESC, idf DESC, hash(candidate_entity_id)) AS rk
              FROM (SELECT s1.entity_id AS source1_entity_id, s23.entity_id AS candidate_entity_id,
                           COUNT(*) AS shared, SUM(s23.idf) AS idf,
                           SUM(CASE WHEN s1.token_type = s23.token_type THEN s23.idf ELSE 0 END) AS idf_same
                    FROM s1_sel s1 JOIN s23_tok s23 ON s1.token = s23.token
                    GROUP BY 1, 2)) WHERE rk <= {K_MAX}""")
        for tname in ("s23_raw", "s23_df", "tot", "s23_keep", "s23_tok", "s1_raw", "s1_sel"):
            con.execute(f"DROP TABLE {tname}")
        print(f"[{time.time()-t0:.0f}s] {c}: top-{K_MAX} candidates built", flush=True)

    kpath = f"{OUT}/k_cands_top{K_MAX}.parquet"
    con.execute(f"COPY ({' UNION ALL '.join(f'SELECT * FROM top_{c}' for c in countries)}) TO '{kpath}' (FORMAT PARQUET)")
    truth = con.sql("SELECT g.source1_entity_id, g.matched_entity_id FROM gt_pairs g JOIN hold h ON h.entity_id = g.source1_entity_id").df()
    con.close()
    print(f"[{time.time()-t0:.0f}s] top-{K_MAX} candidate pairs: {len(pd.read_parquet(kpath, columns=['rk']))}", flush=True)

    sdir = f"{OUT}/k_scored_top{K_MAX}"
    score_parallel(pool, f"read_parquet('{kpath}')", S1, S2, S3, sdir, "/home/ubuntu/amazon/.tmp_k_shards")
    pool.close()

    m = duckdb.sql(f"""SELECT s.source1_entity_id, s.candidate_entity_id, s.score, k.rk
                       FROM read_parquet('{sdir}/*.parquet') s
                       JOIN read_parquet('{kpath}') k USING (source1_entity_id, candidate_entity_id)""").df()
    gt_set = set(zip(truth.source1_entity_id, truth.matched_entity_id))
    m["label"] = [(a, b) in gt_set for a, b in zip(m.source1_entity_id, m.candidate_entity_id)]
    ent = pd.read_parquet(f"{OUT}/holdout_entities.parquet")

    def evaluate(pred):
        g = pred.groupby("source1_entity_id").agg(n_pred=("label", "size"), tp=("label", "sum"))
        e = ent.merge(g, left_on="entity_id", right_index=True, how="left").fillna({"n_pred": 0, "tp": 0})
        p = np.where(e.n_pred > 0, e.tp / e.n_pred.clip(lower=1), 0.0)
        r = np.where(e.n_truth > 0, e.tp / e.n_truth.clip(lower=1), 0.0)
        f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r + 1e-12), 0.0)
        f = np.where(e.n_truth == 0, (e.n_pred == 0).astype(float), f)
        return e.assign(F=f)

    grid = np.round(np.concatenate([np.arange(0.50, 0.9001, 0.02), np.arange(0.90, 0.9951, 0.005)]), 4)
    print("\n===== ESTIMATED HOLDOUT F_0.5 BY K (current model; reference: current blocking best = 0.6468 @0.96) =====")
    rows = []
    for K in KS:
        sub = m[m.rk <= K]
        found = sub[sub.label].groupby("source1_entity_id").size()
        e = ent.assign(found=ent.entity_id.map(found).fillna(0))
        r = np.where(e.n_truth > 0, e.found / e.n_truth.clip(lower=1), 0)
        ceil = np.where(e.n_truth == 0, 1.0, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)).mean()
        best = max(((t, evaluate(sub[sub.score >= t])) for t in grid), key=lambda x: x[1].F.mean())
        eb = best[1]
        by_c = eb.groupby("country").F.mean().round(4).to_dict()
        rows.append(dict(K=K, pairs=len(sub), pair_recall=sub.label.sum() / len(truth), ceiling_F05=ceil,
                         best_threshold=best[0], best_F05=eb.F.mean(), frac_empty=(eb.n_pred == 0).mean(),
                         frac_F0=(eb.F == 0).mean(), **{f"F05_{k}": v for k, v in by_c.items()}))
        print(f"  K={K} done ({time.time()-t0:.0f}s)", flush=True)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[{time.time()-t0:.0f}s] done", flush=True)
