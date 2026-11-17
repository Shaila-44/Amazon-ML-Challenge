"""Phase-2 forensics on the 40k holdout (read-only w.r.t. production).

A. full-scale blocking recall by country x source
B. exact uncapped blocking for holdout -> recall under alternative top-k / tie-break schemes
C. dense threshold sweep, P/R, cardinality, S2 vs S3 behaviour
D. competitor scoring + record-level one-to-one (dedup) variants
E. TP / FP / FN cause breakdown
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "/home/ubuntu/amazon/code/business_entity_resolution")
from src.features import FEATURE_COLUMNS, compute_features_for_chunk

P = "/home/ubuntu/amazon/dataset/processed"
OUT = "/home/ubuntu/amazon/eval_out"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
S1, S2, S3 = f"{P}/train_source1.parquet", f"{P}/train_source2.parquet", f"{P}/train_source3.parquet"
CANDS = f"{P}/_eval_train_candidates.parquet"
VEC = f"{P}/_vec_checkpoint.pkl"

_VEC = None


def _init(path):
    global _VEC
    with open(path, "rb") as f:
        _VEC = pickle.load(f)


def _feat(chunk):
    return compute_features_for_chunk(chunk, _VEC)


if __name__ == "__main__":
    pool = Pool(8, initializer=_init, initargs=(VEC,))  # fork before DuckDB threads exist

    import duckdb
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from src import blocking

    pd.set_option("display.width", 250)
    t0 = time.time()

    def log(m):
        print(f"\n[{time.time()-t0:.0f}s] ===== {m} =====", flush=True)

    def show(x):
        print(x.to_string(index=False, float_format=lambda v: f"{v:.4f}") if isinstance(x, pd.DataFrame) else x, flush=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='40GB'")
    con.execute(f"""
        CREATE TABLE gt_pairs AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
        FROM read_csv('{GT}', delim='\t', header=true, quote='"', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    """)
    con.execute(f"CREATE TABLE s1c AS SELECT entity_id, country FROM read_parquet('{S1}')")
    ent = pd.read_parquet(f"{OUT}/holdout_entities.parquet")
    con.register("hold_df", ent[["entity_id", "country"]])
    con.execute("CREATE TABLE hold_ents AS SELECT * FROM hold_df")
    hold_set = set(ent.entity_id)

    # ---------------- A ----------------
    log("A. full-scale blocking recall (all 2.2M train S1) by country x source")
    show(con.sql(f"""
        SELECT s.country, left(g.matched_entity_id,2) AS src, COUNT(*) AS n_true,
               COUNT(c.candidate_entity_id) AS n_found,
               COUNT(c.candidate_entity_id)*1.0/COUNT(*) AS recall
        FROM gt_pairs g JOIN s1c s ON s.entity_id = g.source1_entity_id
        LEFT JOIN read_parquet('{CANDS}') c
          ON c.source1_entity_id = g.source1_entity_id AND c.candidate_entity_id = g.matched_entity_id
        GROUP BY GROUPING SETS ((s.country, src), (src), ())
        ORDER BY 1 NULLS LAST, 2 NULLS LAST""").df())
    show(con.sql(f"""SELECT left(candidate_entity_id,2) AS src, COUNT(*) AS n_cands, AVG(shared_tokens) AS avg_shared
                    FROM read_parquet('{CANDS}') GROUP BY 1 ORDER BY 1""").df())

    # ---------------- B ----------------
    log("B. exact uncapped blocking for holdout entities")
    parts = []
    for country in sorted(ent.country.unique()):
        blocking._generate_for_country(con, S1, S2, S3, country, 50, 2, 0.02, 1500, 8)  # defines the token views
        con.execute(f"""
            CREATE OR REPLACE TABLE unc_{country} AS
            SELECT s1.entity_id AS source1_entity_id, s23.entity_id AS candidate_entity_id, COUNT(*) AS shared_tokens
            FROM s1_tokens s1 JOIN hold_ents h ON h.entity_id = s1.entity_id
            JOIN s23_tokens s23 ON s1.token = s23.token
            GROUP BY 1, 2""")
        parts.append(f"SELECT * FROM unc_{country}")
        print(country, con.execute(f"SELECT COUNT(*) FROM unc_{country}").fetchone()[0], "uncapped pairs", flush=True)
    con.execute(f"""
        CREATE TABLE unc AS
        SELECT u.*, left(u.candidate_entity_id,2) AS src, (g.matched_entity_id IS NOT NULL) AS label
        FROM ({' UNION ALL '.join(parts)}) u
        LEFT JOIN gt_pairs g ON g.source1_entity_id = u.source1_entity_id AND g.matched_entity_id = u.candidate_entity_id""")
    con.execute("""
        CREATE TABLE unc_r AS SELECT *,
          ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY shared_tokens DESC, candidate_entity_id) AS rk_current,
          ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY shared_tokens DESC, hash(candidate_entity_id)) AS rk_hash,
          ROW_NUMBER() OVER (PARTITION BY source1_entity_id, src ORDER BY shared_tokens DESC, hash(candidate_entity_id)) AS rk_src
        FROM unc""")
    con.execute("""CREATE TABLE hold_truth AS SELECT g.source1_entity_id, g.matched_entity_id, left(g.matched_entity_id,2) AS src
                   FROM gt_pairs g JOIN hold_ents h ON h.entity_id = g.source1_entity_id""")
    show(con.sql("SELECT src, COUNT(*) AS n_truth FROM hold_truth GROUP BY 1 ORDER BY 1").df())
    show(con.sql("""
        WITH t AS (SELECT src, COUNT(*) AS n_truth FROM hold_truth GROUP BY 1),
             s AS (
               SELECT src,
                 SUM((label AND rk_current<=50)::INT) AS cur_top50,
                 SUM((label AND rk_hash<=50)::INT) AS hash_top50,
                 SUM((label AND rk_src<=25)::INT) AS per_src_25,
                 SUM((label AND rk_src<=50)::INT) AS per_src_50,
                 SUM((label AND rk_hash<=100)::INT) AS hash_top100,
                 SUM(label::INT) AS uncapped
               FROM unc_r GROUP BY src)
        SELECT s.src, t.n_truth,
               cur_top50*1.0/n_truth AS recall_current_top50,
               hash_top50*1.0/n_truth AS recall_hashtie_top50,
               per_src_25*1.0/n_truth AS recall_25_per_source,
               per_src_50*1.0/n_truth AS recall_50_per_source,
               hash_top100*1.0/n_truth AS recall_hashtie_top100,
               uncapped*1.0/n_truth AS recall_uncapped
        FROM s JOIN t USING (src) ORDER BY 1""").df())
    show(con.sql("""SELECT src, SUM((rk_current<=50)::INT) AS cur_top50, SUM((rk_hash<=50)::INT) AS hash_top50,
                           SUM((rk_src<=25)::INT) AS per_src_25, COUNT(*) AS uncapped
                    FROM unc_r GROUP BY 1 ORDER BY 1""").df())
    print("candidate-list size per holdout entity, uncapped:", flush=True)
    show(con.sql("""WITH c AS (SELECT source1_entity_id, COUNT(*) n FROM unc GROUP BY 1)
                    SELECT quantile_cont(n,0.1) p10, quantile_cont(n,0.5) p50, quantile_cont(n,0.9) p90,
                           AVG((n>50)::INT) AS frac_over_50 FROM c""").df())
    print("true pairs: shared_tokens vs the entity's 50th-candidate shared_tokens (ties at the cutoff):", flush=True)
    show(con.sql("""
        WITH cut AS (SELECT source1_entity_id, MIN(shared_tokens) AS cutoff FROM unc_r WHERE rk_current<=50 GROUP BY 1)
        SELECT u.src,
               AVG((u.rk_current>50 AND u.shared_tokens > c.cutoff)::INT) AS missed_above_cutoff,
               AVG((u.rk_current>50 AND u.shared_tokens = c.cutoff)::INT) AS missed_tied_at_cutoff,
               AVG((u.rk_current>50 AND u.shared_tokens < c.cutoff)::INT) AS missed_below_cutoff
        FROM unc_r u JOIN cut c USING (source1_entity_id) WHERE u.label GROUP BY 1 ORDER BY 1""").df())
    for c in ent.country.unique():
        con.execute(f"DROP TABLE unc_{c}")
    con.execute("DROP TABLE unc")

    # ---------------- C ----------------
    log("C. holdout threshold behaviour")
    df = pd.read_parquet(f"{OUT}/holdout_scored.parquet")
    df["src"] = df.candidate_entity_id.str[:2]
    truth = con.sql("SELECT * FROM hold_truth").df()
    n_truth_src = truth.src.value_counts()

    def evaluate(pred):
        g = pred.groupby("source1_entity_id").agg(n_pred=("label", "size"), tp=("label", "sum"))
        e = ent.merge(g, left_on="entity_id", right_index=True, how="left").fillna({"n_pred": 0, "tp": 0})
        p = np.where(e.n_pred > 0, e.tp / e.n_pred.clip(lower=1), 0.0)
        r = np.where(e.n_truth > 0, e.tp / e.n_truth.clip(lower=1), 0.0)
        f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r + 1e-12), 0.0)
        f = np.where(e.n_truth == 0, (e.n_pred == 0).astype(float), f)
        return e.assign(P=p, R=r, F=f)

    def summary(e, t, name):
        return dict(variant=name, threshold=t, F05=e.F.mean(),
                    pair_precision=e.tp.sum() / max(e.n_pred.sum(), 1),
                    pair_recall=e.tp.sum() / e.n_truth.sum(),
                    mean_npred_if_any=e.n_pred[e.n_pred > 0].mean(),
                    frac_empty=(e.n_pred == 0).mean(), frac_F0=(e.F == 0).mean())

    grid = np.round(np.concatenate([np.arange(0.05, 0.50, 0.05), np.arange(0.50, 0.9901, 0.01),
                                    [0.992, 0.995, 0.997, 0.999]]), 4)
    rows = [summary(evaluate(df[df.score >= t]), t, "unconstrained") for t in grid]
    sweep0 = pd.DataFrame(rows)
    show(sweep0)
    best0 = float(sweep0.loc[sweep0.F05.idxmax(), "threshold"])
    print(f"BEST unconstrained threshold={best0} F05={sweep0.F05.max():.4f}; at 0.65 F05={sweep0.loc[np.isclose(sweep0.threshold,0.65),'F05'].item():.4f}", flush=True)
    print(f"truth cardinality: mean n_truth if any={ent.n_truth[ent.n_truth>0].mean():.3f}, frac truth-empty={(ent.n_truth==0).mean():.4f}")

    log("C2. S2 vs S3 score distributions and features (holdout candidates)")
    show(df.groupby(["src", "label"]).score.describe(percentiles=[.05, .1, .25, .5, .75, .9]).reset_index())
    show(df.groupby(["src", "label"])[FEATURE_COLUMNS].mean().reset_index())

    def by_source(pred, t, name):
        out = []
        for s in ("S2", "S3"):
            ps, cs = pred[pred.src == s], df[df.src == s]
            out.append(dict(variant=name, threshold=t, src=s, n_cand=len(cs), n_pred=len(ps),
                            accept_rate=len(ps) / len(cs), tp=int(ps.label.sum()), fp=int((~ps.label).sum()),
                            precision=ps.label.mean() if len(ps) else np.nan,
                            recall=ps.label.sum() / n_truth_src.get(s, 1)))
        return out

    # ---------------- D ----------------
    log("D. competitor scoring for record-level one-to-one")
    con.execute(f"""CREATE TABLE hold_recs AS SELECT DISTINCT c.candidate_entity_id
                    FROM read_parquet('{CANDS}') c JOIN hold_ents h ON h.entity_id = c.source1_entity_id""")
    con.execute(f"""
        CREATE TABLE comp AS
        SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens
        FROM read_parquet('{CANDS}') c
        WHERE c.candidate_entity_id IN (SELECT candidate_entity_id FROM hold_recs)
          AND c.source1_entity_id NOT IN (SELECT entity_id FROM hold_ents)""")
    print("holdout records:", con.execute("SELECT COUNT(*) FROM hold_recs").fetchone()[0],
          " competitor pairs:", con.execute("SELECT COUNT(*) FROM comp").fetchone()[0], flush=True)
    joined = con.sql(f"""
        SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens,
               s1.name_norm AS name_norm_1, s23.name_norm AS name_norm_2,
               s1.addr_norm AS addr_norm_1, s23.addr_norm AS addr_norm_2,
               s1.name_tokens AS name_tokens_1, s23.name_tokens AS name_tokens_2,
               s1.addr_tokens AS addr_tokens_1, s23.addr_tokens AS addr_tokens_2
        FROM comp c
        JOIN read_parquet('{S1}') s1 ON s1.entity_id = c.source1_entity_id
        JOIN (SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{S2}')
              UNION ALL
              SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{S3}')) s23
          ON s23.entity_id = c.candidate_entity_id""")
    booster = lgb.Booster(model_file=f"{P}/_model_checkpoint.txt")
    reader = joined.fetch_arrow_reader(batch_size=200_000)
    scored, buf, done = [], [], 0

    def flush(buf):
        res = []
        for feats in pool.map(_feat, buf):
            res.append(feats[["source1_entity_id", "candidate_entity_id"]].assign(score=booster.predict(feats[FEATURE_COLUMNS])))
        return res

    for b in reader:
        c = b.to_pandas()
        if len(c):
            buf.append(c)
        if len(buf) == 16:
            scored += flush(buf)
            done += sum(len(x) for x in buf)
            buf = []
            print(f"  competitor pairs featurized: {done}  ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        scored += flush(buf)
    comp_scored = pd.concat(scored, ignore_index=True)
    comp_scored.to_parquet(f"{OUT}/competitor_scored.parquet")
    pool.close()

    allp = pd.concat([df[["source1_entity_id", "candidate_entity_id", "score"]], comp_scored], ignore_index=True)
    allp = allp.sort_values(["candidate_entity_id", "score"], ascending=[True, False], ignore_index=True)
    allp["rk"] = allp.groupby("candidate_entity_id").cumcount()
    first = allp[allp.rk == 0].set_index("candidate_entity_id").score
    second = allp[allp.rk == 1].set_index("candidate_entity_id").score
    allp["rec_max"] = allp.candidate_entity_id.map(first)
    allp["rec_second"] = allp.candidate_entity_id.map(second).fillna(-1.0)
    h = allp[allp.source1_entity_id.isin(hold_set)].merge(
        df[["source1_entity_id", "candidate_entity_id", "label", "src"]], on=["source1_entity_id", "candidate_entity_id"])
    h["is_max"] = h.score >= h.rec_max
    del allp

    rows = []
    for t in grid:
        base = h[h.score >= t]
        dup = np.where(base.is_max, base.rec_second >= t, True)
        e1 = evaluate(base[base.is_max])
        s = summary(e1, t, "record_argmax")
        s["dup_rate_unconstrained"] = dup.mean() if len(base) else np.nan
        s["prec_dup_pairs"] = base.label[dup].mean() if dup.any() else np.nan
        s["prec_nondup_pairs"] = base.label[~dup].mean() if (~dup).any() else np.nan
        rows.append(s)
    sweep1 = pd.DataFrame(rows)
    show(sweep1)
    best1 = float(sweep1.loc[sweep1.F05.idxmax(), "threshold"])
    print(f"BEST record_argmax threshold={best1} F05={sweep1.F05.max():.4f}  vs unconstrained best {sweep0.F05.max():.4f}", flush=True)

    # ---------------- E ----------------
    log("E. per-source behaviour and TP/FP/FN causes at key thresholds")
    owner = con.sql("""SELECT g.matched_entity_id AS candidate_entity_id, g.source1_entity_id AS owner
                       FROM gt_pairs g JOIN hold_recs r ON r.candidate_entity_id = g.matched_entity_id""").df()
    h = h.merge(owner, on="candidate_entity_id", how="left")
    cand_keys = set(zip(df.source1_entity_id, df.candidate_entity_id))
    truth["in_cands"] = [k in cand_keys for k in zip(truth.source1_entity_id, truth.matched_entity_id)]
    sc = h.set_index(["source1_entity_id", "candidate_entity_id"])
    src_rows = []
    for t, name in [(0.65, "unconstrained"), (best0, "unconstrained"), (best1, "record_argmax"), (best0, "record_argmax")]:
        pred = h[(h.score >= t) & (h.is_max if name == "record_argmax" else True)]
        src_rows += by_source(pred, t, name)
        fp = pred[~pred.label]
        fp_cls = np.where(fp.owner.isna(), "record has no S1 owner (noise)", "record belongs to another S1")
        print(f"\n--- {name} @ {t:.3f}  FP={len(fp)} TP={int(pred.label.sum())} ---", flush=True)
        show(pd.crosstab(fp.src, fp_cls, margins=True).reset_index())
        tr = truth.copy()
        k = list(zip(tr.source1_entity_id, tr.matched_entity_id))
        s_ = sc.reindex(k)
        tr["score"] = s_.score.values
        tr["is_max"] = s_.is_max.values
        cause = np.select(
            [~tr.in_cands, tr.score < t, (name == "record_argmax") & (tr.is_max == False)],
            ["missed by blocking", "scored below threshold", "removed by dedup"], "found (TP)")
        show(pd.crosstab(tr.src, cause, margins=True).reset_index())
    log("per-source summary")
    show(pd.DataFrame(src_rows))
    log("done")
