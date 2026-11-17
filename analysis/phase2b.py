"""Phase 2b: record-level one-to-one (dedup) evaluation + TP/FP/FN causes on
the 40k holdout. Competitors = every S1 entity (outside the holdout) whose
candidate list contains a record that is also a holdout candidate."""
import sys
import time

sys.path.insert(0, "/home/ubuntu/amazon")
from par_score import make_pool, score_parallel

P = "/home/ubuntu/amazon/dataset/processed"
OUT = "/home/ubuntu/amazon/eval_out"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
S1, S2, S3 = f"{P}/train_source1.parquet", f"{P}/train_source2.parquet", f"{P}/train_source3.parquet"
CANDS = f"{P}/_eval_train_candidates.parquet"

if __name__ == "__main__":
    pool = make_pool(f"{P}/_vec_checkpoint.pkl", f"{P}/_model_checkpoint.txt")

    import os
    import duckdb
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 250)
    t0 = time.time()

    def log(m):
        print(f"\n[{time.time()-t0:.0f}s] ===== {m} =====", flush=True)

    def show(x):
        print(x.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='40GB'")
    con.execute(f"""
        CREATE TABLE gt_pairs AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
        FROM read_csv('{GT}', delim='\t', header=true, quote='"', all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
    ent = pd.read_parquet(f"{OUT}/holdout_entities.parquet")
    con.register("hold_df", ent[["entity_id", "country"]])
    con.execute("CREATE TABLE hold_ents AS SELECT * FROM hold_df")
    hold_set = set(ent.entity_id)
    con.execute(f"""CREATE TABLE hold_recs AS SELECT DISTINCT c.candidate_entity_id
                    FROM read_parquet('{CANDS}') c JOIN hold_ents h ON h.entity_id = c.source1_entity_id""")
    comp_path = f"{OUT}/competitor_pairs.parquet"
    if not os.path.exists(comp_path):
        con.execute(f"""
            COPY (SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens
                  FROM read_parquet('{CANDS}') c
                  WHERE c.candidate_entity_id IN (SELECT candidate_entity_id FROM hold_recs)
                    AND c.source1_entity_id NOT IN (SELECT entity_id FROM hold_ents))
            TO '{comp_path}' (FORMAT PARQUET)""")
    truth = con.sql("""SELECT g.source1_entity_id, g.matched_entity_id, left(g.matched_entity_id,2) AS src
                       FROM gt_pairs g JOIN hold_ents h ON h.entity_id = g.source1_entity_id""").df()
    owner = con.sql("""SELECT g.matched_entity_id AS candidate_entity_id, g.source1_entity_id AS owner
                       FROM gt_pairs g JOIN hold_recs r ON r.candidate_entity_id = g.matched_entity_id""").df()
    con.close()

    log("D. scoring competitor pairs (parallel)")
    comp_dir = f"{OUT}/competitor_scored"
    n = score_parallel(pool, f"read_parquet('{comp_path}')", S1, S2, S3, comp_dir, "/home/ubuntu/amazon/.tmp_comp_shards")
    pool.close()
    print(f"competitor pairs scored: {n}", flush=True)

    df = pd.read_parquet(f"{OUT}/holdout_scored.parquet",
                         columns=["source1_entity_id", "candidate_entity_id", "score", "label"])
    df["src"] = df.candidate_entity_id.str[:2]
    comp = duckdb.sql(f"SELECT source1_entity_id, candidate_entity_id, score FROM read_parquet('{comp_dir}/*.parquet')").df()
    allp = pd.concat([df[["source1_entity_id", "candidate_entity_id", "score"]], comp], ignore_index=True)
    del comp
    allp = allp.sort_values(["candidate_entity_id", "score"], ascending=[True, False], ignore_index=True)
    allp["rk"] = allp.groupby("candidate_entity_id").cumcount()
    first = allp[allp.rk == 0].set_index("candidate_entity_id").score
    second = allp[allp.rk == 1].set_index("candidate_entity_id").score
    h = allp[allp.source1_entity_id.isin(hold_set)].copy()
    del allp
    h["rec_max"] = h.candidate_entity_id.map(first)
    h["rec_second"] = h.candidate_entity_id.map(second).fillna(-1.0)
    h = h.merge(df[["source1_entity_id", "candidate_entity_id", "label", "src"]],
                on=["source1_entity_id", "candidate_entity_id"])
    h["is_max"] = h.score >= h.rec_max
    h = h.merge(owner, on="candidate_entity_id", how="left")
    print(f"holdout pairs with competitor context: {len(h)} (holdout scored pairs: {len(df)})", flush=True)

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

    log("D. unconstrained vs record-level argmax (global, ties kept)")
    grid = np.round(np.concatenate([np.arange(0.05, 0.50, 0.05), np.arange(0.50, 0.9901, 0.01),
                                    [0.992, 0.995, 0.997, 0.999]]), 4)
    rows = []
    for t in grid:
        base = h[h.score >= t]
        dup = np.where(base.is_max, base.rec_second >= t, True)
        r0 = summary(evaluate(base), t, "unconstrained")
        r1 = summary(evaluate(base[base.is_max]), t, "record_argmax")
        rows.append(dict(threshold=t, F05_unconstrained=r0["F05"], F05_argmax=r1["F05"],
                         delta=r1["F05"] - r0["F05"],
                         prec_unc=r0["pair_precision"], prec_argmax=r1["pair_precision"],
                         rec_unc=r0["pair_recall"], rec_argmax=r1["pair_recall"],
                         dup_rate=dup.mean() if len(base) else np.nan,
                         prec_dup_pairs=base.label[dup].mean() if dup.any() else np.nan,
                         prec_nondup_pairs=base.label[~dup].mean() if (~dup).any() else np.nan,
                         exact_ties_kept=int((base.is_max & (base.rec_second == base.rec_max)).sum())))
    sw = pd.DataFrame(rows)
    show(sw)
    b0 = float(sw.loc[sw.F05_unconstrained.idxmax(), "threshold"])
    b1 = float(sw.loc[sw.F05_argmax.idxmax(), "threshold"])
    print(f"BEST unconstrained: t={b0} F05={sw.F05_unconstrained.max():.4f}", flush=True)
    print(f"BEST record_argmax: t={b1} F05={sw.F05_argmax.max():.4f}", flush=True)

    log("E. TP / FP / FN causes and per-source behaviour")
    cand_keys = set(zip(df.source1_entity_id, df.candidate_entity_id))
    truth["in_cands"] = [k in cand_keys for k in zip(truth.source1_entity_id, truth.matched_entity_id)]
    n_truth_src = truth.src.value_counts()
    sc = h.set_index(["source1_entity_id", "candidate_entity_id"])
    src_rows = []
    for t, name in [(0.65, "unconstrained"), (b0, "unconstrained"), (b0, "record_argmax"), (b1, "record_argmax")]:
        pred = h[(h.score >= t) & (h.is_max if name == "record_argmax" else True)]
        for s in ("S2", "S3"):
            ps, cs = pred[pred.src == s], df[df.src == s]
            src_rows.append(dict(variant=name, threshold=t, src=s, n_cand=len(cs), n_pred=len(ps),
                                 accept_rate=len(ps) / len(cs), tp=int(ps.label.sum()), fp=int((~ps.label).sum()),
                                 precision=ps.label.mean() if len(ps) else np.nan,
                                 recall=ps.label.sum() / n_truth_src.get(s, 1)))
        fp = pred[~pred.label]
        cls = np.where(fp.owner.isna(), "no S1 owner (noise record)", "owned by another S1")
        print(f"\n--- {name} @ {t:.3f}: TP={int(pred.label.sum())} FP={len(fp)} ---", flush=True)
        show(pd.crosstab(fp.src, cls, margins=True).reset_index())
        tr = truth.copy()
        s_ = sc.reindex(list(zip(tr.source1_entity_id, tr.matched_entity_id)))
        sc_v, mx_v = s_.score.values, s_.is_max.values
        cause = np.select(
            [~tr.in_cands.values, sc_v < t, (name == "record_argmax") & (mx_v == False)],
            ["missed by blocking", "scored below threshold", "removed by dedup"], "found (TP)")
        show(pd.crosstab(tr.src, cause, margins=True).reset_index())
    log("per-source summary")
    show(pd.DataFrame(src_rows))
    log("done")
