"""Forensic full-scale evaluation of the checkpointed model.

Realistic validation: every candidate kept for each held-out entity (no
negative subsampling), blocked against the FULL train S2/S3. Also measures
full-scale blocking recall and the perfect-classifier ceiling.

Caveat: the checkpointed model was trained on ~80% of sampled train
entities, so some holdout entities were seen in training -> numbers here are
mildly optimistic, not pessimistic.
"""
import os
import pickle
import sys
import time

sys.path.insert(0, "/home/ubuntu/amazon/code/business_entity_resolution")
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import blocking, features
from src.features import FEATURE_COLUMNS

P = "/home/ubuntu/amazon/dataset/processed"
GT = "/home/ubuntu/amazon/dataset_meta/train_ground_truth.tsv"
S1, S2, S3 = f"{P}/train_source1.parquet", f"{P}/train_source2.parquet", f"{P}/train_source3.parquet"
CANDS = f"{P}/_eval_train_candidates.parquet"
HOLD_FEATS = f"{P}/_eval_holdout_features.parquet"
OUT = "/home/ubuntu/amazon/eval_out"
os.makedirs(OUT, exist_ok=True)
N_HOLDOUT = 40_000

t0 = time.time()
con = duckdb.connect()
con.execute("SET memory_limit='46GB'")

if not os.path.exists(CANDS):
    blocking.generate_candidates(con, S1, S2, S3, top_k=50).write_parquet(CANDS)
print(f"[{time.time()-t0:.0f}s] train candidates ready")

con.execute(f"""
    CREATE TABLE gt_pairs AS
    SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
    FROM read_csv('{GT}', delim='\t', header=true, quote='"', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
""")
con.execute(f"CREATE TABLE s1 AS SELECT entity_id, country FROM read_parquet('{S1}')")
con.execute("""
    CREATE TABLE truth_n AS
    SELECT s1.entity_id, s1.country, COUNT(g.matched_entity_id) AS n_truth
    FROM s1 LEFT JOIN gt_pairs g ON g.source1_entity_id = s1.entity_id
    GROUP BY 1, 2
""")

print("\n=== FULL-SCALE BLOCKING (all train S1) ===")
print(con.sql(f"""
    SELECT COUNT(*) AS n_pairs, COUNT(DISTINCT source1_entity_id) AS n_s1_with_cands
    FROM read_parquet('{CANDS}')""").df().to_string(index=False))
con.execute(f"""
    CREATE TABLE found AS
    SELECT g.source1_entity_id, COUNT(*) AS n_found
    FROM gt_pairs g JOIN read_parquet('{CANDS}') c
      ON c.source1_entity_id = g.source1_entity_id AND c.candidate_entity_id = g.matched_entity_id
    GROUP BY 1
""")
print(con.sql("""
    WITH e AS (
        SELECT t.entity_id, t.country, t.n_truth, COALESCE(f.n_found, 0) AS n_found
        FROM truth_n t LEFT JOIN found f ON f.source1_entity_id = t.entity_id
    )
    SELECT country,
           SUM(n_found)*1.0/NULLIF(SUM(n_truth),0) AS pair_recall,
           AVG(CASE WHEN n_truth>0 AND n_found=0 THEN 1 ELSE 0 END) AS frac_entities_all_missed,
           AVG(CASE WHEN n_truth=0 THEN 1.0
                    WHEN n_found=0 THEN 0.0
                    ELSE 1.25*(n_found*1.0/n_truth)/(0.25+(n_found*1.0/n_truth)) END) AS perfect_classifier_ceiling
    FROM e GROUP BY ROLLUP(country) ORDER BY country NULLS LAST
""").df().to_string(index=False))
print("\ncandidates per S1 entity (train):")
print(con.sql(f"""
    WITH c AS (SELECT source1_entity_id, COUNT(*) n FROM read_parquet('{CANDS}') GROUP BY 1)
    SELECT quantile_cont(n,0.1) p10, quantile_cont(n,0.5) p50, quantile_cont(n,0.9) p90,
           AVG((n=50)::INT) AS frac_hitting_top50_cap FROM c
""").df().to_string(index=False))
print(f"[{time.time()-t0:.0f}s] blocking stats done")

con.execute(f"""
    CREATE TABLE holdout AS
    SELECT entity_id, country FROM s1 USING SAMPLE reservoir({N_HOLDOUT} ROWS) REPEATABLE (7)
""")
hold_cands = con.sql(f"""
    SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens
    FROM read_parquet('{CANDS}') c JOIN holdout h ON h.entity_id = c.source1_entity_id
""")
with open(f"{P}/_vec_checkpoint.pkl", "rb") as f:
    vec = pickle.load(f)
n = features.compute_features(con, hold_cands, S1, S2, S3, HOLD_FEATS, vec)
print(f"[{time.time()-t0:.0f}s] holdout features: {n} pairs")

booster = lgb.Booster(model_file=f"{P}/_model_checkpoint.txt")
df = pd.read_parquet(HOLD_FEATS)
df["score"] = booster.predict(df[FEATURE_COLUMNS])
gt = con.sql("SELECT g.source1_entity_id, g.matched_entity_id FROM gt_pairs g JOIN holdout h ON h.entity_id = g.source1_entity_id").df()
gt_set = set(zip(gt.source1_entity_id, gt.matched_entity_id))
df["label"] = [(a, b) in gt_set for a, b in zip(df.source1_entity_id, df.candidate_entity_id)]
ent = con.sql("SELECT t.entity_id, t.country, t.n_truth FROM truth_n t JOIN holdout h USING (entity_id)").df()
df.to_parquet(f"{OUT}/holdout_scored.parquet")
ent.to_parquet(f"{OUT}/holdout_entities.parquet")


def per_entity(t):
    d = df.assign(pred=df.score >= t)
    d = d.assign(tp=d.pred & d.label)
    g = d.groupby("source1_entity_id").agg(n_pred=("pred", "sum"), tp=("tp", "sum"))
    e = ent.merge(g, left_on="entity_id", right_index=True, how="left").fillna({"n_pred": 0, "tp": 0})
    p = np.where(e.n_pred > 0, e.tp / e.n_pred.clip(lower=1), 0.0)
    r = np.where(e.n_truth > 0, e.tp / e.n_truth.clip(lower=1), 0.0)
    f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r + 1e-12), 0.0)
    f = np.where(e.n_truth == 0, (e.n_pred == 0).astype(float), f)
    e = e.assign(P=p, R=r, F=f)
    return e


print("\n=== REALISTIC HOLDOUT F_0.5 (all candidates kept) ===")
grid = np.concatenate([np.linspace(0.05, 0.90, 18), np.linspace(0.90, 0.995, 20), [0.997, 0.999]])
rows = []
for t in grid:
    e = per_entity(t)
    rows.append((t, e.F.mean(), e.P[e.n_pred > 0].mean(), e.R[e.n_truth > 0].mean(), (e.n_pred > 0).mean()))
sweep = pd.DataFrame(rows, columns=["threshold", "F05", "mean_P_when_pred", "mean_R_when_truth", "frac_pred_nonempty"])
print(sweep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
best = sweep.loc[sweep.F05.idxmax()]
print(f"\nBEST threshold={best.threshold:.4f} F05={best.F05:.4f}")

for label_t in (0.65, float(best.threshold)):
    e = per_entity(label_t)
    print(f"\n--- threshold={label_t:.3f}: F05={e.F.mean():.4f} ---")
    print(e.groupby("country").F.mean().to_string())
    zero = e[e.F == 0]
    print(f"entities scoring 0: {len(zero)/len(e):.4f}")
    print(f"  truth>0, predicted empty:     {((e.n_truth>0)&(e.n_pred==0)).mean():.4f}")
    print(f"  truth>0, predicted, 0 correct:{((e.n_truth>0)&(e.n_pred>0)&(e.tp==0)).mean():.4f}")
    print(f"  truth=0, predicted non-empty: {((e.n_truth==0)&(e.n_pred>0)).mean():.4f}")
    print("  # predicted per entity:", (e.n_pred.clip(upper=10).value_counts(normalize=True).sort_index().round(4).to_dict()))

print("\n=== SCORE DISTRIBUTION: positives vs negatives (holdout) ===")
print(df.groupby("label").score.describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())
print(f"\npositives per entity (in candidates) mean={df.groupby('source1_entity_id').label.sum().mean():.3f}; "
      f"negatives per entity mean={(~df.label).groupby(df.source1_entity_id).sum().mean():.3f}")
print(f"[{time.time()-t0:.0f}s] done")
