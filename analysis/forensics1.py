import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='5GB'")
D = "/Users/pravin/Desktop/Amazon"

def nmatch(col):
    return f"CASE WHEN {col} IS NULL OR {col}='' THEN 0 ELSE len(string_split({col}, ',')) END"

con.execute(f"""
CREATE TABLE gt AS
SELECT s.entity_id, s.country, {nmatch('g.matched_entity_ids')} AS n
FROM read_parquet('{D}/dataset/processed/train_source1.parquet') s
LEFT JOIN read_csv('{D}/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='"', all_varchar=true) g
  ON g.source1_entity_id = s.entity_id
""")
con.execute(f"""
CREATE TABLE pr AS
SELECT s.entity_id, s.country, {nmatch('m.matched_entity_ids')} AS n
FROM read_parquet('{D}/dataset/processed/test_source1.parquet') s
LEFT JOIN read_csv('{D}/output/matching_results.tsv', delim='\t', header=true, quote='"', all_varchar=true) m
  ON m.source1_entity_id = s.entity_id
""")

q = """
SELECT country, COUNT(*) AS entities,
       ROUND(AVG((n>0)::INT),4) AS frac_with_match,
       ROUND(AVG(n),3) AS mean_n,
       ROUND(AVG(n) FILTER (WHERE n>0),3) AS mean_n_if_any,
       quantile_cont(n, 0.5) FILTER (WHERE n>0) AS median_n_if_any
FROM {t} GROUP BY ROLLUP(country) ORDER BY country NULLS LAST
"""
print("=== TRAIN GROUND TRUTH (per S1 entity) ===")
print(con.sql(q.format(t="gt")).df().to_string(index=False))
print("\n=== TEST PREDICTIONS (per S1 entity) ===")
print(con.sql(q.format(t="pr")).df().to_string(index=False))

hist = """
SELECT CASE WHEN n>=10 THEN '10+' ELSE lpad(n::VARCHAR,2,' ') END AS n_matches,
       ROUND(COUNT(*)*1.0/SUM(COUNT(*)) OVER (),4) AS frac
FROM {t} GROUP BY 1 ORDER BY 1
"""
g = con.sql(hist.format(t="gt")).df().rename(columns={"frac":"train_gt"})
p = con.sql(hist.format(t="pr")).df().rename(columns={"frac":"test_pred"})
print("\n=== DISTRIBUTION OF # MATCHES PER ENTITY ===")
print(g.merge(p, on="n_matches", how="outer").fillna(0).sort_values("n_matches").to_string(index=False))

print("\n=== S2 vs S3 split in matches ===")
for name, path, col in [("train_gt", f"{D}/dataset/train/train_ground_truth.tsv", "matched_entity_ids"),
                        ("test_pred", f"{D}/output/matching_results.tsv", "matched_entity_ids")]:
    r = con.sql(f"""
      WITH u AS (SELECT unnest(string_split({col}, ',')) AS id
                 FROM read_csv('{path}', delim='\t', header=true, quote='"', all_varchar=true)
                 WHERE {col} IS NOT NULL AND {col} != '')
      SELECT left(id,2) AS src, COUNT(*) FROM u GROUP BY 1 ORDER BY 1""").fetchall()
    print(name, r)
