import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='5GB'")
D = "/Users/pravin/Desktop/Amazon"
for name, path in [("train_gt", f"{D}/dataset/train/train_ground_truth.tsv"), ("test_pred", f"{D}/output/matching_results.tsv")]:
    print(f"=== {name}: how many S1 entities claim each S2/S3 record ===")
    print(con.sql(f"""
      WITH u AS (SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS id
                 FROM read_csv('{path}', delim='\t', header=true, quote='"', all_varchar=true)
                 WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''),
           c AS (SELECT id, COUNT(DISTINCT source1_entity_id) AS k FROM u GROUP BY id)
      SELECT CASE WHEN k>=5 THEN '5+' ELSE k::VARCHAR END AS n_s1_claiming, COUNT(*) AS n_records,
             ROUND(COUNT(*)*1.0/SUM(COUNT(*)) OVER (),4) AS frac
      FROM c GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
print("\n=== test: S1 entities predicted empty -- had candidates or not? ===")
print(con.sql(f"""
  SELECT (c.candidate_entity_ids IS NULL OR c.candidate_entity_ids='') AS no_candidates, COUNT(*)
  FROM read_csv('{D}/output/matching_results.tsv', delim='\t', header=true, quote='"', all_varchar=true) m
  JOIN read_csv('{D}/output/candidate_pairs.tsv', delim='\t', header=true, quote='"', all_varchar=true) c USING (source1_entity_id)
  WHERE m.matched_entity_ids IS NULL OR m.matched_entity_ids=''
  GROUP BY 1""").df().to_string(index=False))
print("\n=== record counts: train vs test ===")
for s in ["train_source1","train_source2","train_source3","test_source1","test_source2","test_source3"]:
    r = con.sql(f"SELECT country, COUNT(*) FROM read_parquet('{D}/dataset/processed/{s}.parquet') GROUP BY 1 ORDER BY 1").fetchall()
    print(s, r)
