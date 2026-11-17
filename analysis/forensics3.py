import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='5GB'")
D = "/Users/pravin/Desktop/Amazon"
def pairs(path, col):
    return f"""SELECT m.source1_entity_id, unnest(string_split(m.{col}, ',')) AS id
               FROM read_csv('{path}', delim='\t', header=true, quote='"', all_varchar=true) m
               WHERE m.{col} IS NOT NULL AND m.{col} != ''"""
con.execute(f"CREATE TABLE s1 AS SELECT entity_id, country FROM read_parquet('{D}/dataset/processed/test_source1.parquet')")
con.execute(f"CREATE TABLE tc AS {pairs(D+'/output/candidate_pairs.tsv','candidate_entity_ids')}")
con.execute(f"CREATE TABLE tp AS {pairs(D+'/output/matching_results.tsv','matched_entity_ids')}")
print("=== TEST: candidates vs accepted, by country x source ===")
print(con.sql("""
  WITH c AS (SELECT s.country, left(id,2) src, COUNT(*) n_cand FROM tc JOIN s1 s ON s.entity_id=tc.source1_entity_id GROUP BY 1,2),
       p AS (SELECT s.country, left(id,2) src, COUNT(*) n_pred FROM tp JOIN s1 s ON s.entity_id=tp.source1_entity_id GROUP BY 1,2),
       r AS (SELECT country, left(entity_id,2) src, COUNT(*) n_records FROM (
               SELECT entity_id, country FROM read_parquet('""" + D + """/dataset/processed/test_source2.parquet')
               UNION ALL SELECT entity_id, country FROM read_parquet('""" + D + """/dataset/processed/test_source3.parquet')) GROUP BY 1,2)
  SELECT c.country, c.src, r.n_records, c.n_cand, p.n_pred,
         ROUND(p.n_pred*1.0/c.n_cand,4) AS accept_rate,
         ROUND(p.n_pred*1.0/r.n_records,4) AS pred_per_record
  FROM c JOIN p USING (country, src) JOIN r USING (country, src) ORDER BY 1,2""").df().to_string(index=False))
print("\n=== TRAIN GT: true matches by country x source (fraction of records matched) ===")
con.execute(f"CREATE TABLE s1tr AS SELECT entity_id, country FROM read_parquet('{D}/dataset/processed/train_source1.parquet')")
print(con.sql(f"""
  WITH g AS ({pairs(D+'/dataset/train/train_ground_truth.tsv','matched_entity_ids')}),
       m AS (SELECT s.country, left(id,2) src, COUNT(*) n_true FROM g JOIN s1tr s ON s.entity_id=g.source1_entity_id GROUP BY 1,2),
       r AS (SELECT country, left(entity_id,2) src, COUNT(*) n_records FROM (
               SELECT entity_id, country FROM read_parquet('{D}/dataset/processed/train_source2.parquet')
               UNION ALL SELECT entity_id, country FROM read_parquet('{D}/dataset/processed/train_source3.parquet')) GROUP BY 1,2)
  SELECT country, src, n_records, n_true, ROUND(n_true*1.0/n_records,4) AS frac_records_matched
  FROM m JOIN r USING (country, src) ORDER BY 1,2""").df().to_string(index=False))
