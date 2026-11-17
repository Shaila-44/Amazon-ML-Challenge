import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='5GB'"); con.execute("SET threads=4")
D = "/Users/pravin/Desktop/Amazon/dataset"; EV = "/private/tmp/claude-501/-Users-pravin-Desktop-Amazon/a6a16f51-691f-42fd-b8a3-7b4177ca9008/scratchpad/ev"
KEY = "array_to_string(list_sort(list_distinct(name_tokens)), ' ')"
con.execute(f"""CREATE TABLE gt AS SELECT source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) rec
    FROM read_csv('{D}/train/train_ground_truth.tsv', delim='\t', header=true, quote='"', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
con.execute(f"""CREATE TABLE rk AS SELECT entity_id, country, {KEY} k FROM read_parquet('{D}/processed/train_source2.parquet')
                UNION ALL SELECT entity_id, country, {KEY} FROM read_parquet('{D}/processed/train_source3.parquet')""")
con.execute("CREATE TABLE bs AS SELECT country, k, COUNT(*) block FROM rk WHERE k != '' GROUP BY 1, 2")
con.execute(f"""CREATE TABLE t AS SELECT g.s1, g.rec, h.country, (c.rk IS NOT NULL) found_k200,
      {KEY.replace('name_tokens','s.name_tokens')} k1, r.k k2
    FROM gt g JOIN read_parquet('{EV}/holdout_entities.parquet') h ON h.entity_id = g.s1
    JOIN read_parquet('{D}/processed/train_source1.parquet') s ON s.entity_id = g.s1
    JOIN rk r ON r.entity_id = g.rec
    LEFT JOIN read_parquet('{EV}/k_cands_top200.parquet') c ON c.source1_entity_id = g.s1 AND c.candidate_entity_id = g.rec""")
print(con.sql("""SELECT found_k200, COUNT(*) n, ROUND(AVG((k1 = k2 AND k1 != '')::INT),4) sorted_name_key_equal FROM t GROUP BY 1""").df().to_string(index=False))
print(con.sql("""SELECT COUNT(*) n_true,
    ROUND(AVG(found_k200::INT),4) recall_k200_now,
    ROUND(AVG((found_k200 OR (k1 = k2 AND k1 != ''))::INT),4) recall_k200_plus_exact_name_key
  FROM t""").df().to_string(index=False))
print("block sizes of the exact-name key for S1 entities in holdout (how many S2/S3 records share the key):")
print(con.sql("""WITH e AS (SELECT DISTINCT t.s1, t.country, t.k1 FROM t)
  SELECT quantile_cont(COALESCE(b.block,0), [0.5, 0.9, 0.99]) q, AVG((COALESCE(b.block,0) <= 50)::INT) frac_block_le_50, MAX(b.block) mx
  FROM e LEFT JOIN bs b ON b.country = e.country AND b.k = e.k1""").fetchall())
