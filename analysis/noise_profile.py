import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='5GB'"); con.execute("SET threads=4")
D = "/Users/pravin/Desktop/Amazon/dataset"; EV = "/private/tmp/claude-501/-Users-pravin-Desktop-Amazon/a6a16f51-691f-42fd-b8a3-7b4177ca9008/scratchpad/ev"
con.execute(f"""CREATE TABLE gt AS SELECT source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) rec
    FROM read_csv('{D}/train/train_ground_truth.tsv', delim='\t', header=true, quote='"', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
con.execute(f"""CREATE TABLE tp AS SELECT g.s1, g.rec, left(g.rec,2) src, h.country, (k.rk IS NOT NULL) found_k200
    FROM gt g JOIN read_parquet('{EV}/holdout_entities.parquet') h ON h.entity_id = g.s1
    LEFT JOIN read_parquet('{EV}/k_cands_top200.parquet') k ON k.source1_entity_id = g.s1 AND k.candidate_entity_id = g.rec""")
con.execute(f"""CREATE TABLE r AS
    SELECT entity_id, business_name, business_address, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{D}/processed/train_source2.parquet') WHERE entity_id IN (SELECT rec FROM tp)
    UNION ALL SELECT entity_id, business_name, business_address, name_norm, addr_norm, name_tokens, addr_tokens FROM read_parquet('{D}/processed/train_source3.parquet') WHERE entity_id IN (SELECT rec FROM tp)""")
con.execute(f"""CREATE TABLE x AS SELECT t.*, s.business_name n1, s.business_address a1, r.business_name n2, r.business_address a2,
      s.name_norm nn1, r.name_norm nn2, s.addr_norm an1, r.addr_norm an2,
      len(list_intersect(s.name_tokens, r.name_tokens)) name_shared, len(list_intersect(s.addr_tokens, r.addr_tokens)) addr_shared,
      regexp_matches(r.business_name, '[\\x{{0900}}-\\x{{0DFF}}]') rec_name_indic, regexp_matches(coalesce(r.business_address,''), '[\\x{{0900}}-\\x{{0DFF}}]') rec_addr_indic,
      regexp_matches(lower(r.business_name), '(\\.(com|in|net|org|co|biz|info|fr)\\b|www)') rec_name_domain,
      (r.business_address IS NULL OR trim(r.business_address) = '') rec_addr_missing,
      jaro_winkler_similarity(coalesce(s.name_norm,''), coalesce(r.name_norm,'')) name_jw,
      jaro_winkler_similarity(coalesce(s.addr_norm,''), coalesce(r.addr_norm,'')) addr_jw
    FROM tp t JOIN read_parquet('{D}/processed/train_source1.parquet') s ON s.entity_id = t.s1 JOIN r ON r.entity_id = t.rec""")
print("true pairs:", con.execute("SELECT COUNT(*), AVG(found_k200::INT) FROM x").fetchall())
q = """SELECT {g} AS grp, COUNT(*) n, ROUND(AVG((NOT found_k200)::INT),3) miss_rate_k200,
  ROUND(AVG(rec_name_indic::INT),3) name_indic, ROUND(AVG(rec_addr_indic::INT),3) addr_indic, ROUND(AVG(rec_name_domain::INT),3) name_domain,
  ROUND(AVG(rec_addr_missing::INT),3) addr_missing, ROUND(AVG((name_shared=0)::INT),3) no_name_token, ROUND(AVG((addr_shared=0)::INT),3) no_addr_token,
  ROUND(AVG((name_shared=0 AND addr_shared=0)::INT),3) no_token_at_all, ROUND(AVG((nn1=nn2)::INT),3) name_exact, ROUND(AVG((an1=an2)::INT),3) addr_exact,
  ROUND(AVG(name_jw),3) name_jw, ROUND(AVG(addr_jw),3) addr_jw FROM x GROUP BY 1 ORDER BY 1"""
print(con.sql(q.format(g="found_k200")).df().to_string(index=False))
print(con.sql(q.format(g="src || ' ' || country")).df().to_string(index=False))
print("\n=== what the MISSED pairs look like: share by noise pattern ===")
print(con.sql("""SELECT CASE
    WHEN rec_name_domain THEN 'name is a domain'
    WHEN rec_name_indic AND (rec_addr_missing OR rec_addr_indic) THEN 'indic name + no/indic address'
    WHEN rec_name_indic THEN 'indic name, latin address'
    WHEN rec_addr_missing AND name_shared = 0 THEN 'no address + no shared name token'
    WHEN rec_addr_missing THEN 'no address, shares name token(s)'
    WHEN name_shared = 0 AND addr_shared = 0 THEN 'latin, no shared token at all'
    WHEN name_shared = 0 THEN 'name totally different, address shared'
    ELSE 'shares name AND/OR address tokens' END pattern,
    COUNT(*) n_missed, ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER (),1) pct_of_missed,
    ROUND(AVG(name_jw),3) name_jw, ROUND(AVG(addr_jw),3) addr_jw
  FROM x WHERE NOT found_k200 GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))
con.execute("COPY (SELECT * FROM x WHERE NOT found_k200 USING SAMPLE 400 ROWS) TO '" + EV + "/missed_sample.parquet' (FORMAT PARQUET)")
