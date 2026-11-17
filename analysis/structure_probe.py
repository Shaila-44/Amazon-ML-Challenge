import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='5GB'"); con.execute("SET threads=4")
D = "/Users/pravin/Desktop/Amazon/dataset"
con.execute(f"""CREATE TABLE gt AS SELECT source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) rec
    FROM read_csv('{D}/train/train_ground_truth.tsv', delim='\t', header=true, quote='"', all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''""")
print("=== (a) per-entity match counts by source (entities with >=1 match) ===")
print(con.sql("""WITH c AS (SELECT s1, SUM((left(rec,2)='S2')::INT) n2, SUM((left(rec,2)='S3')::INT) n3 FROM gt GROUP BY 1)
  SELECT n2, n3, COUNT(*) n, ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER (),2) pct FROM c GROUP BY 1,2 ORDER BY n DESC LIMIT 16""").df().to_string(index=False))

print("\n=== (b) file row-order: are consecutive S2 rows owned by the same S1? ===")
con.execute("SET preserve_insertion_order=true")
for src in ("train_source2", "train_source3"):
    con.execute(f"CREATE OR REPLACE TABLE r AS SELECT entity_id, ROW_NUMBER() OVER () rn FROM read_csv('{D}/train/{src}.tsv', delim='\t', header=true, quote='\"', all_varchar=true)")
    print(src, con.sql("""WITH o AS (SELECT r.rn, g.s1 FROM r LEFT JOIN gt g ON g.rec = r.entity_id),
        p AS (SELECT s1, LEAD(s1) OVER (ORDER BY rn) nxt FROM o)
        SELECT AVG((s1 = nxt)::INT) same_owner_next_row, COUNT(*) n FROM p WHERE s1 IS NOT NULL AND nxt IS NOT NULL""").fetchall())

print("\n=== (c) ID numbers: same-entity record IDs closer than random? ===")
print(con.sql("""WITH x AS (SELECT s1, TRY_CAST(split_part(rec,'-',2) AS BIGINT) id FROM gt WHERE left(rec,2)='S2'),
  pairs AS (SELECT a.id ia, b.id ib FROM x a JOIN x b ON a.s1=b.s1 AND a.id<b.id USING SAMPLE 200000 ROWS),
  rnd AS (SELECT a.id ia, b.id ib FROM (SELECT id, ROW_NUMBER() OVER () k FROM x USING SAMPLE 200000 ROWS) a
          JOIN (SELECT id, ROW_NUMBER() OVER () k FROM x USING SAMPLE 200000 ROWS) b USING (k))
  SELECT (SELECT median(abs(ia-ib)) FROM pairs) same_entity_median_gap, (SELECT median(abs(ia-ib)) FROM rnd) random_median_gap""").fetchall())
print("S1 id vs owned S2 id correlation:", con.sql("""SELECT corr(TRY_CAST(split_part(s1,'-',2) AS DOUBLE), TRY_CAST(split_part(rec,'-',2) AS DOUBLE)) FROM gt""").fetchall())

print("\n=== (d) exact-duplicate records (same name+address) across S2/S3: same owner? ===")
print(con.sql(f"""WITH recs AS (
    SELECT entity_id, lower(trim(business_name)) n, lower(trim(coalesce(business_address,''))) a FROM read_parquet('{D}/processed/train_source2.parquet')
    UNION ALL SELECT entity_id, lower(trim(business_name)), lower(trim(coalesce(business_address,''))) FROM read_parquet('{D}/processed/train_source3.parquet')),
  o AS (SELECT r.*, g.s1 FROM recs r LEFT JOIN gt g ON g.rec = r.entity_id),
  grp AS (SELECT n, a, COUNT(*) k, COUNT(DISTINCT s1) owners, COUNT(s1) owned FROM o WHERE a != '' GROUP BY 1,2 HAVING COUNT(*) > 1)
  SELECT COUNT(*) dup_groups, SUM(k) records_in_dup_groups, AVG((owners=1 AND owned=k)::INT) frac_groups_single_owner_all_owned,
         AVG((owned=0)::INT) frac_groups_all_unowned FROM grp""").fetchall())
