import duckdb, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
con = duckdb.connect(); con.execute("SET memory_limit='5GB'"); con.execute("SET threads=4")
D = "/Users/pravin/Desktop/Amazon/dataset"; EV = f"{'/private/tmp/claude-501/-Users-pravin-Desktop-Amazon/a6a16f51-691f-42fd-b8a3-7b4177ca9008/scratchpad'}/ev"
m = f"read_parquet('{EV}/missed_sample.parquet')"
# name-token document frequency in S2+S3 (name field), per country -- same df definition the blocker uses
con.execute(f"""CREATE TABLE ndf AS SELECT country, token, COUNT(DISTINCT entity_id) df FROM (
    SELECT entity_id, country, unnest(list_distinct(name_tokens)) token FROM read_parquet('{D}/processed/train_source2.parquet')
    UNION ALL SELECT entity_id, country, unnest(list_distinct(name_tokens)) FROM read_parquet('{D}/processed/train_source3.parquet'))
  WHERE token IN (SELECT unnest(string_split(nn1, ' ')) FROM {m}) GROUP BY 1, 2""")
con.execute(f"""CREATE TABLE md AS SELECT x.s1, x.rec, x.country, x.nn1, x.nn2, x.name_shared, x.addr_shared, x.rec_addr_missing,
    min(n.df) min_name_df, max(n.df) max_name_df
  FROM (SELECT *, unnest(string_split(nn1, ' ')) tok FROM {m}) x LEFT JOIN ndf n ON n.country = x.country AND n.token = x.tok
  GROUP BY ALL""")
print("missed sample, S1 name tokens' S2/S3 doc-frequency (blocker drops tokens with df > 1500):")
print(con.sql("""SELECT (nn1 = nn2) name_exact, COUNT(*) n,
    ROUND(AVG((min_name_df > 1500)::INT),3) all_name_tokens_dropped, quantile_cont(min_name_df, 0.5) median_rarest_name_token_df
  FROM md GROUP BY 1""").df().to_string(index=False))
print("\n25 missed examples:")
print(con.sql(f"""SELECT src, country, n1, a1, n2, a2 FROM {m} USING SAMPLE 25 ROWS (reservoir, 3)""").df().to_string(index=False))
