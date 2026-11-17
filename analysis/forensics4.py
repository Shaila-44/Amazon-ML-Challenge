import duckdb, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
con = duckdb.connect(); con.execute("SET memory_limit='5GB'")
D = "/Users/pravin/Desktop/Amazon"
ents = ["S1-965667", "S1-55344266"]
g = con.sql(f"""SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS id
  FROM read_csv('{D}/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='"', all_varchar=true)
  WHERE source1_entity_id IN {tuple(ents)}""").df()
for e in ents:
    print(f"\n######## {e}")
    print(con.sql(f"SELECT entity_id, business_name, business_address, name_tokens, addr_tokens FROM read_parquet('{D}/dataset/processed/train_source1.parquet') WHERE entity_id='{e}'").df().to_string(index=False))
    ids = tuple(g[g.source1_entity_id == e].id)
    for s in ("train_source2", "train_source3"):
        print(con.sql(f"SELECT entity_id, business_name, business_address, name_tokens, addr_tokens FROM read_parquet('{D}/dataset/processed/{s}.parquet') WHERE entity_id IN {ids}").df().to_string(index=False))
print("\n=== avg token counts by source (train) ===")
for s in ("train_source1","train_source2","train_source3","test_source2","test_source3"):
    print(s, con.sql(f"SELECT country, ROUND(AVG(len(name_tokens)),2) name_tok, ROUND(AVG(len(addr_tokens)),2) addr_tok, ROUND(AVG((business_address IS NULL OR business_address='')::INT),3) addr_missing FROM read_parquet('{D}/dataset/processed/{s}.parquet') GROUP BY 1 ORDER BY 1").fetchall())
