"""Pairwise similarity features for (source1_entity, candidate) pairs.

Computed in chunks so peak memory stays bounded at full scale (blocking can
produce tens of millions of candidate pairs). rapidfuzz features are the
primary signal (fast, C-speed per pair); TF-IDF char-n-gram cosine on the
name field is added per the problem statement's own "Tips for Success"
(explicitly recommends Jaccard/Levenshtein/TF-IDF cosine).

Country is deliberately NOT a feature: blocking already hard-filters to
same-country candidates (0 cross-country true matches in the real ground
truth), so it would be a constant, zero-signal column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_SIZE = 500_000

FEATURE_COLUMNS = [
    "shared_tokens",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_tfidf_cosine",
    "addr_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_jaccard",
    "name_len_diff",
]


def fit_tfidf(names: list[str]) -> TfidfVectorizer:
    """Fit a character-n-gram TF-IDF vectorizer on normalized business names.
    Char n-grams (not word n-grams) so it still captures partial signal on
    romanized/transliterated names with non-standard tokenization."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=200_000)
    vec.fit([n for n in names if n])
    return vec


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tfidf_cosine_batch(vec: TfidfVectorizer, left: list[str], right: list[str]) -> np.ndarray:
    """Row-wise cosine similarity between paired (left[i], right[i]) strings."""
    L = vec.transform([s or "" for s in left])
    R = vec.transform([s or "" for s in right])
    # row-wise dot product of two L2-normalized (TfidfVectorizer default) sparse matrices
    dot = L.multiply(R).sum(axis=1)
    return np.asarray(dot).ravel()


def compute_features_for_chunk(chunk: pd.DataFrame, vec: TfidfVectorizer) -> pd.DataFrame:
    """chunk must have columns: source1_entity_id, candidate_entity_id,
    shared_tokens, name_norm_1, name_norm_2, addr_norm_1, addr_norm_2,
    name_tokens_1, name_tokens_2, addr_tokens_1, addr_tokens_2."""
    n1, n2 = chunk["name_norm_1"].fillna(""), chunk["name_norm_2"].fillna("")
    a1, a2 = chunk["addr_norm_1"].fillna(""), chunk["addr_norm_2"].fillna("")

    out = pd.DataFrame({
        "source1_entity_id": chunk["source1_entity_id"],
        "candidate_entity_id": chunk["candidate_entity_id"],
        "shared_tokens": chunk["shared_tokens"],
    })
    out["name_ratio"] = [fuzz.ratio(x, y) for x, y in zip(n1, n2)]
    out["name_token_sort_ratio"] = [fuzz.token_sort_ratio(x, y) for x, y in zip(n1, n2)]
    out["name_token_set_ratio"] = [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]
    out["name_partial_ratio"] = [fuzz.partial_ratio(x, y) for x, y in zip(n1, n2)]
    out["name_jaccard"] = [
        _jaccard(set(t1), set(t2)) for t1, t2 in zip(chunk["name_tokens_1"], chunk["name_tokens_2"])
    ]
    out["name_tfidf_cosine"] = _tfidf_cosine_batch(vec, list(n1), list(n2))
    out["addr_ratio"] = [fuzz.ratio(x, y) for x, y in zip(a1, a2)]
    out["addr_token_set_ratio"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]
    out["addr_partial_ratio"] = [fuzz.partial_ratio(x, y) for x, y in zip(a1, a2)]
    out["addr_jaccard"] = [
        _jaccard(set(t1), set(t2)) for t1, t2 in zip(chunk["addr_tokens_1"], chunk["addr_tokens_2"])
    ]
    out["name_len_diff"] = (n1.str.len() - n2.str.len()).abs()
    return out


def compute_features(
    con,
    candidates_relation,
    s1_parquet: str,
    s2_parquet: str,
    s3_parquet: str,
    out_path: str,
    vec: TfidfVectorizer,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Join candidate pairs to normalized name/address fields via DuckDB,
    compute features in chunks, and write to `out_path` (Parquet).
    Returns the number of rows written.
    """
    con.register("candidates", candidates_relation)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1_lookup AS
        SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens
        FROM read_parquet('{s1_parquet}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s23_lookup AS
        SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens
        FROM read_parquet('{s2_parquet}')
        UNION ALL
        SELECT entity_id, name_norm, addr_norm, name_tokens, addr_tokens
        FROM read_parquet('{s3_parquet}')
    """)
    joined = con.sql("""
        SELECT c.source1_entity_id, c.candidate_entity_id, c.shared_tokens,
               s1.name_norm AS name_norm_1, s23.name_norm AS name_norm_2,
               s1.addr_norm AS addr_norm_1, s23.addr_norm AS addr_norm_2,
               s1.name_tokens AS name_tokens_1, s23.name_tokens AS name_tokens_2,
               s1.addr_tokens AS addr_tokens_1, s23.addr_tokens AS addr_tokens_2
        FROM candidates c
        JOIN s1_lookup s1 ON s1.entity_id = c.source1_entity_id
        JOIN s23_lookup s23 ON s23.entity_id = c.candidate_entity_id
    """)

    writer = None
    n_written = 0
    reader = joined.fetch_arrow_reader(batch_size=chunk_size)
    for batch in reader:
        chunk = batch.to_pandas()
        if chunk.empty:
            continue
        feats = compute_features_for_chunk(chunk, vec)
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        n_written += len(feats)

    if writer is not None:
        writer.close()
    return n_written
