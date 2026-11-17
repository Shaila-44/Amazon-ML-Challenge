"""Candidate generation / blocking stage, implemented as DuckDB SQL over the
preprocessed Parquet files from `preprocess.py`.

Strategy (empirically grounded, see amazon-ml-challenge-2026 project notes /
Documentation_template.md Section 3):
  - Token blocking on BOTH normalized name tokens and normalized address
    tokens, not name alone. Real-data check on transliterated India names
    (Devanagari/Tamil/Gujarati) showed name-token overlap fails for those
    (average fuzzy ratio ~48, barely above the ~38 noise floor) while address
    token_set_ratio for the SAME true-matched pairs averages ~96 -- address
    stays in Latin script/numerals even when the name doesn't.
  - Country is used as a hard join filter: ground-truth analysis on the real
    train data found 0 cross-country true matches, so this costs no recall
    and prunes the join dramatically -- and (below) doubles as the
    partitioning axis that keeps each individual query's working set small
    enough to fit in memory.
  - Overly common tokens (generic words like "private", "road", or a
    frequent city name) are dropped before the join -- they create huge,
    uninformative blocks. Threshold is document-frequency based, not a fixed
    stopword list, so it adapts per token_type/country rather than assuming
    English-only vocabulary.
  - Each S1 entity is restricted to its N rarest tokens that actually occur
    in S2/S3 before the join -- the main fix that keeps join cost bounded at
    full scale.
  - Candidates are ranked per S1 entity by IDF-weighted overlap on matching
    fields and capped at `top_k`. Ranking by raw shared-token count lost
    ~25 points of recall at full scale: median uncapped list ~1,100
    candidates, and missed true matches were all tied at the cutoff with
    one-token junk (ties then broke alphabetically, starving Source 3).

Processing is partitioned by country (see `generate_candidates`): a single
combined query across all countries repeatedly hit DuckDB's memory limit at
just ~1/6 of full scale (350k S1 entities) even after the token-count caps
below, because DuckDB's working set for the join+rank still scales with the
*combined* corpus. Running one smaller query per country and unioning the
(disk-backed) results keeps peak memory bounded by the largest single
country's data instead of the sum of all countries.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import duckdb


class _DiskPeak:
    """Samples used disk space in the background (DuckDB spills to a temp
    directory on the same filesystem) so blocking can report its peak."""

    def __init__(self, path: str = ".", interval: float = 1.0):
        self._path, self._interval = path, interval
        self._peak = shutil.disk_usage(path).used
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._peak = max(self._peak, shutil.disk_usage(self._path).used)

    def reset(self) -> None:
        self._peak = shutil.disk_usage(self._path).used

    def peak_gb(self) -> float:
        return round(max(self._peak, shutil.disk_usage(self._path).used) / 1024**3, 2)

    def stop(self) -> None:
        self._stop.set()


def _generate_for_country(
    con: duckdb.DuckDBPyConnection,
    s1_parquet: str,
    s2_parquet: str,
    s3_parquet: str,
    country: str,
    top_k: int,
    min_token_len: int,
    max_doc_freq_frac: float,
    max_doc_freq_abs: int,
    max_tokens_per_entity: int,
    out_prefix: str,
    n_buckets: int,
    disk: "_DiskPeak",
    stats: list | None,
) -> list[str]:
    """Candidates for one country, written as `n_buckets` Parquet parts.

    Filtered to the country from the start, and the S1 side is further split
    into hash buckets for the join + rank. Ranking is per S1 entity and all
    document frequencies are computed on the full country data, so bucketing
    does not change the output -- it only bounds the working set: the
    unbucketed join (~3B pre-cap rows at full train scale, carrying two IDF
    sums) exhausted 40GB of DuckDB spill space.
    """
    esc_country = country.replace("'", "''")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1_tokens_raw AS
        SELECT entity_id, unnest(list_distinct(name_tokens)) AS token, 'name' AS token_type
        FROM read_parquet('{s1_parquet}') WHERE country = '{esc_country}' AND len(name_tokens) > 0
        UNION ALL
        SELECT entity_id, unnest(list_distinct(addr_tokens)) AS token, 'addr' AS token_type
        FROM read_parquet('{s1_parquet}') WHERE country = '{esc_country}' AND len(addr_tokens) > 0
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s23_tokens_raw AS
        SELECT entity_id, token, token_type FROM (
            SELECT entity_id, unnest(list_distinct(name_tokens)) AS token, 'name' AS token_type
            FROM read_parquet('{s2_parquet}') WHERE country = '{esc_country}' AND len(name_tokens) > 0
            UNION ALL
            SELECT entity_id, unnest(list_distinct(addr_tokens)) AS token, 'addr' AS token_type
            FROM read_parquet('{s2_parquet}') WHERE country = '{esc_country}' AND len(addr_tokens) > 0
            UNION ALL
            SELECT entity_id, unnest(list_distinct(name_tokens)) AS token, 'name' AS token_type
            FROM read_parquet('{s3_parquet}') WHERE country = '{esc_country}' AND len(name_tokens) > 0
            UNION ALL
            SELECT entity_id, unnest(list_distinct(addr_tokens)) AS token, 'addr' AS token_type
            FROM read_parquet('{s3_parquet}') WHERE country = '{esc_country}' AND len(addr_tokens) > 0
        )
    """)

    con.execute("""
        CREATE OR REPLACE TEMP VIEW s23_group_totals AS
        SELECT token_type, COUNT(DISTINCT entity_id) AS total_in_group
        FROM s23_tokens_raw GROUP BY token_type
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s23_doc_freq AS
        SELECT d.token_type, d.token, d.df, t.total_in_group
        FROM (
            SELECT token_type, token, COUNT(DISTINCT entity_id) AS df
            FROM s23_tokens_raw WHERE len(token) >= {min_token_len}
            GROUP BY token_type, token
        ) d
        JOIN s23_group_totals t USING (token_type)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s23_keep AS
        SELECT token_type, token, df, ln(total_in_group * 1.0 / df) AS idf
        FROM s23_doc_freq
        WHERE df <= total_in_group * {max_doc_freq_frac} AND df <= {max_doc_freq_abs}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW s23_tokens AS
        SELECT s.entity_id, s.token, s.token_type, k.idf
        FROM s23_tokens_raw s JOIN s23_keep k USING (token_type, token)
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1_doc_freq AS
        SELECT token_type, token, COUNT(DISTINCT entity_id) AS df
        FROM s1_tokens_raw WHERE len(token) >= {min_token_len}
        GROUP BY token_type, token
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1_tokens_capped AS
        SELECT s.entity_id, s.token, s.token_type
        FROM s1_tokens_raw s
        JOIN (SELECT token_type, token FROM s1_doc_freq WHERE df <= {max_doc_freq_abs}) f
          USING (token_type, token)
        WHERE len(s.token) >= {min_token_len}
    """)
    # Only tokens that can actually match something in S2/S3 compete for the
    # per-entity slots; ranking absent tokens as "rarest" wasted slots.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1_tokens AS
        SELECT entity_id, token, token_type FROM (
            SELECT t.entity_id, t.token, t.token_type,
                   ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY k.df ASC, t.token, t.token_type) AS rarity_rank
            FROM s1_tokens_capped t JOIN s23_keep k USING (token_type, token)
        ) WHERE rarity_rank <= {max_tokens_per_entity}
    """)

    t0 = time.time()
    disk.reset()
    con.execute("CREATE OR REPLACE TEMP TABLE s23_tokens_m AS SELECT * FROM s23_tokens")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s1_tokens_m AS
                    SELECT entity_id, token, token_type, hash(entity_id) % {n_buckets} AS bucket FROM s1_tokens""")
    if stats is not None:
        stats.append(dict(country=country, step="token_tables",
                          s23_token_rows=con.execute("SELECT COUNT(*) FROM s23_tokens_m").fetchone()[0],
                          s1_token_rows=con.execute("SELECT COUNT(*) FROM s1_tokens_m").fetchone()[0],
                          seconds=round(time.time() - t0, 1), peak_disk_used_gb=disk.peak_gb()))

    # Rank by IDF-weighted overlap on matching fields (name<->name,
    # addr<->addr). A raw shared-token count ties true matches with hundreds
    # of one-token junk candidates at the top-k cutoff.
    # The IDF values are summed in sorted order: a plain parallel SUM adds
    # them in varying order, so candidates with identical token sets got sums
    # differing in the last bit and float noise, not the hash tie-break,
    # decided their order (run-to-run candidate differences at the cutoff).
    paths = []
    for b in range(n_buckets):
        path = f"{out_prefix}_{b:02d}.parquet"
        t0 = time.time()
        disk.reset()
        con.sql(f"""
            WITH joined AS (
                SELECT s1.entity_id AS source1_entity_id, s23.entity_id AS candidate_entity_id,
                       COUNT(*) AS shared_tokens,
                       list_sum(list_sort(list(s23.idf))) AS idf_all,
                       list_sum(list_sort(list(CASE WHEN s1.token_type = s23.token_type THEN s23.idf ELSE 0 END))) AS idf_same
                FROM s1_tokens_m s1 JOIN s23_tokens_m s23 ON s1.token = s23.token
                WHERE s1.bucket = {b}
                GROUP BY s1.entity_id, s23.entity_id
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY idf_same DESC, idf_all DESC, hash(candidate_entity_id)
                ) AS rn
                FROM joined
            )
            SELECT source1_entity_id, candidate_entity_id, shared_tokens, rn AS candidate_rank
            FROM ranked WHERE rn <= {top_k}
        """).write_parquet(path)
        paths.append(path)
        if stats is not None:
            s1_n, rows = con.execute(f"""SELECT COUNT(DISTINCT source1_entity_id), COUNT(*)
                                         FROM read_parquet('{path}')""").fetchone()
            stats.append(dict(country=country, step="bucket", bucket=b,
                              s1_entities_in_bucket=con.execute(
                                  f"SELECT COUNT(DISTINCT entity_id) FROM s1_tokens_m WHERE bucket = {b}").fetchone()[0],
                              s1_entities_with_candidates=s1_n, candidate_rows=rows,
                              seconds=round(time.time() - t0, 1), peak_disk_used_gb=disk.peak_gb()))
    con.execute("DROP TABLE s23_tokens_m")
    con.execute("DROP TABLE s1_tokens_m")
    return paths


def generate_candidates(
    con: duckdb.DuckDBPyConnection,
    s1_parquet: str,
    s2_parquet: str,
    s3_parquet: str,
    top_k: int = 50,
    min_token_len: int = 2,
    max_doc_freq_frac: float = 0.02,
    max_doc_freq_abs: int = 1500,
    max_tokens_per_entity: int = 8,
    n_buckets: int = 8,
    stats: list | None = None,
) -> "duckdb.DuckDBPyRelation":
    """Return a DuckDB relation with columns
    (source1_entity_id, candidate_entity_id, shared_tokens, candidate_rank) --
    the ranked, top_k-capped candidate set, unioned across every country
    present in `s1_parquet` (processed one country at a time -- see module
    docstring). candidate_rank (1 = best) lets a smaller K be cut later
    without re-blocking.
    """
    countries = [r[0] for r in con.execute(
        f"SELECT DISTINCT country FROM read_parquet('{s1_parquet}') WHERE country IS NOT NULL"
    ).fetchall()]

    con.execute("SET preserve_insertion_order = false")
    tmpdir = tempfile.mkdtemp(prefix="blocking_by_country_")
    part_paths = []
    disk = _DiskPeak()
    try:
        for i, country in enumerate(countries):
            part_paths += _generate_for_country(
                con, s1_parquet, s2_parquet, s3_parquet, country,
                top_k, min_token_len, max_doc_freq_frac, max_doc_freq_abs, max_tokens_per_entity,
                os.path.join(tmpdir, f"country{i}"), n_buckets, disk, stats,
            )
    finally:
        disk.stop()

    paths_sql = ", ".join(f"'{p}'" for p in part_paths)
    return con.sql(f"SELECT * FROM read_parquet([{paths_sql}])")
