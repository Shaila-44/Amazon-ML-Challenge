"""Stream each raw source TSV once, normalize name/address, and write compact
Parquet with token lists. Decouples the (Python, per-row) normalization cost
from the (DuckDB, set-based) blocking joins that consume it.

Streams in chunks so peak memory stays bounded regardless of file size (train
source2/source3 are 5M+ rows each) -- safe on an 8GB machine.
"""
from __future__ import annotations

import argparse
import csv
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import normalize_address, normalize_name

CHUNK_SIZE = 200_000


def _rows(path: str):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            if not row:
                continue
            yield row, idx


def preprocess_source(in_path: str, out_path: str) -> int:
    """Normalize one source*.tsv into Parquet with columns:
    entity_id, business_name, business_address, country, name_norm, addr_norm,
    name_tokens (list<str>), addr_tokens (list<str>).
    Returns the row count written.
    """
    writer = None
    buf = {"entity_id": [], "business_name": [], "business_address": [], "country": [],
           "name_norm": [], "addr_norm": [], "name_tokens": [], "addr_tokens": []}
    n_written = 0

    def flush():
        nonlocal writer, n_written
        if not buf["entity_id"]:
            return
        table = pa.table({
            "entity_id": pa.array(buf["entity_id"], type=pa.string()),
            "business_name": pa.array(buf["business_name"], type=pa.string()),
            "business_address": pa.array(buf["business_address"], type=pa.string()),
            "country": pa.array(buf["country"], type=pa.string()),
            "name_norm": pa.array(buf["name_norm"], type=pa.string()),
            "addr_norm": pa.array(buf["addr_norm"], type=pa.string()),
            "name_tokens": pa.array(buf["name_tokens"], type=pa.list_(pa.string())),
            "addr_tokens": pa.array(buf["addr_tokens"], type=pa.list_(pa.string())),
        })
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        n_written += len(buf["entity_id"])
        for v in buf.values():
            v.clear()

    for row, idx in _rows(in_path):
        entity_id = row[idx["entity_id"]]
        name = row[idx["business_name"]]
        addr = row[idx["business_address"]]
        country = row[idx["country"]]
        name_norm = normalize_name(name)
        addr_norm = normalize_address(addr)
        buf["entity_id"].append(entity_id)
        buf["business_name"].append(name)
        buf["business_address"].append(addr)
        buf["country"].append(country)
        buf["name_norm"].append(name_norm)
        buf["addr_norm"].append(addr_norm)
        buf["name_tokens"].append(name_norm.split() if name_norm else [])
        buf["addr_tokens"].append(addr_norm.split() if addr_norm else [])
        if len(buf["entity_id"]) >= CHUNK_SIZE:
            flush()
            print(f"  {out_path}: {n_written:,} rows...", file=sys.stderr)

    flush()
    if writer is not None:
        writer.close()
    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()
    n = preprocess_source(args.in_path, args.out_path)
    print(f"wrote {n:,} rows to {args.out_path}")


if __name__ == "__main__":
    main()
