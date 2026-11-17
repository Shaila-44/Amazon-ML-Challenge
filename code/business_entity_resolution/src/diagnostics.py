"""Post-hoc error analysis: what to look at if the leaderboard score comes
back lower than local validation suggests.

The single validation F_0.5 number `pipeline.py` prints is a genuine
full-scale estimate, but it's computed on a split carved from *train* data
(US + India only) -- test also has France, which never appears in train. If
the leaderboard score disappoints, the first question is "is this uniform,
or is it France dragging the average down", and that requires a per-country
breakdown this module provides.
"""
from __future__ import annotations

from typing import Dict, Set

import duckdb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .scoring import score_entity


def feature_importance_report(model) -> pd.DataFrame:
    """Which features the classifier actually relies on. A model leaning
    almost entirely on name features despite the address-channel finding
    (see blocking.py docstring) would be a red flag worth investigating."""
    return pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _entity_country_map(con: duckdb.DuckDBPyConnection, s1_parquet: str) -> Dict[str, str]:
    rows = con.execute(f"SELECT entity_id, country FROM read_parquet('{s1_parquet}')").fetchall()
    return dict(rows)


def validation_breakdown_by_country(
    con: duckdb.DuckDBPyConnection,
    s1_parquet: str,
    val_part: pd.DataFrame,
    val_scores: np.ndarray,
    threshold: float,
    val_gt: Dict[str, Set[str]],
    val_s1: Set[str],
) -> pd.DataFrame:
    """Macro F_0.5 (and precision/recall separately) split by country.
    Only covers whatever countries are in the validation split (train-side
    only, so no France here -- see module docstring) but still catches a
    US-vs-India imbalance if one exists.
    """
    country_of = _entity_country_map(con, s1_parquet)

    preds: Dict[str, Set[str]] = {}
    for s1, cand, score in zip(val_part["source1_entity_id"], val_part["candidate_entity_id"], val_scores):
        if score >= threshold:
            preds.setdefault(s1, set()).add(cand)

    rows = []
    for s1 in val_s1:
        country = country_of.get(s1, "UNKNOWN")
        truth = val_gt.get(s1, set())
        predicted = preds.get(s1, set())
        tp = len(predicted & truth)
        precision = tp / len(predicted) if predicted else (1.0 if not truth else 0.0)
        recall = tp / len(truth) if truth else 1.0
        f_score = score_entity(predicted, truth)
        rows.append({"country": country, "precision": precision, "recall": recall, "f_0_5": f_score,
                      "is_singleton": len(truth) == 0})

    df = pd.DataFrame(rows)
    summary = df.groupby("country").agg(
        n_entities=("f_0_5", "size"),
        macro_f_0_5=("f_0_5", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        singleton_rate=("is_singleton", "mean"),
    ).reset_index()
    return summary.sort_values("n_entities", ascending=False)


def sample_error_examples(
    con: duckdb.DuckDBPyConnection,
    s1_parquet: str,
    s2_parquet: str,
    s3_parquet: str,
    val_part: pd.DataFrame,
    val_scores: np.ndarray,
    threshold: float,
    val_gt: Dict[str, Set[str]],
    n_examples: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Readable false-positive / false-negative examples (actual names and
    addresses, not just IDs) -- for eyeballing what kind of noise the model
    still gets wrong."""
    lookup = con.execute(f"""
        SELECT entity_id, business_name, business_address, country FROM read_parquet('{s1_parquet}')
        UNION ALL
        SELECT entity_id, business_name, business_address, country FROM read_parquet('{s2_parquet}')
        UNION ALL
        SELECT entity_id, business_name, business_address, country FROM read_parquet('{s3_parquet}')
    """).df().set_index("entity_id")

    rows = val_part.copy()
    rows["score"] = val_scores
    rows["is_true_match"] = [
        s1 in val_gt and cand in val_gt[s1]
        for s1, cand in zip(rows["source1_entity_id"], rows["candidate_entity_id"])
    ]
    rows["predicted_match"] = rows["score"] >= threshold

    false_positives = rows[(rows["predicted_match"]) & (~rows["is_true_match"])].nlargest(n_examples, "score")
    false_negatives = rows[(~rows["predicted_match"]) & (rows["is_true_match"])].nsmallest(n_examples, "score")

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _, r in df.iterrows():
            s1_info = lookup.loc[r["source1_entity_id"]] if r["source1_entity_id"] in lookup.index else None
            cand_info = lookup.loc[r["candidate_entity_id"]] if r["candidate_entity_id"] in lookup.index else None
            out.append({
                "score": r["score"],
                "s1_name": s1_info["business_name"] if s1_info is not None else "?",
                "s1_addr": s1_info["business_address"] if s1_info is not None else "?",
                "cand_name": cand_info["business_name"] if cand_info is not None else "?",
                "cand_addr": cand_info["business_address"] if cand_info is not None else "?",
                "country": s1_info["country"] if s1_info is not None else "?",
            })
        return pd.DataFrame(out)

    return {"false_positives": _enrich(false_positives), "false_negatives": _enrich(false_negatives)}


def precision_recall_curve(
    val_part: pd.DataFrame,
    val_scores: np.ndarray,
    val_gt: Dict[str, Set[str]],
    val_s1: Set[str],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full precision/recall/F_0.5 sweep (not just the single chosen
    threshold) -- shows the tradeoff so a different threshold can be picked
    later without retraining, if e.g. the leaderboard rewards a different
    precision/recall balance than expected."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1, cand, score in zip(val_part["source1_entity_id"], val_part["candidate_entity_id"], val_scores):
            if score >= t:
                preds.setdefault(s1, set()).add(cand)
        scores = [score_entity(preds.get(s1, set()), val_gt.get(s1, set())) for s1 in val_s1]
        n_matched = sum(1 for s1 in val_s1 if preds.get(s1))
        rows.append({"threshold": t, "macro_f_0_5": np.mean(scores), "pct_entities_matched": n_matched / len(val_s1)})
    return pd.DataFrame(rows)


def print_full_report(
    con: duckdb.DuckDBPyConnection,
    s1_parquet: str,
    s2_parquet: str,
    s3_parquet: str,
    model,
    val_part: pd.DataFrame,
    val_scores: np.ndarray,
    threshold: float,
    val_gt: Dict[str, Set[str]],
    val_s1: Set[str],
) -> None:
    """Everything at once, for a quick post-run read."""
    print("\n=== FEATURE IMPORTANCES ===")
    print(feature_importance_report(model).to_string(index=False))

    print("\n=== VALIDATION BREAKDOWN BY COUNTRY ===")
    print(validation_breakdown_by_country(con, s1_parquet, val_part, val_scores, threshold, val_gt, val_s1).to_string(index=False))

    print("\n=== PRECISION/RECALL CURVE ===")
    print(precision_recall_curve(val_part, val_scores, val_gt, val_s1).to_string(index=False))

    print("\n=== SAMPLE FALSE POSITIVES (model said match, ground truth says no) ===")
    examples = sample_error_examples(con, s1_parquet, s2_parquet, s3_parquet, val_part, val_scores, threshold, val_gt)
    print(examples["false_positives"].to_string(index=False))

    print("\n=== SAMPLE FALSE NEGATIVES (ground truth says match, model missed it) ===")
    print(examples["false_negatives"].to_string(index=False))
