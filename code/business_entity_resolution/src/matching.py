"""Pair-scoring model: given candidate pairs + features, decide which are true matches.

Constraint (problem_statement.md): final model must be MIT/Apache-2.0 licensed
and <= 8B parameters. LightGBM (MIT license) trained from scratch on our own
features carries no license/parameter question at all -- it's not a
redistributed pretrained model. See amazon-ml-challenge-2026 project notes for
the full reasoning on why this satisfies the constraint (and the open
question worth confirming with organizers via the Google Form).

F_0.5 evaluation is precision-heavy (beta=0.5), so `tune_threshold` sweeps
candidate thresholds against the real macro F_0.5 (src/scoring.py), not a
generic metric like accuracy or plain F1.
"""
from __future__ import annotations

from typing import Dict, Set

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .scoring import macro_f_0_5


def label_pairs(candidate_pairs: pd.DataFrame, ground_truth_pairs: Set[tuple]) -> pd.Series:
    """candidate_pairs needs columns source1_entity_id, candidate_entity_id.
    ground_truth_pairs is a set of (source1_entity_id, matched_entity_id)
    tuples. Returns a 0/1 label Series aligned to candidate_pairs' index.
    """
    keys = list(zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"]))
    return pd.Series([1 if k in ground_truth_pairs else 0 for k in keys], index=candidate_pairs.index)


def train_matcher(features_df: pd.DataFrame, labels: pd.Series, **lgb_params) -> lgb.LGBMClassifier:
    """Train a LightGBM binary classifier on candidate-pair features.

    Candidate pairs are heavily imbalanced (most candidates are non-matches),
    so `is_unbalance=True` by default -- override via lgb_params if a
    different imbalance-handling strategy is preferred.
    """
    params = dict(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=42,
    )
    params.update(lgb_params)
    model = lgb.LGBMClassifier(**params)
    model.fit(features_df[FEATURE_COLUMNS], labels)
    return model


def score_pairs(model: lgb.LGBMClassifier, features_df: pd.DataFrame) -> np.ndarray:
    """Return match probability per candidate pair."""
    return model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1]


def _predictions_from_scores(
    source1_ids: pd.Series, candidate_ids: pd.Series, scores: np.ndarray, threshold: float
) -> Dict[str, Set[str]]:
    preds: Dict[str, Set[str]] = {}
    for s1, cand, score in zip(source1_ids, candidate_ids, scores):
        if score >= threshold:
            preds.setdefault(s1, set()).add(cand)
    return preds


def tune_threshold(
    source1_ids: pd.Series,
    candidate_ids: pd.Series,
    scores: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_source1_ids: Set[str],
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep thresholds, score each with the real macro F_0.5 (including
    singletons -- entities with no candidates above threshold predict empty,
    which `ground_truth` must account for via `all_source1_ids`).

    Returns (best_threshold, best_f_0_5).

    Default grid: the original [0.1, 0.95] coarse sweep was demonstrably
    truncating the search -- F_0.5 was still climbing at the 0.95 boundary
    (0.9740 there vs a true interior optimum of 0.9768 at 0.985, confirmed
    by the curve then collapsing sharply toward 1.0 -- 0.9434 at 0.999,
    0.6613 at 0.9999). This grid covers the full range with finer
    resolution where the curve is actually sensitive (F_0.5 is
    precision-heavy, so the optimum tends to sit high and the collapse
    near 1.0 can be sharp).
    """
    if thresholds is None:
        thresholds = np.concatenate([
            np.linspace(0.05, 0.90, 18),
            np.linspace(0.90, 0.995, 20),
            np.array([0.997, 0.999, 0.9999]),
        ])

    best_threshold, best_score = 0.5, -1.0
    for t in thresholds:
        preds = _predictions_from_scores(source1_ids, candidate_ids, scores, t)
        full_preds = {s1: preds.get(s1, set()) for s1 in all_source1_ids}
        score = macro_f_0_5(full_preds, ground_truth)
        if score > best_score:
            best_score, best_threshold = score, t
    return best_threshold, best_score


def per_entity_f05(pred_pairs: pd.DataFrame, n_truth: pd.Series) -> pd.Series:
    """Per-entity F_0.5 against the FULL ground truth.

    pred_pairs: predicted pairs with columns source1_entity_id, label (bool).
    n_truth: true-match count per entity, indexed by source1_entity_id, for
    EVERY evaluated entity -- including true matches that blocking never
    surfaced (they count as misses) and entities with no candidates at all.
    Validating only against positives present in the candidate set hides
    blocking recall entirely.
    """
    g = pred_pairs.groupby("source1_entity_id")["label"].agg(["size", "sum"])
    n_pred = g["size"].reindex(n_truth.index, fill_value=0).to_numpy()
    tp = g["sum"].reindex(n_truth.index, fill_value=0).to_numpy()
    nt = n_truth.to_numpy()
    p = np.divide(tp, n_pred, out=np.zeros(len(nt)), where=n_pred > 0)
    r = np.divide(tp, nt, out=np.zeros(len(nt)), where=nt > 0)
    denom = 0.25 * p + r
    f = np.divide(1.25 * p * r, denom, out=np.zeros(len(nt)), where=denom > 0)
    f = np.where(nt == 0, (n_pred == 0).astype(float), f)
    return pd.Series(f, index=n_truth.index)


def tune_threshold_full(
    pairs: pd.DataFrame, n_truth: pd.Series, thresholds: np.ndarray | None = None
) -> tuple[float, float, pd.DataFrame]:
    """Sweep thresholds over ALL candidates of held-out entities (no negative
    subsampling -- that shifts the class prior and picks a threshold far too
    permissive for the real candidate distribution).

    pairs: columns source1_entity_id, score, label (+ candidate_entity_id for
    the duplicate-claim and S2/S3 columns). Returns
    (best_threshold, best_f05, sweep_table)."""
    if thresholds is None:
        thresholds = np.round(np.concatenate([
            np.arange(0.05, 0.50, 0.05), np.arange(0.50, 0.99, 0.01), [0.99, 0.992, 0.995, 0.997, 0.999],
        ]), 4)
    has_cand = "candidate_entity_id" in pairs.columns
    rows = []
    for t in thresholds:
        sel = pairs[pairs["score"] >= t]
        f = per_entity_f05(sel, n_truth)
        per_ent = sel.groupby("source1_entity_id").size()
        row = dict(threshold=t, f05=f.mean(), pairs=len(sel),
                   precision=sel["label"].mean() if len(sel) else np.nan,
                   recall=sel["label"].sum() / max(n_truth.sum(), 1),
                   mean_pred_per_entity_if_any=per_ent.mean() if len(per_ent) else 0.0,
                   frac_entities_empty=1.0 - len(per_ent) / len(n_truth))
        if has_cand:
            claims = sel["candidate_entity_id"].map(sel["candidate_entity_id"].value_counts())
            row.update(dup_claim_pct=float((claims > 1).mean()) if len(sel) else 0.0,
                       n_pred_S2=int(sel["candidate_entity_id"].str.startswith("S2").sum()),
                       n_pred_S3=int(sel["candidate_entity_id"].str.startswith("S3").sum()))
        rows.append(row)
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["f05"].idxmax()]
    return float(best["threshold"]), float(best["f05"]), sweep


def select_matches(
    source1_ids: pd.Series, candidate_ids: pd.Series, scores: np.ndarray, threshold: float
) -> Dict[str, Set[str]]:
    """Apply the tuned threshold to get final matches per source1_entity_id."""
    return _predictions_from_scores(source1_ids, candidate_ids, scores, threshold)
