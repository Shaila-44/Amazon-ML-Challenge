"""Local F_0.5 scoring, matching the official formula in problem_statement.md,
for evaluating a held-out validation split before submitting.

F_0.5 = (1.25 * P * R) / (0.25 * P + R), macro-averaged per Source 1 entity.
Singletons (no true matches) score 1.0 for an empty prediction, 0.0 for any
false positive.
"""
from __future__ import annotations

from typing import Dict, Set


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def score_entity(predicted: Set[str], truth: Set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_beta(precision, recall)


def macro_f_0_5(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """predictions/ground_truth: source1_entity_id -> set of matched entity_ids.
    Every source1_entity_id in ground_truth must have an entry in predictions
    (missing entities should be treated as empty predictions by the caller).
    """
    scores = [score_entity(predictions.get(s1_id, set()), truth) for s1_id, truth in ground_truth.items()]
    return sum(scores) / len(scores) if scores else 0.0
