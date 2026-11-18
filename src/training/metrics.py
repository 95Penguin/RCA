"""Classification and service ranking metrics; ranks start at one."""

from __future__ import annotations

import numpy as np


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recalled = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recalled))


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    predictions = probabilities >= 0.5
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "accuracy": float((predictions == labels).mean()),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": f1,
        "average_precision": average_precision(labels, probabilities),
    }


def multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    """Report accuracy and macro-F1 without external metric dependencies."""
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise ValueError("multiclass probabilities must be [samples, classes]")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("class names do not match multiclass probabilities")
    predictions = probabilities.argmax(axis=1)
    per_class: dict[str, dict[str, float | int]] = {}
    supported_f1_values: list[float] = []
    for index, name in enumerate(class_names):
        true = labels == index
        predicted = predictions == index
        tp = int(np.logical_and(true, predicted).sum())
        fp = int(np.logical_and(~true, predicted).sum())
        fn = int(np.logical_and(true, ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if true.any():
            supported_f1_values.append(f1)
        per_class[name] = {
            "support": int(true.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "samples": int(labels.size),
        "accuracy": float((predictions == labels).mean()) if labels.size else None,
        "macro_f1": (
            float(np.mean(supported_f1_values)) if supported_f1_values else None
        ),
        "per_class": per_class,
    }


def rca_metrics(node_targets: np.ndarray, node_scores: np.ndarray) -> dict[str, float | int | None]:
    positive_rows = np.flatnonzero(node_targets.sum(axis=1) > 0)
    if not len(positive_rows):
        return {
            "positive_windows": 0,
            "hit_at_1": None,
            "hit_at_2": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "mfr": None,
            "mar": None,
            "mrr": None,
        }
    first_ranks: list[float] = []
    average_ranks: list[float] = []
    hits_2: list[float] = []
    hits_1: list[float] = []
    hits_3: list[float] = []
    hits_5: list[float] = []
    reciprocal_ranks: list[float] = []
    for row in positive_rows:
        order = np.argsort(-node_scores[row], kind="stable")
        truth = node_targets[row].astype(bool)
        ranks = np.flatnonzero(truth[order])
        first_rank = int(ranks[0])
        first_ranks.append(float(first_rank + 1))
        average_ranks.append(float(np.mean(ranks + 1)))
        hits_2.append(float(first_rank < 2))
        hits_1.append(float(first_rank < 1))
        hits_3.append(float(first_rank < 3))
        hits_5.append(float(first_rank < 5))
        reciprocal_ranks.append(1.0 / (first_rank + 1))
    return {
        "positive_windows": int(len(positive_rows)),
        "hit_at_1": float(np.mean(hits_1)),
        "hit_at_2": float(np.mean(hits_2)),
        "mfr": float(np.mean(first_ranks)),
        "mar": float(np.mean(average_ranks)),
        "hit_at_3": float(np.mean(hits_3)),
        "hit_at_5": float(np.mean(hits_5)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }
