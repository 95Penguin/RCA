"""Continuous AP, validation-only thresholds and common service ranking."""
from __future__ import annotations

import numpy as np


def binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "threshold": float(threshold), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn}


def validation_threshold(labels, scores):
    """Highest threshold among max-F1 choices, using score ties as groups."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError("Threshold fitting needs finite, nonempty aligned validation scores")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[ends]
    denom = ends + 1 + y.sum()
    f1 = np.divide(2*tp, denom, out=np.zeros(len(ends), dtype=float), where=denom > 0)
    if not y.sum():
        return float(np.nextafter(s[0], np.inf))
    return float(s[ends[int(np.argmax(f1))]])


def summarize_predictions(data, ends, risk, roots, threshold, probability):
    from training.metrics import average_precision, rca_metrics
    d = data.dataset
    labels = np.asarray([d.failure_targets[i] for i in ends])
    targets = d.node_targets[ends].copy()
    for row, end in enumerate(ends):
        if not labels[row] or not d._root_label_valid.get(d.timestamps[end], bool(targets[row].any())):
            targets[row] = 0
    if not np.isfinite(risk).all() or not np.isfinite(roots).all():
        raise ValueError("Nonfinite prediction; do not silently drop failed windows")
    event_groups = {}
    for row, end in enumerate(ends):
        if targets[row].any() and d.event_ids[end]:
            event_groups.setdefault((d.case_ids[end], d.event_ids[end]), []).append(row)
    event_scores, event_targets = [], []
    for rows in event_groups.values():
        if not np.all(targets[rows] == targets[rows[0]]):
            raise ValueError("Inconsistent root labels within event")
        event_scores.append(roots[rows].mean(axis=0))
        event_targets.append(targets[rows[0]])
    n = len(d.services)
    return {
        "samples": len(ends), "positive_samples": int(labels.sum()),
        "average_precision": average_precision(labels, risk),
        "at_validation_threshold": binary_metrics(labels, risk, threshold),
        "at_probability_0_5": binary_metrics(labels, risk, 0.5) if probability else None,
        "root_cause_window": rca_metrics(targets, roots),
        "root_cause_retrospective_event": rca_metrics(
            np.asarray(event_targets).reshape(-1, n), np.asarray(event_scores).reshape(-1, n)),
        "retrospective_event_aggregation": "mean raw service scores over attributed event windows; not first-warning RCA",
        "root_all_tied_windows": int((roots.max(axis=1) == roots.min(axis=1)).sum()),
        "tie_break": "fixed panel service order",
    }
