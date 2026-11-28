"""Aggregate already-normalized operation responsibility without labels."""
from collections import defaultdict
import math

MODES = ("hard", "soft", "uniform", "request-weighted")


def aggregate_operations(scores, counts, ratios=None, *, mode="uniform", temperature=1.):
    """Return service scores and operation weights; absent services count zero.

    Counts are scored requests after the experiment's per-operation sampling cap.
    Uniform averages operations, not per-service covered-only observations.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown aggregation: {mode}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}, {}
    if any(counts.get(key, 0) <= 0 for key in scores):
        raise ValueError("Each operation needs a positive sampled request count")
    ratios = ratios or {}
    if mode == "hard":
        selected = max(scores, key=lambda key: (ratios.get(key, 0.), counts[key]))
        weights = {key: float(key == selected) for key in scores}
    elif mode == "soft":
        peak = max(ratios.get(key, 0.) for key in scores)
        values = {key: math.exp((ratios.get(key, 0.) - peak) / temperature) for key in scores}
        denominator = sum(values.values())
        weights = {key: value / denominator for key, value in values.items()}
    elif mode == "request-weighted":
        denominator = sum(counts[key] for key in scores)
        weights = {key: counts[key] / denominator for key in scores}
    else:
        weights = {key: 1. / len(scores) for key in scores}
    merged = defaultdict(float)
    for key, values in scores.items():
        for service, value in values.items():
            # Match the previously audited uniform arithmetic exactly.
            merged[service] += value / len(scores) if mode == "uniform" else weights[key] * value
    return dict(merged), weights


def operation_presence(scores):
    """Structural-frequency control: count operation presence, ignoring latency."""
    presence = defaultdict(float)
    if scores:
        for values in scores.values():
            for service in values:
                presence[service] += 1. / len(scores)
    return dict(presence)
