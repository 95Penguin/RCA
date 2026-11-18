from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auprc: float
    positives: int


@dataclass(frozen=True)
class BinaryMetricsV2:
    """Submission metrics for one discrete failure horizon.

    ``BinaryMetrics`` is intentionally kept unchanged because the legacy
    experiment scripts serialize its fields.  The v2 protocol uses this
    expanded record and treats single-class evaluation sets explicitly.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    auprc: float
    mcc: float
    balanced_accuracy: float
    brier: float
    ece: float
    prevalence: float
    positives: int
    negatives: int


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auprc=average_precision(y_true, scores),
        positives=int(y_true.sum()),
    )


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y)
    precision = tp / (np.arange(len(sorted_y)) + 1)
    return float((precision * sorted_y).sum() / positives)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error.

    Empty bins have zero weight.  Probabilities are clipped only for numerical
    safety; this does not change thresholded predictions in the valid range.
    """

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probabilities = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, max(int(n_bins), 1) + 1)
    # Include p=1 in the final bin.
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), len(edges) - 2)
    ece = 0.0
    for idx in range(len(edges) - 1):
        mask = assignments == idx
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def binary_metrics_v2(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_calibration_bins: int = 10,
) -> BinaryMetricsV2:
    """Compute discrimination, calibration, and imbalance-aware metrics."""

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal length")
    pred = (probabilities >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mcc_denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0.0
    probabilities_clipped = np.clip(probabilities, 0.0, 1.0)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    return BinaryMetricsV2(
        accuracy=(tp + tn) / max(len(y_true), 1),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auprc=average_precision(y_true, probabilities),
        mcc=float(mcc),
        balanced_accuracy=float((recall + specificity) / 2.0),
        brier=float(np.mean((probabilities_clipped - y_true) ** 2)) if len(y_true) else 0.0,
        ece=expected_calibration_error(y_true, probabilities_clipped, n_bins=n_calibration_bins),
        prevalence=float(positives / max(len(y_true), 1)),
        positives=positives,
        negatives=negatives,
    )


def tune_shared_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> float:
    """Tune one validation-only threshold shared by every horizon.

    A shared threshold preserves alarm nesting whenever probabilities are
    monotone.  The objective is macro F1 over horizons; ties prefer the larger
    threshold to avoid gratuitous alerts.
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape != probabilities.shape or y_true.ndim != 2:
        raise ValueError("y_true and probabilities must be equally shaped 2-D arrays")
    if label_mask is None:
        label_mask = np.ones_like(y_true, dtype=bool)
    else:
        label_mask = np.asarray(label_mask, dtype=bool)
        if label_mask.shape != y_true.shape:
            raise ValueError("label_mask must match y_true")
    flattened = probabilities[label_mask]
    if len(flattened) == 0:
        return 0.5
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0]),
                np.quantile(flattened, np.linspace(0.01, 0.99, 199)),
            ]
        )
    )
    best = (float("-inf"), float("-inf"))
    best_threshold = 0.5
    for threshold in candidates:
        macro_f1 = float(
            np.mean(
                [
                    binary_metrics_v2(
                        y_true[label_mask[:, idx], idx],
                        probabilities[label_mask[:, idx], idx],
                        float(threshold),
                    ).f1
                    for idx in range(y_true.shape[1])
                    if label_mask[:, idx].any()
                ]
            )
        )
        candidate = (macro_f1, float(threshold))
        if candidate > best:
            best = candidate
            best_threshold = float(threshold)
    return best_threshold


def fit_shared_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit one positive temperature on validation data only.

    A single temperature retains ordering both across examples and across the
    nested horizon logits.  SciPy is used when available; the deterministic
    grid fallback keeps the artifact runnable in a minimal environment.
    """

    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if logits.shape != y_true.shape:
        raise ValueError("logits and y_true must have equal shape")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        scaled = np.clip(logits / temperature, -50.0, 50.0)
        # Stable BCE-with-logits.
        loss = np.maximum(scaled, 0.0) - scaled * y_true + np.log1p(np.exp(-np.abs(scaled)))
        return float(loss.mean())

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-8})
        if result.success and np.isfinite(result.x):
            return float(math.exp(float(result.x)))
    except (ImportError, ValueError, FloatingPointError):
        pass
    grid = np.linspace(-4.0, 4.0, 321)
    return float(math.exp(float(min(grid, key=objective))))


def sigmoid_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temperature = max(float(temperature), 1e-8)
    values = np.clip(np.asarray(logits, dtype=float) / temperature, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def evaluate_multi_horizon_v2(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    horizons: Sequence[float],
    threshold: float,
    label_mask: np.ndarray | None = None,
) -> list[dict]:
    """Evaluate every horizon with one shared threshold."""

    if label_mask is None:
        label_mask = np.ones_like(y_true, dtype=bool)
    else:
        label_mask = np.asarray(label_mask, dtype=bool)
        if label_mask.shape != np.asarray(y_true).shape:
            raise ValueError("label_mask must match y_true")
    rows = []
    for idx, horizon in enumerate(horizons):
        valid = label_mask[:, idx]
        metric = binary_metrics_v2(y_true[valid, idx], probabilities[valid, idx], threshold)
        row = {"horizon": float(horizon), "threshold": float(threshold)}
        row.update(metric.__dict__)
        row["n_valid"] = int(valid.sum())
        rows.append(row)
    return rows


def assert_nested_probabilities(probabilities: np.ndarray, atol: float = 1e-7) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probabilities must be a 2-D array")
    if values.shape[1] > 1 and np.any(np.diff(values, axis=1) < -float(atol)):
        raise ValueError("multi-horizon probabilities are not monotone")


def paired_scenario_bootstrap(
    candidate: Sequence[float],
    reference: Sequence[float],
    scenario_ids: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 7,
) -> dict[str, float]:
    """Paired bootstrap over scenarios, never over overlapping windows."""

    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scenario_ids = np.asarray(scenario_ids, dtype=str)
    if not (len(candidate) == len(reference) == len(scenario_ids)):
        raise ValueError("candidate, reference, and scenario_ids must have equal length")
    unique = np.unique(scenario_ids)
    if len(unique) == 0:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_scenarios": 0}
    per_scenario = np.asarray(
        [np.nanmean((candidate - reference)[scenario_ids == scenario]) for scenario in unique],
        dtype=float,
    )
    per_scenario = per_scenario[np.isfinite(per_scenario)]
    if len(per_scenario) == 0:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_scenarios": 0}
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        samples[idx] = float(rng.choice(per_scenario, size=len(per_scenario), replace=True).mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "difference": float(per_scenario.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_scenarios": int(len(per_scenario)),
    }


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(scores)) == 1:
        return float(scores[0])
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        f1 = binary_metrics(y_true, scores, float(threshold)).f1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_multi_horizon(
    y_true: np.ndarray,
    scores: np.ndarray,
    horizons: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict]:
    rows = []
    for idx, horizon in enumerate(horizons):
        metrics = binary_metrics(y_true[:, idx], scores[:, idx], thresholds[idx])
        rows.append(
            {
                "horizon": horizon,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "auprc": metrics.auprc,
                "positives": metrics.positives,
                "threshold": float(thresholds[idx]),
            }
        )
    return rows


def root_topk_accuracy(
    rankings: Sequence[Sequence[str]],
    labels: Sequence[str | None],
    k: int,
) -> float:
    hits = 0
    total = 0
    for ranking, label in zip(rankings, labels):
        if not label:
            continue
        total += 1
        normalized_label = _normalize(label)
        topk = [_normalize(item) for item in ranking[:k]]
        if normalized_label in topk:
            hits += 1
            continue
        label_service = normalized_label.split("_", 1)[0]
        if any(item.split("_", 1)[0] == label_service for item in topk):
            hits += 1
    return hits / max(total, 1)


def _normalize(value: str) -> str:
    return value.strip().lower().replace(".", "_").replace("/", "_")
