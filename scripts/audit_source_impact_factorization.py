#!/usr/bin/env python3
"""Leakage-safe feasibility audit for source--impact factorization on RE2.

This is deliberately a probe rather than a new neural architecture.  It asks:

1. Can a train-only normal reference identify the injected source even when the
   injection does not produce a user-visible impact?
2. Does separating source evidence from business-impact evidence improve the
   next-minute failure decision over either evidence family alone?

The deployment folds, feature windows, and next-minute targets are identical to
the main LTFP experiments.  Injection metadata is used only as auxiliary source
supervision on post-injection windows and for diagnostic subgroup reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset
from training.metrics import classification_metrics


DATASETS = {
    "ob": {
        "panel": ROOT / "artifacts/re2_ob_panel_fair_v1",
        "labels": ROOT / "data/processed/re2_ob_labels_v1",
    },
    "tt": {
        "panel": ROOT / "artifacts/re2_tt_panel_fair_v1",
        "labels": ROOT / "data/processed/re2_tt_labels_v1",
    },
}
RESOURCE_NAMES = ("metric_cpu", "metric_mem", "metric_diskio", "metric_socket")
OBSERVED_NAMES = ("observed_cpu", "observed_mem", "observed_diskio", "observed_socket")
MODEL_NAMES = ("source_only", "impact_only", "additive", "factorized")


def _read_metadata(dataset_name: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    directory = DATASETS[dataset_name]["labels"]
    with (directory / "cases.csv").open(newline="", encoding="utf-8-sig") as handle:
        cases = {row["case_id"]: row for row in csv.DictReader(handle)}
    phase_by_time: dict[str, str] = {}
    with (directory / "next_minute_labels.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            phase_by_time[row["input_minute_utc"]] = row["phase"]
    return cases, phase_by_time


def _fit_normal_reference(
    dataset: Re2WindowDataset,
    train_ends: list[int],
    phase_by_time: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    resource = [dataset.feature_columns.index(name) for name in RESOURCE_NAMES]
    observed = [dataset.feature_columns.index(name) for name in OBSERVED_NAMES]
    train_cases = {dataset.case_ids[end] for end in train_ends}
    minute_indices = [
        index
        for index, timestamp in enumerate(dataset.timestamps)
        if dataset.case_ids[index] in train_cases and phase_by_time.get(timestamp) == "pre"
    ]
    values = dataset.features[minute_indices][:, :, resource]
    masks = dataset.features[minute_indices][:, :, observed] > 0
    center = np.zeros((len(dataset.services), len(resource)), dtype=np.float32)
    scale = np.ones_like(center)
    for service in range(len(dataset.services)):
        for metric in range(len(resource)):
            observed_values = values[:, service, metric][masks[:, service, metric]]
            if not len(observed_values):
                continue
            median = float(np.median(observed_values))
            mad = float(np.median(np.abs(observed_values - median)) * 1.4826)
            standard = float(observed_values.std())
            center[service, metric] = median
            scale[service, metric] = (
                mad if mad > 1e-3 else standard if standard > 1e-3 else 1.0
            )
    return center, scale


def _source_summary(
    dataset: Re2WindowDataset,
    end: int,
    reference: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    resource = [dataset.feature_columns.index(name) for name in RESOURCE_NAMES]
    observed = [dataset.feature_columns.index(name) for name in OBSERVED_NAMES]
    window = dataset.features[end - dataset.history + 1 : end + 1]
    values = window[:, :, resource]
    masks = window[:, :, observed] > 0
    center, scale = reference
    deviation = np.clip(np.abs((values - center[None]) / scale[None]), 0.0, 10.0)
    deviation = np.where(masks, deviation, 0.0)
    count = np.maximum(masks.sum(axis=0), 1)
    history_mean = deviation.sum(axis=0) / count
    summary = np.concatenate(
        [deviation[-1], deviation.max(axis=0), history_mean], axis=-1
    ).astype(np.float32)
    return summary, deviation.mean(axis=-1)


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _fit_source_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    case_ids: list[str],
    l2: float = 1e-2,
) -> np.ndarray:
    """Fit one shared linear scorer over services with case-balanced CE."""
    counts = {case_id: case_ids.count(case_id) for case_id in set(case_ids)}
    weights = np.asarray([1.0 / counts[case_id] for case_id in case_ids], np.float32)
    weights /= weights.mean()
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.long)
    sample_weight = torch.tensor(weights, dtype=torch.float64)
    parameter = torch.zeros(features.shape[-1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameter], lr=1.0, max_iter=150, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = torch.einsum("wnf,f->wn", x, parameter)
        losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (losses * sample_weight).mean() + l2 * parameter.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return parameter.detach().cpu().numpy().astype(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _source_decision_features(
    summaries: np.ndarray,
    raw_history: np.ndarray,
    standardized: np.ndarray,
    source_weights: np.ndarray,
) -> np.ndarray:
    logits = np.einsum("wnf,f->wn", standardized, source_weights)
    probabilities = _softmax(logits)
    sorted_logits = np.sort(logits, axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    entropy /= math.log(probabilities.shape[1])
    raw_service = summaries.mean(axis=-1)
    sorted_raw = np.sort(raw_service, axis=1)
    return np.column_stack(
        [
            raw_service.max(axis=1),
            raw_service.mean(axis=1),
            sorted_raw[:, -1] - sorted_raw[:, -2],
            raw_history[:, -1].max(axis=1),
            logits.max(axis=1),
            sorted_logits[:, -1] - sorted_logits[:, -2],
            sorted_probabilities[:, -1],
            sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
            entropy,
        ]
    ).astype(np.float32)


def _aggregate_series(
    window: np.ndarray, columns: tuple[str, ...]
) -> np.ndarray:
    index = {name: position for position, name in enumerate(columns)}
    trace_observed = window[:, :, index["trace_span_count"]] > 0
    entry_observed = window[:, :, index["observed_entry_p95"]] > 0
    baseline_observed = window[:, :, index["observed_entry_baseline"]] > 0

    def masked_max(name: str, mask: np.ndarray) -> np.ndarray:
        values = np.where(mask, window[:, :, index[name]], -np.inf)
        result = values.max(axis=1)
        return np.where(np.isfinite(result), result, 0.0)

    series = np.column_stack(
        [
            masked_max("entry_p95_ms", entry_observed),
            masked_max("entry_degradation_score_rolling", baseline_observed),
            window[:, :, index["entry_span_count"]].sum(axis=1),
            masked_max("trace_duration_mean_ms", trace_observed),
            window[:, :, index["trace_span_count"]].sum(axis=1),
            window[:, :, index["trace_error_count"]].sum(axis=1),
            window[:, :, index["log_count"]].sum(axis=1),
            window[:, :, index["log_error_count"]].sum(axis=1),
        ]
    )
    return series.astype(np.float32)


def _impact_features(dataset: Re2WindowDataset, end: int) -> np.ndarray:
    window = dataset.features[end - dataset.history + 1 : end + 1]
    series = _aggregate_series(window, dataset.feature_columns)
    return np.concatenate(
        [series[-1], series.max(axis=0), series.mean(axis=0), series[-1] - series[0]]
    ).astype(np.float32)


def _factorized_features(source: np.ndarray, impact: np.ndarray) -> np.ndarray:
    # Raw source magnitude and learned source margin gate the impact evidence.
    interactions = np.concatenate(
        [source[:, [0]] * impact, source[:, [5]] * impact], axis=1
    )
    return np.concatenate([source, impact, interactions], axis=1)


def _case_balanced_weights(case_ids: list[str], labels: np.ndarray) -> np.ndarray:
    counts = {case_id: case_ids.count(case_id) for case_id in set(case_ids)}
    weights = np.asarray([1.0 / counts[case_id] for case_id in case_ids], np.float64)
    positive_mass = weights[labels == 1].sum()
    negative_mass = weights[labels == 0].sum()
    if positive_mass > 0 and negative_mass > 0:
        weights[labels == 1] *= negative_mass / positive_mass
    return (weights / weights.mean()).astype(np.float32)


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    l2: float = 1e-2,
) -> tuple[np.ndarray, float]:
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    sample_weight = torch.tensor(
        _case_balanced_weights(case_ids, labels), dtype=torch.float64
    )
    parameter = torch.zeros(features.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameter, bias], lr=1.0, max_iter=200, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ parameter + bias
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        loss = (losses * sample_weight).mean() + l2 * parameter.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return parameter.detach().cpu().numpy().astype(np.float32), float(bias.detach())


def _predict_logistic(features: np.ndarray, model: tuple[np.ndarray, float]) -> np.ndarray:
    weights, bias = model
    logits = np.clip(features @ weights + bias, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = probabilities >= threshold
        tp = int(np.logical_and(predictions, labels == 1).sum())
        fp = int(np.logical_and(predictions, labels == 0).sum())
        fn = int(np.logical_and(~predictions, labels == 1).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        best = max(best, (f1, float(threshold)))
    return best[1]


def _load_ltfp_probabilities(dataset_name: str, fold: int) -> dict[str, float]:
    path = ROOT / (
        f"artifacts/re2_{dataset_name}_ltfp_deployment_fold{fold}_"
        "fair_30ep_seed42_baseline_v1/predictions.jsonl"
    )
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[row["input_minute_utc"]] = float(row["failure_probability"])
    return result


def _rank(true_index: int, scores: np.ndarray) -> int:
    order = np.argsort(-scores, kind="stable")
    return int(np.flatnonzero(order == true_index)[0] + 1)


def _run_fold(dataset_name: str, fold: int) -> dict[str, object]:
    torch.manual_seed(42)
    dataset = Re2WindowDataset(
        DATASETS[dataset_name]["panel"], history=5,
        split_policy="deployment", split_fold=fold,
    )
    cases, phase_by_time = _read_metadata(dataset_name)
    train_ends, validation_ends, test_ends = dataset.predefined_splits
    reference = _fit_normal_reference(dataset, train_ends, phase_by_time)
    all_ends = train_ends + validation_ends + test_ends
    source_summaries = []
    raw_histories = []
    impact = []
    for end in all_ends:
        summary, raw_history = _source_summary(dataset, end, reference)
        source_summaries.append(summary)
        raw_histories.append(raw_history)
        impact.append(_impact_features(dataset, end))
    source_summaries = np.stack(source_summaries)
    raw_histories = np.stack(raw_histories)
    impact = np.stack(impact)
    row_by_end = {end: row for row, end in enumerate(all_ends)}

    source_train = [
        end for end in train_ends if phase_by_time.get(dataset.timestamps[end]) == "post"
    ]
    source_mean, source_scale = _fit_standardizer(
        source_summaries[[row_by_end[end] for end in source_train]].reshape(
            -1, source_summaries.shape[-1]
        )
    )
    standardized_source = (source_summaries - source_mean) / source_scale
    service_index = {name: index for index, name in enumerate(dataset.services)}
    source_targets = np.asarray(
        [service_index[cases[dataset.case_ids[end]]["root_service"]] for end in source_train]
    )
    source_weights = _fit_source_ranker(
        standardized_source[[row_by_end[end] for end in source_train]],
        source_targets,
        [dataset.case_ids[end] for end in source_train],
    )
    source_decision = _source_decision_features(
        source_summaries, raw_histories, standardized_source, source_weights
    )

    source_cases = []
    post_test_by_case: dict[str, list[int]] = defaultdict(list)
    for end in test_ends:
        if phase_by_time.get(dataset.timestamps[end]) == "post":
            post_test_by_case[dataset.case_ids[end]].append(end)
    for case_id, ends in sorted(post_test_by_case.items()):
        rows = [row_by_end[end] for end in ends]
        true = service_index[cases[case_id]["root_service"]]
        raw_scores = source_summaries[rows].mean(axis=(0, 2))
        learned_scores = np.einsum(
            "wnf,f->wn", standardized_source[rows], source_weights
        ).mean(axis=0)
        source_cases.append(
            {
                "case_id": case_id,
                "impact": int(cases[case_id]["impact_minutes"]) > 0,
                "post_windows": len(ends),
                "raw_rank": _rank(true, raw_scores),
                "learned_rank": _rank(true, learned_scores),
            }
        )

    partitions = {
        "train": train_ends, "validation": validation_ends, "test": test_ends
    }
    matrices = {
        "source_only": source_decision,
        "impact_only": impact,
        "additive": np.concatenate([source_decision, impact], axis=1),
        "factorized": _factorized_features(source_decision, impact),
    }
    model_results = {}
    for name, matrix in matrices.items():
        train_rows = [row_by_end[end] for end in train_ends]
        validation_rows = [row_by_end[end] for end in validation_ends]
        test_rows = [row_by_end[end] for end in test_ends]
        mean, scale = _fit_standardizer(matrix[train_rows])
        standardized = (matrix - mean) / scale
        train_labels = dataset.failure_targets[train_ends]
        model = _fit_logistic(
            standardized[train_rows], train_labels,
            [dataset.case_ids[end] for end in train_ends],
        )
        validation_probabilities = _predict_logistic(standardized[validation_rows], model)
        threshold = _best_f1_threshold(
            dataset.failure_targets[validation_ends], validation_probabilities
        )
        test_probabilities = _predict_logistic(standardized[test_rows], model)
        model_results[name] = {
            "threshold": threshold,
            "probabilities": test_probabilities.tolist(),
        }

    ltfp = _load_ltfp_probabilities(dataset_name, fold)
    ltfp_probabilities = [ltfp[dataset.timestamps[end]] for end in test_ends]
    return {
        "dataset": dataset_name,
        "fold": fold,
        "counts": {name: len(ends) for name, ends in partitions.items()},
        "test_ends": test_ends,
        "test_timestamps": [dataset.timestamps[end] for end in test_ends],
        "test_labels": dataset.failure_targets[test_ends].tolist(),
        "test_cases": [dataset.case_ids[end] for end in test_ends],
        "test_phases": [phase_by_time.get(dataset.timestamps[end], "") for end in test_ends],
        "test_event_ids": [dataset.event_ids[end] for end in test_ends],
        "case_has_impact": [
            int(cases[dataset.case_ids[end]]["impact_minutes"]) > 0 for end in test_ends
        ],
        "source_cases": source_cases,
        "models": model_results,
        "ltfp": {"threshold": 0.5, "probabilities": ltfp_probabilities},
    }


def _source_summary_report(rows: list[dict[str, object]]) -> dict[str, object]:
    report = {}
    for group, selected in (
        ("all", rows),
        ("impact", [row for row in rows if row["impact"]]),
        ("no_impact", [row for row in rows if not row["impact"]]),
    ):
        report[group] = {"cases": len(selected)}
        for method in ("raw", "learned"):
            ranks = np.asarray([row[f"{method}_rank"] for row in selected], dtype=float)
            report[group][method] = {
                "hit_at_1": float((ranks <= 1).mean()) if len(ranks) else None,
                "hit_at_3": float((ranks <= 3).mean()) if len(ranks) else None,
                "mrr": float((1.0 / ranks).mean()) if len(ranks) else None,
                "mean_rank": float(ranks.mean()) if len(ranks) else None,
            }
    return report


def _model_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    phases: np.ndarray,
    has_impact: np.ndarray,
    event_ids: np.ndarray,
) -> dict[str, object]:
    # Fold-specific validation thresholds are already aligned per OOF row.
    predictions = probabilities >= thresholds
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    report: dict[str, object] = {
        "fixed_0_5": classification_metrics(labels, probabilities),
        "validation_threshold": {
            "mean_threshold": float(thresholds.mean()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_positives": fp,
        },
    }
    subgroups = {
        "pre_negative": np.logical_and(phases == "pre", labels == 0),
        "no_impact_post_negative": np.logical_and.reduce(
            (phases == "post", ~has_impact, labels == 0)
        ),
        "impact_case_post_negative": np.logical_and.reduce(
            (phases == "post", has_impact, labels == 0)
        ),
    }
    report["negative_subgroups"] = {}
    for name, mask in subgroups.items():
        report["negative_subgroups"][name] = {
            "samples": int(mask.sum()),
            "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
            "false_positive_rate_0_5": float((probabilities[mask] >= 0.5).mean())
            if mask.any() else None,
            "false_positive_rate_validation_threshold": float(predictions[mask].mean())
            if mask.any() else None,
        }
    onset_indices = []
    for event_id in sorted(set(event_ids) - {""}):
        members = np.flatnonzero(event_ids == event_id)
        positive_members = members[labels[members] == 1]
        if len(positive_members):
            onset_indices.append(int(positive_members[0]))
    report["event_onset"] = {
        "events": len(onset_indices),
        "recall_0_5": float((probabilities[onset_indices] >= 0.5).mean())
        if onset_indices else None,
        "recall_validation_threshold": float(predictions[onset_indices].mean())
        if onset_indices else None,
    }
    return report


def _summarize(dataset_name: str, folds: list[dict[str, object]]) -> dict[str, object]:
    source_rows = [row for fold in folds for row in fold["source_cases"]]
    labels = np.concatenate([np.asarray(fold["test_labels"]) for fold in folds])
    phases = np.concatenate([np.asarray(fold["test_phases"]) for fold in folds])
    has_impact = np.concatenate([np.asarray(fold["case_has_impact"]) for fold in folds])
    event_ids = np.concatenate([np.asarray(fold["test_event_ids"]) for fold in folds])
    models = {}
    for name in (*MODEL_NAMES, "ltfp"):
        probabilities = np.concatenate(
            [np.asarray(fold[name]["probabilities"] if name == "ltfp" else
                        fold["models"][name]["probabilities"]) for fold in folds]
        )
        thresholds = np.concatenate(
            [np.full(len(fold["test_labels"]),
                     fold[name]["threshold"] if name == "ltfp" else
                     fold["models"][name]["threshold"])
             for fold in folds]
        )
        models[name] = _model_report(
            labels, probabilities, thresholds, phases, has_impact, event_ids
        )
    return {
        "dataset": dataset_name,
        "test_windows": int(labels.size),
        "positive_windows": int(labels.sum()),
        "source_localization": _source_summary_report(source_rows),
        "failure_prediction": models,
    }


def _print_summary(summary: dict[str, object]) -> None:
    print(f"\n{summary['dataset'].upper()}  source localization")
    for group, values in summary["source_localization"].items():
        print(
            f"  {group:9s} n={values['cases']:2d}  "
            f"raw H@1={values['raw']['hit_at_1']:.3f} MRR={values['raw']['mrr']:.3f}  "
            f"learned H@1={values['learned']['hit_at_1']:.3f} "
            f"MRR={values['learned']['mrr']:.3f}"
        )
    print(f"{summary['dataset'].upper()}  next-minute prediction")
    for name, values in summary["failure_prediction"].items():
        fixed = values["fixed_0_5"]
        subgroup = values["negative_subgroups"]["no_impact_post_negative"]
        print(
            f"  {name:11s} AP={fixed['average_precision']:.3f} "
            f"F1@.5={fixed['f1_at_0_5']:.3f} "
            f"no-impact-post FP={subgroup['false_positive_rate_0_5']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["ob", "tt"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/source_impact_feasibility.json")
    args = parser.parse_args()
    result = {
        "audit": "source_impact_factorization_feasibility",
        "history_minutes": 5,
        "forecast_horizon_minutes": 1,
        "split_policy": "deployment_5fold_oof",
        "source_active_proxy": "phase=post; injection metadata is auxiliary supervision only",
        "datasets": {},
    }
    for dataset_name in args.datasets:
        folds = []
        for fold in range(5):
            print(f"running {dataset_name} fold {fold}", flush=True)
            folds.append(_run_fold(dataset_name, fold))
        summary = _summarize(dataset_name, folds)
        result["datasets"][dataset_name] = {"summary": summary, "folds": folds}
        _print_summary(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
