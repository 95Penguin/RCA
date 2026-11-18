#!/usr/bin/env python3
"""Test whether simple causal change features rescue LTFP transition errors.

This is a diagnostic probe, not a proposed model. For each deployment fold, a
weighted linear classifier is fitted on training-fold change features only:

* latest minute minus the preceding minute;
* latest minute minus the first minute in the five-minute history;
* mean absolute consecutive change across the history.

The threshold is selected on the validation fold. The held-out test analysis
then asks whether the probe can discriminate transitions inside the subset
that the existing LTFP predicts as non-transition. No test labels are used for
fitting, normalization, or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset  # noqa: E402


PANELS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}
LTFP_RUN = (
    "re2_{dataset}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1"
)
TASKS = ("onset", "recovery")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _source_labels(panel: Path) -> Path:
    schema = json.loads((panel / "feature_schema.json").read_text())
    source = Path(schema["source_labels"])
    if source.is_file():
        return source
    portable = ROOT / "data" / "processed" / source.parent.name / source.name
    if portable.is_file():
        return portable
    raise FileNotFoundError(f"source labels missing at {source} and {portable}")


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recalled = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recalled))


def _metrics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    if not len(labels):
        return {
            "samples": 0,
            "positives": 0,
            "prevalence": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
            "ap_minus_prevalence": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.logical_and(labels, predictions).sum())
    fp = int(np.logical_and(~labels, predictions).sum())
    fn = int(np.logical_and(labels, ~predictions).sum())
    tn = int(np.logical_and(~labels, ~predictions).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(labels.mean())
    ap = _average_precision(labels.astype(int), scores)
    return {
        "samples": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "ap_minus_prevalence": ap - prevalence if ap is not None else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if not len(labels) or np.unique(labels).size < 2:
        return 0.5
    candidates = np.unique(np.r_[0.01, np.linspace(0.05, 0.95, 91), 0.99, scores])
    best = (-1.0, 0.5)
    for threshold in candidates:
        values = _metrics(labels, scores, scores >= threshold)
        best = max(best, (float(values["f1"]), float(threshold)))
    return best[1]


def _current_states(dataset: Re2WindowDataset, panel: Path) -> dict[int, int | None]:
    rows = _read_csv(_source_labels(panel))
    by_target = {row["target_minute_utc"]: row for row in rows}
    states: dict[int, int | None] = {}
    for end in dataset.sample_indices:
        row = by_target.get(dataset.timestamps[end])
        if (
            row is None
            or row["case_id"] != dataset.case_ids[end]
            or int(row["entry_observed"]) != 1
        ):
            states[end] = None
        else:
            states[end] = int(row["failure_next_minute"])
    return states


def _task_indices(
    dataset: Re2WindowDataset,
    states: dict[int, int | None],
    split: list[int],
    task: str,
) -> tuple[list[int], np.ndarray]:
    current = 0 if task == "onset" else 1
    indices = [end for end in split if states.get(end) == current]
    if task == "onset":
        labels = np.asarray(
            [dataset.failure_targets[end] for end in indices], dtype=np.int64
        )
    else:
        labels = np.asarray(
            [1 - dataset.failure_targets[end] for end in indices], dtype=np.int64
        )
    return indices, labels


def _change_features(dataset: Re2WindowDataset, indices: list[int]) -> np.ndarray:
    rows = []
    for end in indices:
        history = dataset.features[end - dataset.history + 1 : end + 1]
        latest_delta = (history[-1] - history[-2]).reshape(-1)
        window_delta = (history[-1] - history[0]).reshape(-1)
        mean_abs_change = np.abs(np.diff(history, axis=0)).mean(axis=0).reshape(-1)
        rows.append(np.concatenate((latest_delta, window_delta, mean_abs_change)))
    return np.asarray(rows, dtype=np.float32)


def _standardize(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return tuple(
        np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        for values in (train, validation, test)
    )


def _fit_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    model = nn.Linear(train_x.shape[1], 1)
    positives = max(int(train_y.sum()), 1)
    negatives = max(len(train_y) - int(train_y.sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y.astype(np.float32)).unsqueeze(1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        validation = torch.sigmoid(model(torch.from_numpy(validation_x))).squeeze(1)
        test = torch.sigmoid(model(torch.from_numpy(test_x))).squeeze(1)
    return validation.numpy(), test.numpy()


def _ltfp_scores(dataset: str, fold: int, task: str) -> dict[str, float]:
    path = ROOT / "artifacts" / LTFP_RUN.format(dataset=dataset, fold=fold) / "predictions.jsonl"
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            probability = float(row["failure_probability"])
            scores[row["input_minute_utc"]] = probability if task == "onset" else 1 - probability
    return scores


def _run_fold(
    dataset_name: str, fold: int, task: str, epochs: int, seed: int
) -> dict[str, Any]:
    panel = PANELS[dataset_name]
    dataset = Re2WindowDataset(
        panel, history=5, split_policy="deployment", split_fold=fold
    )
    states = _current_states(dataset, panel)
    selected = [
        _task_indices(dataset, states, list(split), task)
        for split in dataset.predefined_splits
    ]
    (train_indices, train_y), (validation_indices, validation_y), (test_indices, test_y) = selected
    train_x = _change_features(dataset, train_indices)
    validation_x = _change_features(dataset, validation_indices)
    test_x = _change_features(dataset, test_indices)
    train_x, validation_x, test_x = _standardize(train_x, validation_x, test_x)
    validation_scores, test_scores = _fit_linear(
        train_x,
        train_y,
        validation_x,
        test_x,
        seed=seed + fold,
        epochs=epochs,
    )
    threshold = _best_f1_threshold(validation_y, validation_scores)
    existing = _ltfp_scores(dataset_name, fold, task)
    ltfp_scores = np.asarray(
        [existing[dataset.timestamps[end]] for end in test_indices], dtype=float
    )
    return {
        "fold": fold,
        "case_ids": [dataset.case_ids[end] for end in test_indices],
        "labels": test_y.tolist(),
        "change_scores": test_scores.tolist(),
        "change_predictions": (test_scores >= threshold).tolist(),
        "ltfp_scores": ltfp_scores.tolist(),
        "ltfp_predictions": (ltfp_scores >= 0.5).tolist(),
        "validation_threshold": threshold,
        "train_samples": len(train_y),
        "train_positives": int(train_y.sum()),
        "validation_samples": len(validation_y),
        "validation_positives": int(validation_y.sum()),
    }


def _bootstrap_conditional(records: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    by_fold_case: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        by_fold_case[row["fold"]][row["case_id"]].append(row)
    rng = np.random.default_rng(20261117)
    advantages: list[float] = []
    for _ in range(samples):
        selected: list[dict[str, Any]] = []
        for fold, cases in by_fold_case.items():
            names = sorted(cases)
            chosen = rng.choice(names, size=len(names), replace=True)
            for name in chosen:
                selected.extend(cases[str(name)])
        labels = np.asarray([row["label"] for row in selected], dtype=int)
        if not len(labels) or labels.sum() == 0:
            continue
        scores = np.asarray([row["change_score"] for row in selected], dtype=float)
        ap = _average_precision(labels, scores)
        advantages.append(float(ap - labels.mean()))
    return {
        "resamples": len(advantages),
        "ap_minus_prevalence_95pct_ci": (
            [float(value) for value in np.percentile(advantages, (2.5, 97.5))]
            if advantages
            else None
        ),
    }


def _topk_ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, kind="stable")
    positives = int(labels.sum())
    result: dict[str, Any] = {}
    for requested in (10, 25, 50):
        k = min(requested, len(labels))
        hits = int(labels[order[:k]].sum())
        result[f"top_{requested}"] = {
            "evaluated": k,
            "hits": hits,
            "precision": hits / k if k else None,
            "recall": hits / positives if positives else None,
        }
    if positives:
        hits = int(labels[order[:positives]].sum())
        result["top_positive_count"] = {
            "evaluated": positives,
            "hits": hits,
            "precision": hits / positives,
            "recall": hits / positives,
        }
    return result


def _summarize(folds: list[dict[str, Any]], bootstraps: int) -> dict[str, Any]:
    records = []
    for fold in folds:
        for values in zip(
            fold["case_ids"],
            fold["labels"],
            fold["change_scores"],
            fold["change_predictions"],
            fold["ltfp_scores"],
            fold["ltfp_predictions"],
        ):
            case_id, label, change_score, change_prediction, ltfp_score, ltfp_prediction = values
            records.append({
                "fold": fold["fold"],
                "case_id": case_id,
                "label": int(label),
                "change_score": float(change_score),
                "change_prediction": bool(change_prediction),
                "ltfp_score": float(ltfp_score),
                "ltfp_prediction": bool(ltfp_prediction),
            })
    labels = np.asarray([row["label"] for row in records], dtype=int)
    change_scores = np.asarray([row["change_score"] for row in records], dtype=float)
    change_predictions = np.asarray(
        [row["change_prediction"] for row in records], dtype=bool
    )
    ltfp_scores = np.asarray([row["ltfp_score"] for row in records], dtype=float)
    ltfp_predictions = np.asarray(
        [row["ltfp_prediction"] for row in records], dtype=bool
    )
    missed = ~ltfp_predictions
    conditional_records = [row for row in records if not row["ltfp_prediction"]]
    conditional_labels = labels[missed]
    conditional_scores = change_scores[missed]
    conditional_predictions = change_predictions[missed]
    rescued = int(
        np.logical_and.reduce((labels == 1, ~ltfp_predictions, change_predictions)).sum()
    )
    new_errors = int(
        np.logical_and.reduce((labels == 0, ~ltfp_predictions, change_predictions)).sum()
    )
    rescue_precision = rescued / (rescued + new_errors) if rescued + new_errors else None
    return {
        "ltfp": _metrics(labels, ltfp_scores, ltfp_predictions),
        "change_only": _metrics(labels, change_scores, change_predictions),
        "inside_ltfp_negative_subset": {
            **_metrics(
                conditional_labels,
                conditional_scores,
                conditional_predictions,
            ),
            "rescued_ltfp_errors": rescued,
            "new_errors_if_used_as_override": new_errors,
            "rescue_precision": rescue_precision,
            "ranking": _topk_ranking(conditional_labels, conditional_scores),
            "case_bootstrap": _bootstrap_conditional(
                conditional_records, bootstraps
            ),
        },
        "folds": [
            {
                key: value
                for key, value in fold.items()
                if key not in {
                    "case_ids",
                    "labels",
                    "change_scores",
                    "change_predictions",
                    "ltfp_scores",
                    "ltfp_predictions",
                }
            }
            for fold in folds
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.bootstraps < 1:
        parser.error("epochs and bootstraps must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    results: dict[str, Any] = {}
    for dataset in PANELS:
        results[dataset] = {}
        for task in TASKS:
            print(f"audit dataset={dataset} task={task}", flush=True)
            folds = [
                _run_fold(dataset, fold, task, args.epochs, args.seed)
                for fold in range(5)
            ]
            results[dataset][task] = _summarize(folds, args.bootstraps)
    report = {
        "protocol": {
            "task": "change-only rescue of existing LTFP transition errors",
            "features": [
                "latest minus previous minute",
                "latest minus first history minute",
                "mean absolute consecutive change",
            ],
            "split": "deployment five-fold",
            "normalization": "training fold only",
            "threshold": "validation-fold best F1",
            "ltfp_threshold": 0.5,
            "epochs": args.epochs,
            "seed": args.seed,
            "case_bootstraps": args.bootstraps,
        },
        "results": results,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"saved {args.output}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
