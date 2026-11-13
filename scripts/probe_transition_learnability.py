#!/usr/bin/env python3
"""Probe whether RE2 onset and recovery transitions are learnable.

This is a diagnostic experiment, not a proposed paper model.  A weighted
logistic classifier is trained independently on each deployment fold using
only training-fold statistics.  Validation labels select the classification
threshold; test labels are never used for fitting or threshold selection.

Two representations are compared:
  * state: the final observed service-feature matrix;
  * state+change: state plus last-minus-first and mean absolute change over
    the five-minute history.

Onset is evaluated only when the decision-minute business state is observed
normal.  Recovery is evaluated only when that state is observed faulty.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
TASKS = ("onset", "recovery")
REPRESENTATIONS = ("state", "state_change")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recall_increments = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recall_increments))


def _metrics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    positives = int(labels.sum())
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    tn = int(np.logical_and(~predictions, labels == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = positives / len(labels) if len(labels) else 0.0
    ap = _average_precision(labels, scores)
    return {
        "samples": int(len(labels)),
        "positives": positives,
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "ap_lift_over_prevalence": ap / prevalence if ap is not None and prevalence else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if not len(labels) or labels.sum() == 0:
        return 0.5
    candidates = np.unique(np.r_[0.01, np.linspace(0.05, 0.95, 91), 0.99, scores])
    best = (-1.0, 0.5)
    for threshold in candidates:
        predictions = scores >= threshold
        tp = int(np.logical_and(predictions, labels == 1).sum())
        fp = int(np.logical_and(predictions, labels == 0).sum())
        fn = int(np.logical_and(~predictions, labels == 1).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Prefer the larger threshold on ties to avoid an unnecessarily broad alarm.
        best = max(best, (f1, float(threshold)))
    return best[1]


def _current_states(dataset: Re2WindowDataset, panel: Path) -> dict[int, int | None]:
    schema = json.loads((panel / "feature_schema.json").read_text())
    rows = _read_csv(Path(schema["source_labels"]))
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


def _representation(dataset: Re2WindowDataset, indices: list[int], name: str) -> np.ndarray:
    rows = []
    for end in indices:
        history = dataset.features[end - dataset.history + 1 : end + 1]
        state = history[-1].reshape(-1)
        if name == "state":
            row = state
        elif name == "state_change":
            delta = (history[-1] - history[0]).reshape(-1)
            mean_abs_change = np.abs(np.diff(history, axis=0)).mean(axis=0).reshape(-1)
            row = np.concatenate((state, delta, mean_abs_change))
        else:
            raise ValueError(name)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _task_indices(
    dataset: Re2WindowDataset,
    states: dict[int, int | None],
    split: list[int],
    task: str,
) -> tuple[list[int], np.ndarray]:
    if task == "onset":
        indices = [end for end in split if states.get(end) == 0]
        labels = np.asarray([dataset.failure_targets[end] for end in indices], dtype=np.int64)
    elif task == "recovery":
        indices = [end for end in split if states.get(end) == 1]
        labels = np.asarray([1 - dataset.failure_targets[end] for end in indices], dtype=np.int64)
    else:
        raise ValueError(task)
    return indices, labels


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


def _fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    model = nn.Linear(train_x.shape[1], 1)
    positives = max(int(train_y.sum()), 1)
    negatives = max(len(train_y) - int(train_y.sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y.astype(np.float32)).unsqueeze(1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_scores = torch.sigmoid(
            model(torch.from_numpy(validation_x))
        ).squeeze(1).numpy()
        test_scores = torch.sigmoid(model(torch.from_numpy(test_x))).squeeze(1).numpy()
    return validation_scores, test_scores


def _run_fold(
    dataset_name: str,
    fold: int,
    task: str,
    representation: str,
    epochs: int,
) -> dict[str, Any]:
    panel = PANELS[dataset_name]
    dataset = Re2WindowDataset(
        panel, history=5, split_policy="deployment", split_fold=fold
    )
    states = _current_states(dataset, panel)
    split_indices = dataset.predefined_splits
    selected = [
        _task_indices(dataset, states, list(split), task) for split in split_indices
    ]
    (train_indices, train_y), (validation_indices, validation_y), (test_indices, test_y) = selected
    if int(train_y.sum()) < 1 or int((train_y == 0).sum()) < 1:
        raise ValueError(
            f"{dataset_name}/{task}/fold{fold} lacks both train classes"
        )
    train_x = _representation(dataset, train_indices, representation)
    validation_x = _representation(dataset, validation_indices, representation)
    test_x = _representation(dataset, test_indices, representation)
    train_x, validation_x, test_x = _standardize(train_x, validation_x, test_x)
    validation_scores, test_scores = _fit_logistic(
        train_x,
        train_y,
        validation_x,
        test_x,
        epochs=epochs,
        seed=20261112 + fold,
    )
    threshold = _best_f1_threshold(validation_y, validation_scores)
    return {
        "fold": fold,
        "train_samples": len(train_y),
        "train_positives": int(train_y.sum()),
        "validation_samples": len(validation_y),
        "validation_positives": int(validation_y.sum()),
        "test_labels": test_y.tolist(),
        "test_scores": test_scores.tolist(),
        "test_predictions": (test_scores >= threshold).tolist(),
        "validation_threshold": threshold,
    }


def _run_setting(
    dataset: str, task: str, representation: str, epochs: int
) -> dict[str, Any]:
    folds = [
        _run_fold(dataset, fold, task, representation, epochs) for fold in range(5)
    ]
    labels = np.asarray(
        [label for fold in folds for label in fold["test_labels"]], dtype=np.int64
    )
    scores = np.asarray(
        [score for fold in folds for score in fold["test_scores"]], dtype=float
    )
    predictions = np.asarray(
        [value for fold in folds for value in fold["test_predictions"]], dtype=bool
    )
    return {
        "metrics": _metrics(labels, scores, predictions),
        "folds": [
            {
                key: value
                for key, value in fold.items()
                if key not in {"test_labels", "test_scores", "test_predictions"}
            }
            for fold in folds
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RE2状态转移可学习性探针",
        "",
        "本实验使用加权Logistic Regression，不是最终模型。所有标准化参数只由训练折估计，",
        "分类阈值只由验证折选择，测试折仅用于最终评价。",
        "",
    ]
    for dataset, tasks in report["results"].items():
        lines.extend([
            f"## RE2-{dataset.upper()}",
            "",
            "| 子任务 | 表示 | 样本 | 正例 | 正例率 | Precision | Recall | F1 | AP | AP/正例率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for task in TASKS:
            for representation in REPRESENTATIONS:
                values = tasks[task][representation]["metrics"]
                lines.append(
                    f"| {task} | {representation} | {values['samples']} | "
                    f"{values['positives']} | {values['prevalence']:.3f} | "
                    f"{values['precision']:.3f} | {values['recall']:.3f} | "
                    f"{values['f1']:.3f} | {values['average_precision']:.3f} | "
                    f"{values['ap_lift_over_prevalence']:.2f}× |"
                )
        lines.append("")
    lines.extend([
        "## 判读规则",
        "",
        "- onset的AP若稳定高于正例率，说明当前正常时的五分钟遥测中存在下一分钟故障信号。",
        "- state+change若稳定优于state，才支持显式增加变化表示；否则双分支缺少直接实验证据。",
        "- recovery正例很少，只作为可学习性线索，不能据此宣称稳定的恢复预测能力。",
        "- 这是线性探针；结果用于决定研究方向，不替代LTFP等完整模型比较。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    results: dict[str, Any] = {}
    for dataset in PANELS:
        results[dataset] = {}
        for task in TASKS:
            results[dataset][task] = {}
            for representation in REPRESENTATIONS:
                print(f"probe dataset={dataset} task={task} representation={representation}")
                results[dataset][task][representation] = _run_setting(
                    dataset, task, representation, args.epochs
                )
    report = {
        "protocol": {
            "history_minutes": 5,
            "split": "deployment five-fold; validation threshold; pooled test predictions",
            "classifier": "weighted logistic regression",
            "epochs": args.epochs,
        },
        "results": results,
    }
    artifact = ROOT / "artifacts/re2_transition_learnability_probe_v1.json"
    document = ROOT / "docs/2026-11-12-transition-learnability-probe.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
