#!/usr/bin/env python3
"""Audit RE2 state transitions and the last-state persistence baseline.

The script reuses the disjoint test windows from the five deployment folds of
the existing LTFP run.  It does not train or tune a model.  The persistence
baseline predicts the next target state with the user-visible state at the
decision minute; when that state is not observable, it conservatively predicts
normal and reports the missing-state count separately.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
FOLDS = range(5)
LTFP_RUN = (
    "re2_{dataset}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1"
)
TRANSITIONS = (
    "onset",
    "unknown_to_failure",
    "continuing",
    "recovery",
    "stable_normal",
    "unknown_to_normal",
)
TRANSITION_NAMES = {
    "onset": "正常→故障",
    "unknown_to_failure": "未知→故障",
    "continuing": "故障→故障",
    "recovery": "故障→正常",
    "stable_normal": "正常→正常",
    "unknown_to_normal": "未知→正常",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    ranked_scores = scores[order]
    tie_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[tie_ends]
    precision = cumulative / (tie_ends + 1)
    recall_increments = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recall_increments))


def _classification(labels: list[int], scores: list[float]) -> dict[str, Any]:
    if not labels:
        return {
            "samples": 0,
            "positives": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    predicted = score >= 0.5
    tp = int(np.logical_and(predicted, y == 1).sum())
    fp = int(np.logical_and(predicted, y == 0).sum())
    fn = int(np.logical_and(~predicted, y == 1).sum())
    tn = int(np.logical_and(~predicted, y == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(labels),
        "positives": int(y.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": _average_precision(y, score),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _transition(future: dict[str, str], current: dict[str, str] | None) -> str:
    future_failure = int(future["failure_next_minute"])
    current_observed = bool(current and int(current["entry_observed"]) == 1)
    current_failure = int(current["failure_next_minute"]) if current_observed else 0
    if future_failure:
        if not current_observed:
            return "unknown_to_failure"
        if current_failure and current and current["event_id"] == future["event_id"]:
            return "continuing"
        return "onset"
    if not current_observed:
        return "unknown_to_normal"
    if current_failure:
        return "recovery"
    return "stable_normal"


def _load_rows(dataset: str) -> list[dict[str, Any]]:
    panel = ROOT / "artifacts" / f"re2_{dataset}_panel_fair_v1"
    schema = json.loads((panel / "feature_schema.json").read_text())
    labels = _read_csv(Path(schema["source_labels"]))
    by_input = {row["input_minute_utc"]: row for row in labels}
    by_target = {row["target_minute_utc"]: row for row in labels}

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in FOLDS:
        prediction_path = (
            ROOT
            / "artifacts"
            / LTFP_RUN.format(dataset=dataset, fold=fold)
            / "predictions.jsonl"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                prediction = json.loads(line)
                timestamp = prediction["input_minute_utc"]
                if timestamp in seen:
                    raise ValueError(f"duplicate out-of-fold timestamp: {dataset} {timestamp}")
                seen.add(timestamp)
                future = by_input.get(timestamp)
                if future is None:
                    raise ValueError(f"missing source label: {dataset} {timestamp}")
                current = by_target.get(timestamp)
                if current is not None and current["case_id"] != future["case_id"]:
                    current = None
                current_observed = bool(current and int(current["entry_observed"]) == 1)
                current_failure = (
                    int(current["failure_next_minute"]) if current_observed else 0
                )
                label = int(future["failure_next_minute"])
                if label != int(prediction["failure_target"]):
                    raise ValueError(f"prediction/label mismatch: {dataset} {timestamp}")
                rows.append({
                    "fold": fold,
                    "case_id": future["case_id"],
                    "input_minute_utc": timestamp,
                    "target_minute_utc": future["target_minute_utc"],
                    "label": label,
                    "transition": _transition(future, current),
                    "current_observed": current_observed,
                    "persistence_score": float(current_failure),
                    "ltfp_score": float(prediction["failure_probability"]),
                    "previous_event_id": current["event_id"] if current else "",
                })
    return rows


def _subset_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    return _classification(
        [int(row["label"]) for row in rows],
        [float(row[score_key]) for row in rows],
    )


def _analyze(dataset: str) -> dict[str, Any]:
    rows = _load_rows(dataset)
    counts = Counter(row["transition"] for row in rows)
    recoveries = [row for row in rows if row["transition"] == "recovery"]
    observed = [row for row in rows if row["current_observed"]]
    return {
        "dataset": f"RE2-{dataset.upper()}",
        "protocol": "pooled disjoint test windows from five deployment folds",
        "windows": len(rows),
        "transition_counts": {name: counts[name] for name in TRANSITIONS},
        "current_state_observed": len(observed),
        "current_state_missing": len(rows) - len(observed),
        "recovery_case_count": len({row["case_id"] for row in recoveries}),
        "persistence": {
            "definition": "predict target state with decision-minute state; missing state -> normal",
            "overall": _subset_metrics(rows, "persistence_score"),
            "observed_current_subset": _subset_metrics(observed, "persistence_score"),
        },
        "ltfp": {
            "definition": "existing seed-42, 30-epoch out-of-fold predictions",
            "overall": _subset_metrics(rows, "ltfp_score"),
        },
        "recovery_examples": [
            {
                "case_id": row["case_id"],
                "decision_minute_utc": row["input_minute_utc"],
                "normal_target_minute_utc": row["target_minute_utc"],
                "preceding_failure_event": row["previous_event_id"],
            }
            for row in recoveries[:10]
        ],
    }


def _metric_row(name: str, values: dict[str, Any]) -> str:
    def display(key: str) -> str:
        value = values[key]
        return "—" if value is None else f"{value:.3f}"

    return (
        f"| {name} | {values['samples']} | {values['positives']} | "
        f"{display('precision')} | {display('recall')} | {display('f1')} | "
        f"{display('average_precision')} |"
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RE2状态转移与Persistence基线审计",
        "",
        "本报告复用deployment五折互不重叠的LTFP测试窗口，不重新训练或调参。",
        "Persistence直接用决策分钟的业务故障状态预测下一分钟；当前状态不可观测时按正常处理，并单独报告缺失数量。",
        "",
    ]
    for dataset in report["datasets"]:
        lines.extend([
            f"## {dataset['dataset']}",
            "",
            "### 状态转移数量",
            "",
            "| 转移 | 窗口数 |",
            "|---|---:|",
        ])
        for name in TRANSITIONS:
            lines.append(
                f"| {TRANSITION_NAMES[name]} | {dataset['transition_counts'][name]} |"
            )
        lines.extend([
            "",
            f"当前状态可观测：{dataset['current_state_observed']}；"
            f"不可观测：{dataset['current_state_missing']}。",
            f"恢复窗口涉及{dataset['recovery_case_count']}个不同case；同一case可能因业务指标"
            "跨阈值波动而出现多个恢复窗口。",
            "",
            "### 下一分钟预测",
            "",
            "| 方法 | 样本 | 正例 | Precision | Recall | F1 | AP |",
            "|---|---:|---:|---:|---:|---:|---:|",
            _metric_row("Persistence", dataset["persistence"]["overall"]),
            _metric_row("LTFP", dataset["ltfp"]["overall"]),
            "",
            "### 恢复窗口示例",
            "",
            "| case | 决策分钟（仍为故障） | 下一分钟（恢复正常） |",
            "|---|---|---|",
        ])
        for example in dataset["recovery_examples"][:5]:
            lines.append(
                f"| {example['case_id']} | {example['decision_minute_utc']} | "
                f"{example['normal_target_minute_utc']} |"
            )
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- “恢复窗口”表示相邻两个可评价分钟的业务标签由故障变为正常，不等同于人工确认系统已经完成修复。",
        "- Persistence使用业务状态标签，是诊断任务难度的参照物，不是可部署模型；真正模型只能使用决策时刻可获得的遥测。",
        "- 如果Persistence接近或超过学习模型，整体F1会受到状态持续性的显著影响，后续方法必须重点报告首分钟召回和恢复误报。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    report = {
        "protocol": {
            "task": "past five complete minutes predict next-minute user-visible failure state",
            "split": "deployment five-fold out-of-fold union",
            "reference_model": "LTFP, seed 42, 30 epochs",
        },
        "datasets": [_analyze(dataset) for dataset in DATASETS],
    }
    artifact = ROOT / "artifacts" / "re2_transition_persistence_audit_v1.json"
    document = ROOT / "docs" / "2026-11-12-transition-persistence-audit.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
