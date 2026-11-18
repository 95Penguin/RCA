#!/usr/bin/env python3
"""Compare LTFP and its transition-residual variant with paired case bootstrap."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
BASELINE = (
    "re2_{dataset}_ltfp_deployment_fold{fold}_first_visible_1m_"
    "fair_30ep_seed42_baseline_v1"
)
TRANSITION = (
    "re2_{dataset}_ltfp_transition_deployment_fold{fold}_first_visible_1m_"
    "fair_30ep_seed42_v1"
)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    ranked_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recall_increments = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recall_increments))


def _case_map(dataset: str) -> dict[str, str]:
    path = ROOT / "artifacts" / f"re2_{dataset}_panel_fair_v1" / "case_minutes.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["timestamp_utc"]: row["case_id"] for row in csv.DictReader(handle)}


def _root_rank(row: dict[str, Any]) -> int | None:
    truth = set(row["true_services"])
    ranks = [
        rank
        for rank, item in enumerate(row["ranking"], start=1)
        if item["service"] in truth
    ]
    return min(ranks) if ranks else None


def _load_predictions(dataset: str, pattern: str) -> list[dict[str, Any]]:
    case_by_time = _case_map(dataset)
    rows = []
    for fold in range(5):
        path = ROOT / "artifacts" / pattern.format(dataset=dataset, fold=fold) / "predictions.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                prediction = json.loads(line)
                timestamp = prediction["input_minute_utc"]
                rows.append({
                    "fold": fold,
                    "timestamp": timestamp,
                    "case_id": case_by_time[timestamp],
                    "label": int(prediction["failure_target"]),
                    "score": float(prediction["failure_probability"]),
                    "root_rank": _root_rank(prediction),
                })
    return rows


def _load_fold_metrics(dataset: str, pattern: str) -> list[dict[str, float]]:
    rows = []
    for fold in range(5):
        path = ROOT / "artifacts" / pattern.format(dataset=dataset, fold=fold) / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        failure = metrics["failure_prediction"]
        rca = metrics["root_cause_event_proxy"]
        rows.append({
            "ap": failure["average_precision"],
            "f1": failure["f1_at_0_5"],
            "hit1": rca["hit_at_1"],
            "mrr": rca["mrr"],
        })
    return rows


def _ci(values: list[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, (2.5, 97.5))]


def _paired_analysis(
    dataset: str,
    bootstraps: int = 5000,
    *,
    baseline_pattern: str = BASELINE,
    candidate_pattern: str = TRANSITION,
) -> dict[str, Any]:
    baseline = _load_predictions(dataset, baseline_pattern)
    transition = _load_predictions(dataset, candidate_pattern)
    baseline_by_key = {(row["fold"], row["timestamp"]): row for row in baseline}
    transition_by_key = {(row["fold"], row["timestamp"]): row for row in transition}
    if baseline_by_key.keys() != transition_by_key.keys():
        raise ValueError(f"prediction windows differ for {dataset}")

    paired = []
    for key in sorted(baseline_by_key):
        left = baseline_by_key[key]
        right = transition_by_key[key]
        if left["label"] != right["label"] or left["case_id"] != right["case_id"]:
            raise ValueError(f"paired labels differ for {dataset} {key}")
        paired.append({
            **left,
            "baseline_score": left["score"],
            "transition_score": right["score"],
            "baseline_root_rank": left["root_rank"],
            "transition_root_rank": right["root_rank"],
        })

    def summarize(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        labels = np.asarray([row["label"] for row in rows], dtype=int)
        baseline_scores = np.asarray([row["baseline_score"] for row in rows], dtype=float)
        transition_scores = np.asarray([row["transition_score"] for row in rows], dtype=float)
        positives = [row for row in rows if row["label"]]
        baseline_ranks = np.asarray([row["baseline_root_rank"] for row in positives], dtype=float)
        transition_ranks = np.asarray([row["transition_root_rank"] for row in positives], dtype=float)
        return (
            _average_precision(labels, baseline_scores),
            _average_precision(labels, transition_scores),
            float(np.mean(transition_ranks == 1) - np.mean(baseline_ranks == 1)),
            float(np.mean(1.0 / transition_ranks) - np.mean(1.0 / baseline_ranks)),
        )

    baseline_ap, transition_ap, hit1_delta, mrr_delta = summarize(paired)
    by_fold_case: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in paired:
        by_fold_case[row["fold"]][row["case_id"]].append(row)

    rng = np.random.default_rng(20261114)
    ap_deltas = []
    hit1_deltas = []
    mrr_deltas = []
    for _ in range(bootstraps):
        sampled = []
        for fold in range(5):
            cases = sorted(by_fold_case[fold])
            for case_id in rng.choice(cases, size=len(cases), replace=True):
                sampled.extend(by_fold_case[fold][str(case_id)])
        sampled_baseline, sampled_transition, sampled_hit1, sampled_mrr = summarize(sampled)
        ap_deltas.append(sampled_transition - sampled_baseline)
        hit1_deltas.append(sampled_hit1)
        mrr_deltas.append(sampled_mrr)

    baseline_folds = _load_fold_metrics(dataset, baseline_pattern)
    transition_folds = _load_fold_metrics(dataset, candidate_pattern)
    fold_rows = []
    for fold, (left, right) in enumerate(zip(baseline_folds, transition_folds)):
        fold_rows.append({
            "fold": fold,
            "baseline": left,
            "transition": right,
            "delta": {key: right[key] - left[key] for key in left},
        })

    def fold_summary(key: str) -> dict[str, float]:
        left = np.asarray([row[key] for row in baseline_folds])
        right = np.asarray([row[key] for row in transition_folds])
        return {
            "baseline_mean": float(left.mean()),
            "transition_mean": float(right.mean()),
            "paired_delta": float((right - left).mean()),
        }

    return {
        "windows": len(paired),
        "positive_cases": int(sum(row["label"] for row in paired)),
        "fold_summary": {key: fold_summary(key) for key in ("ap", "f1", "hit1", "mrr")},
        "pooled": {
            "baseline_ap": baseline_ap,
            "transition_ap": transition_ap,
            "ap_delta": transition_ap - baseline_ap,
            "hit1_delta": hit1_delta,
            "mrr_delta": mrr_delta,
        },
        "paired_case_bootstrap_95pct_ci": {
            "ap_delta": _ci(ap_deltas),
            "hit1_delta": _ci(hit1_deltas),
            "mrr_delta": _ci(mrr_deltas),
        },
        "folds": fold_rows,
    }


def _display(value: float) -> str:
    return f"{value:+.3f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LTFP Transition Residual最小实验",
        "",
        f"结论：**{report['decision']['status']}**。",
        "",
        "所有比较使用相同的first-visible 1分钟标签、deployment五折、30 epoch、seed 42和RCA loss。",
        "置信区间对同一物理case的两种模型预测进行配对重采样。",
        "",
        "## 五折均值",
        "",
        "| 数据集 | LTFP AP | Transition AP | ΔAP | LTFP Hit@1 | Transition Hit@1 | ΔHit@1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["datasets"].items():
        ap = values["fold_summary"]["ap"]
        hit1 = values["fold_summary"]["hit1"]
        lines.append(
            f"| RE2-{name.upper()} | {ap['baseline_mean']:.3f} | "
            f"{ap['transition_mean']:.3f} | {_display(ap['paired_delta'])} | "
            f"{hit1['baseline_mean']:.3f} | {hit1['transition_mean']:.3f} | "
            f"{_display(hit1['paired_delta'])} |"
        )
    lines.extend([
        "",
        "## 配对case-bootstrap",
        "",
        "| 数据集 | Pooled ΔAP | ΔAP 95% CI | Pooled ΔHit@1 | ΔHit@1 95% CI |",
        "|---|---:|---|---:|---|",
    ])
    for name, values in report["datasets"].items():
        pooled = values["pooled"]
        intervals = values["paired_case_bootstrap_95pct_ci"]
        lines.append(
            f"| RE2-{name.upper()} | {_display(pooled['ap_delta'])} | "
            f"[{intervals['ap_delta'][0]:+.3f}, {intervals['ap_delta'][1]:+.3f}] | "
            f"{_display(pooled['hit1_delta'])} | "
            f"[{intervals['hit1_delta'][0]:+.3f}, {intervals['hit1_delta'][1]:+.3f}] |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        "预设继续门槛是：至少一个数据集平均AP提升0.03，另一个数据集不明显下降，且RCA不退化。",
        "当前最小结构没有达到门槛：两个数据集的平均AP都下降，提升只出现在部分deployment折。",
        "因此不继续在这个结构上增加更多分支；该负结果不否定persistence-bias和first-visible评测故事，",
        "但说明简单的logit分解本身不能作为方法贡献。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    datasets = {name: _paired_analysis(name) for name in DATASETS}
    gate = any(
        values["fold_summary"]["ap"]["paired_delta"] >= 0.03
        for values in datasets.values()
    ) and all(
        values["fold_summary"]["ap"]["paired_delta"] >= -0.01
        for values in datasets.values()
    ) and all(
        values["fold_summary"]["hit1"]["paired_delta"] >= -0.01
        for values in datasets.values()
    )
    report = {
        "protocol": {
            "task": "first-visible failure within (t,t+1m]",
            "split": "deployment five-fold",
            "epochs": 30,
            "seed": 42,
            "bootstrap": "5000 paired physical-case resamples stratified by fold",
        },
        "datasets": datasets,
        "decision": {
            "status": "GO" if gate else "NO-GO for this minimal architecture",
            "passed_predeclared_gate": gate,
        },
    }
    artifact = ROOT / "artifacts" / "ltfp_transition_experiment_2026-11-14.json"
    document = ROOT / "docs" / "2026-11-14-ltfp-transition-experiment.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
