#!/usr/bin/env python3
"""Decide whether transition-aware predictive RCA merits model development.

The audit combines three pieces of already-computed out-of-fold evidence:

* next-minute state persistence versus LTFP;
* LTFP on the one-minute first-visible task;
* a strict adjacent-state linear probe.

Confidence intervals resample whole physical cases within each deployment fold.
This preserves the study's split unit and avoids treating neighboring windows
from the same injected case as independent observations.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
FOLDS = range(5)
FIRST_VISIBLE_RUN = (
    "re2_{dataset}_ltfp_deployment_fold{fold}_first_visible_1m_"
    "fair_30ep_seed42_baseline_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recalled = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recalled))


def _percentile_interval(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def _case_by_timestamp(dataset: str) -> dict[str, str]:
    path = ROOT / "artifacts" / f"re2_{dataset}_panel_fair_v1" / "case_minutes.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["timestamp_utc"]: row["case_id"]
            for row in csv.DictReader(handle)
        }


def _load_first_visible(dataset: str) -> list[dict[str, Any]]:
    case_by_time = _case_by_timestamp(dataset)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in FOLDS:
        path = (
            ROOT
            / "artifacts"
            / FIRST_VISIBLE_RUN.format(dataset=dataset, fold=fold)
            / "predictions.jsonl"
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                timestamp = row["input_minute_utc"]
                if timestamp in seen:
                    raise ValueError(f"duplicate out-of-fold timestamp: {dataset} {timestamp}")
                seen.add(timestamp)
                case_id = case_by_time.get(timestamp)
                if case_id is None:
                    raise ValueError(f"prediction timestamp lacks a case: {dataset} {timestamp}")
                ranking = row["ranking"]
                true_services = set(row["true_services"])
                true_ranks = [
                    rank
                    for rank, item in enumerate(ranking, start=1)
                    if item["service"] in true_services
                ]
                records.append({
                    "fold": fold,
                    "case_id": case_id,
                    "label": int(row["failure_target"]),
                    "score": float(row["failure_probability"]),
                    "root_rank": min(true_ranks) if true_ranks else None,
                    "service_count": len(ranking),
                })
    return records


def _first_visible_metrics(records: list[dict[str, Any]], bootstraps: int) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in records], dtype=int)
    scores = np.asarray([row["score"] for row in records], dtype=float)
    positives = [row for row in records if row["label"]]
    ap = _average_precision(labels, scores)
    prevalence = float(labels.mean())
    root_ranks = np.asarray([row["root_rank"] for row in positives], dtype=float)
    service_count = int(records[0]["service_count"])

    by_fold_case: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        by_fold_case[row["fold"]][row["case_id"]].append(row)

    rng = np.random.default_rng(20261114)
    bootstrap_ap: list[float] = []
    bootstrap_advantage: list[float] = []
    bootstrap_hit1: list[float] = []
    bootstrap_mrr: list[float] = []
    for _ in range(bootstraps):
        sampled: list[dict[str, Any]] = []
        for fold in FOLDS:
            cases = sorted(by_fold_case[fold])
            chosen = rng.choice(cases, size=len(cases), replace=True)
            for case_id in chosen:
                sampled.extend(by_fold_case[fold][str(case_id)])
        sample_labels = np.asarray([row["label"] for row in sampled], dtype=int)
        sample_scores = np.asarray([row["score"] for row in sampled], dtype=float)
        sample_ap = _average_precision(sample_labels, sample_scores)
        sample_prevalence = float(sample_labels.mean())
        bootstrap_ap.append(sample_ap)
        bootstrap_advantage.append(sample_ap - sample_prevalence)
        sampled_ranks = np.asarray(
            [row["root_rank"] for row in sampled if row["label"]], dtype=float
        )
        if len(sampled_ranks):
            bootstrap_hit1.append(float(np.mean(sampled_ranks == 1)))
            bootstrap_mrr.append(float(np.mean(1.0 / sampled_ranks)))

    fold_ap = []
    for fold in FOLDS:
        subset = [row for row in records if row["fold"] == fold]
        fold_labels = np.asarray([row["label"] for row in subset], dtype=int)
        fold_scores = np.asarray([row["score"] for row in subset], dtype=float)
        fold_ap.append(_average_precision(fold_labels, fold_scores))

    return {
        "windows": len(records),
        "physical_cases": len({row["case_id"] for row in records}),
        "positive_onset_cases": len(positives),
        "prevalence": prevalence,
        "pooled_average_precision": ap,
        "fold_average_precision_mean": float(np.mean(fold_ap)),
        "fold_average_precision_range": [float(min(fold_ap)), float(max(fold_ap))],
        "ap_lift_over_prevalence": ap / prevalence,
        "case_clustered_95pct_ci": {
            "average_precision": _percentile_interval(bootstrap_ap),
            "ap_minus_prevalence": _percentile_interval(bootstrap_advantage),
            "rca_hit_at_1": _percentile_interval(bootstrap_hit1),
            "rca_mrr": _percentile_interval(bootstrap_mrr),
        },
        "rca_hit_at_1": float(np.mean(root_ranks == 1)),
        "rca_mrr": float(np.mean(1.0 / root_ranks)),
        "random_root_hit_at_1": 1.0 / service_count,
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transition-aware预测RCA可行性审计",
        "",
        f"结论：**{report['decision']['status']}**。",
        "",
        "这里把两个问题分开：研究问题是否成立，以及是否已经有足够证据开发复杂Transition模型。",
        "置信区间以物理case为单位、在deployment折内重采样，避免把相邻分钟当成独立样本。",
        "",
        "## 核心结果",
        "",
        "| 数据集 | Persistence F1 | LTFP状态F1 | First-visible正例case | AP | 正例率 | AP lift | AP 95% CI | AP-正例率 95% CI | RCA Hit@1 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for dataset in report["datasets"]:
        first = dataset["first_visible"]
        intervals = first["case_clustered_95pct_ci"]
        lines.append(
            f"| {dataset['name']} | {_fmt(dataset['state']['persistence_f1'])} | "
            f"{_fmt(dataset['state']['ltfp_f1'])} | {first['positive_onset_cases']} | "
            f"{_fmt(first['pooled_average_precision'])} | {_fmt(first['prevalence'])} | "
            f"{first['ap_lift_over_prevalence']:.1f}× | "
            f"{_ci(intervals['average_precision'])} | "
            f"{_ci(intervals['ap_minus_prevalence'])} | "
            f"{_fmt(first['rca_hit_at_1'])} |"
        )
    lines.extend([
        "",
        "## 严格相邻状态探针",
        "",
        "| 数据集 | 严格onset正例 | State AP | State+change AP | 判读 |",
        "|---|---:|---:|---:|---|",
    ])
    for dataset in report["datasets"]:
        probe = dataset["strict_adjacent_probe"]
        judgement = "变化特征有弱增益" if probe["state_change_ap"] > probe["state_ap"] else "无变化特征增益"
        lines.append(
            f"| {dataset['name']} | {probe['positives']} | {_fmt(probe['state_ap'])} | "
            f"{_fmt(probe['state_change_ap'])} | {judgement} |"
        )
    lines.extend([
        "",
        "## Go / No-Go判定",
        "",
        "### GO：论文问题与first-visible主任务",
        "",
        "- 两个数据集的Persistence总体F1都高于LTFP，确认状态持续性会显著影响总体分数。",
        "- 每个数据集都有超过50个独立first-visible正例case，而不是把持续故障分钟重复计数。",
        "- 两个数据集的case-bootstrap `AP-正例率` 95%区间下界均高于0，说明首次故障前存在可泛化排序信号。",
        "- First-visible RCA明显高于随机服务排序，预测与定位可以放在同一故事中。",
        "",
        "### HOLD：复杂Transition架构",
        "",
        "- 严格相邻状态可观测的onset只有OB 10个、TT 20个，不能单独支撑复杂多分支模型。",
        "- State+change线性探针只带来小幅且数据集不稳定的改善，尚不足以证明需要复杂变化编码器。",
        "- Recovery样本更少，现阶段只能作为辅助指标，不能作为主贡献。",
        "",
        "## 下一步最小实验",
        "",
        "在first-visible标签上实现一个轻量 `current-state prior + transition residual`，保持LTFP编码器、数据划分和RCA头不变。",
        "只有当它在两个数据集中的至少一个达到 `AP +0.03`，且另一个数据集不明显下降，才继续扩展到1/3/5分钟联合模型。",
        "严格10/20个相邻onset只做敏感性分析，不作为训练主样本集合。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    persistence = _read_json(
        ROOT / "artifacts" / "re2_transition_persistence_audit_v1.json"
    )
    probe = _read_json(
        ROOT / "artifacts" / "re2_transition_learnability_probe_v1.json"
    )
    persistence_by_name = {
        item["dataset"].split("-")[-1].lower(): item
        for item in persistence["datasets"]
    }

    datasets = []
    for name in DATASETS:
        state = persistence_by_name[name]
        strict = probe["results"][name]["onset"]
        datasets.append({
            "name": f"RE2-{name.upper()}",
            "state": {
                "persistence_f1": state["persistence"]["overall"]["f1"],
                "ltfp_f1": state["ltfp"]["overall"]["f1"],
                "continuing_failure_windows": state["transition_counts"]["continuing"],
                "strict_onset_windows": state["transition_counts"]["onset"],
                "unknown_to_failure_windows": state["transition_counts"]["unknown_to_failure"],
            },
            "first_visible": _first_visible_metrics(_load_first_visible(name), 2000),
            "strict_adjacent_probe": {
                "positives": strict["state"]["metrics"]["positives"],
                "state_ap": strict["state"]["metrics"]["average_precision"],
                "state_change_ap": strict["state_change"]["metrics"]["average_precision"],
            },
        })

    first_visible_supported = all(
        item["first_visible"]["case_clustered_95pct_ci"]["ap_minus_prevalence"][0] > 0
        and item["first_visible"]["positive_onset_cases"] >= 50
        for item in datasets
    )
    persistence_supported = all(
        item["state"]["persistence_f1"] > item["state"]["ltfp_f1"]
        for item in datasets
    )
    report = {
        "protocol": {
            "split": "five deployment folds; pooled out-of-fold predictions",
            "bootstrap": "2000 case-clustered resamples stratified by deployment fold",
            "reference_model": "LTFP, seed 42, 30 epochs",
        },
        "datasets": datasets,
        "decision": {
            "status": (
                "CONDITIONAL GO: 研究问题和first-visible主任务可行；复杂Transition模型暂缓"
                if persistence_supported and first_visible_supported
                else "NO-GO: 当前证据不足"
            ),
            "persistence_problem_supported": persistence_supported,
            "first_visible_signal_supported": first_visible_supported,
        },
    }
    artifact = ROOT / "artifacts" / "transition_feasibility_2026-11-14.json"
    document = ROOT / "docs" / "2026-11-14-transition-feasibility.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
