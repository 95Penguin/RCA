#!/usr/bin/env python3
"""Compare dense next-minute state labels with LTFP-style first-visible labels."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDS = range(5)
TASKS = {
    "state-1m": {
        "label": "下一分钟状态（含持续）",
        "directory": "re2_{dataset}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1",
    },
    "first-visible-1m": {
        "label": "首次可感知（1分钟窗）",
        "directory": "re2_{dataset}_ltfp_deployment_fold{fold}_first_visible_1m_fair_30ep_seed42_baseline_v1",
    },
    "first-visible-10m": {
        "label": "首次可感知（10分钟窗）",
        "directory": "re2_{dataset}_ltfp_deployment_fold{fold}_first_visible_10m_fair_30ep_seed42_baseline_v1",
    },
}


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _task_metrics(dataset: str, task: str) -> dict[str, object]:
    rows = []
    for fold in FOLDS:
        directory = ROOT / "artifacts" / TASKS[task]["directory"].format(
            dataset=dataset, fold=fold
        )
        report = json.loads((directory / "metrics.json").read_text())
        prediction = report["failure_prediction"]
        rca = report["root_cause_event_proxy"]
        positives = int(report["split_positive_samples"]["test"])
        test_size = int(report["split_sizes"]["test"])
        prevalence = positives / test_size
        ap = float(prediction["average_precision"])
        rows.append({
            "fold": fold,
            "test_size": test_size,
            "positives": positives,
            "prevalence": prevalence,
            "precision": float(prediction["precision_at_0_5"]),
            "recall": float(prediction["recall_at_0_5"]),
            "f1": float(prediction["f1_at_0_5"]),
            "ap": ap,
            "ap_lift": ap / prevalence,
            "hit_at_1": float(rca["hit_at_1"]),
            "hit_at_3": float(rca["hit_at_3"]),
            "hit_at_5": float(rca["hit_at_5"]),
            "mrr": float(rca["mrr"]),
            "events": int(rca["positive_events"]),
        })
    return {
        "folds": rows,
        "test_windows": sum(row["test_size"] for row in rows),
        "positive_windows": sum(row["positives"] for row in rows),
        "positive_events": sum(row["events"] for row in rows),
        **{
            key: _mean_sd([float(row[key]) for row in rows])
            for key in (
                "prevalence", "precision", "recall", "f1", "ap", "ap_lift",
                "hit_at_1", "hit_at_3", "hit_at_5", "mrr",
            )
        },
    }


def _lead_audit(dataset: str) -> dict[str, object]:
    panel = ROOT / f"artifacts/re2_{dataset}_panel_fair_v1"
    schema = json.loads((panel / "feature_schema.json").read_text())
    source = Path(schema["source_labels"])
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_case[row["case_id"]].append(row)
    leads = []
    for rows in by_case.values():
        positive = [row for row in rows if row["failure_next_minute"] == "1"]
        if not positive:
            continue
        onset = min(
            datetime.fromisoformat(row["target_minute_utc"]).timestamp()
            for row in positive
        )
        post_decisions = [
            datetime.fromisoformat(row["input_minute_utc"]).timestamp()
            for row in rows
            if row["phase"] == "post"
            and datetime.fromisoformat(row["input_minute_utc"]).timestamp() < onset
        ]
        leads.append((onset - min(post_decisions)) / 60 if post_decisions else 0.0)
    return {
        "failure_cases": len(leads),
        "median_available_post_injection_lead_minutes": statistics.median(leads),
        "mean_available_post_injection_lead_minutes": statistics.fmean(leads),
        "max_available_post_injection_lead_minutes": max(leads),
        "cases_with_at_least_minutes": {
            str(horizon): sum(lead >= horizon for lead in leads)
            for horizon in (1, 2, 3, 5, 10)
        },
    }


def _fmt(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f}±{metric['sd']:.3f}"


def main() -> int:
    results = {
        f"RE2-{dataset.upper()}": {
            "lead_audit": _lead_audit(dataset),
            "tasks": {task: _task_metrics(dataset, task) for task in TASKS},
        }
        for dataset in ("ob", "tt")
    }
    payload = {
        "protocol": {
            "model": "LTFP architecture baseline",
            "history_minutes": 5,
            "epochs": 30,
            "seed": 42,
            "split": "deployment-group five-fold holdout",
            "threshold": 0.5,
            "first_visible_rule": (
                "one reference time per physical case; post-failure inputs excluded; "
                "pre-injection windows are healthy controls"
            ),
        },
        "results": results,
    }
    json_path = ROOT / "artifacts/re2_ltfp_task_comparison_seed42_v1.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LTFP任务目标对比：状态预测与首次用户可感知预测",
        "",
        "固定LTFP架构、过去5分钟输入、deployment五折、30轮和seed 42，仅改变标签目标。",
        "首次任务为每个物理case确定一个首次业务退化时刻；严格排除该时刻及之后的输入，注入前窗口作为正常对照。",
        "",
    ]
    for dataset_name, data in results.items():
        lines.extend([
            f"## {dataset_name}",
            "",
            "| 任务 | 正例窗口/总窗口 | 正例率 | Precision | Recall | F1 | AP | AP/正例率 | Hit@1 | Hit@3 | Hit@5 | MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for task, spec in TASKS.items():
            row = data["tasks"][task]
            lines.append(
                f"| {spec['label']} | {row['positive_windows']}/{row['test_windows']} | "
                + " | ".join(_fmt(row[key]) for key in (
                    "prevalence", "precision", "recall", "f1", "ap", "ap_lift",
                    "hit_at_1", "hit_at_3", "hit_at_5", "mrr",
                ))
                + " |"
            )
        audit = data["lead_audit"]
        support = audit["cases_with_at_least_minutes"]
        lines.extend([
            "",
            f"可用注入后提前量：{audit['failure_cases']}个首次故障case，中位数"
            f"{audit['median_available_post_injection_lead_minutes']:.1f}分钟，均值"
            f"{audit['mean_available_post_injection_lead_minutes']:.2f}分钟；至少2/5/10分钟的case分别为"
            f"{support['2']}/{support['5']}/{support['10']}。",
            "",
        ])
    lines.extend([
        "## 结论", "",
        "- 下一分钟状态任务结果最高且稳定，因为持续故障提供了更多正例，但它回答的是短期状态预测，不是纯粹的首次预警。",
        "- 首次1分钟任务有学习信号：AP显著高于约5%的随机正例率；但每个case基本只有一个正例窗口，F1和跨折稳定性明显下降。",
        "- 首次10分钟在TT上改善明显，在OB上仍高度不稳定。两套数据的中位实际注入后提前量都只有1分钟，因此不能把10分钟窗口描述成普遍提前10分钟。",
        "- 如果论文强调科学问题，建议以首次用户可感知故障为主任务、下一分钟状态预测为辅助训练目标；若只保留当前数据和单任务，状态预测更稳但主动预警意义较弱。",
    ])
    markdown_path = ROOT / "docs/2026-11-12-ltfp-task-comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text())
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
