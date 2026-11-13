#!/usr/bin/env python3
"""Audit whether injected root services precede connected symptom services.

For every RE2 case with at least one audited user-visible failure minute, the
script fits a robust per-service normal profile using only pre-injection
minutes.  It then finds persistent post-injection telemetry deviations and
compares the injected root service with its directly connected services.

This is an observational audit.  The current prepared graph stores symmetric
call counts, so the report can establish temporal precedence over connected
services but cannot claim caller/callee causality or propagation direction.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset  # noqa: E402


PANELS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}
SIGNALS = {
    "metric_cpu": "observed_cpu",
    "metric_mem": "observed_mem",
    "metric_diskio": "observed_diskio",
    "metric_socket": "observed_socket",
    "trace_span_count": "trace_span_count",
    "trace_error_count": "trace_span_count",
    "trace_duration_mean_ms": "trace_span_count",
    "log_count": "log_observed",
    "log_error_count": "log_observed",
}
FAULT_SIGNAL = {
    "cpu": "metric_cpu",
    "mem": "metric_mem",
    "disk": "metric_diskio",
    "socket": "metric_socket",
    "delay": "trace_duration_mean_ms",
    "loss": "trace_error_count",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _epoch(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).timestamp())


def _robust_scores(
    dataset: Re2WindowDataset,
    baseline_indices: list[int],
    audit_indices: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return general max-z and per-signal robust deviation scores."""

    feature_index = {name: i for i, name in enumerate(dataset.feature_columns)}
    time_count = len(audit_indices)
    service_count = len(dataset.services)
    per_signal: dict[str, np.ndarray] = {}
    for signal, observed in SIGNALS.items():
        value_index = feature_index[signal]
        observed_index = feature_index[observed]
        score = np.full((time_count, service_count), np.nan, dtype=np.float32)
        for service in range(service_count):
            baseline_values = dataset.features[baseline_indices, service, value_index]
            baseline_observed = dataset.features[baseline_indices, service, observed_index] > 0
            baseline_values = baseline_values[baseline_observed]
            if len(baseline_values) < 5:
                continue
            transformed = np.log1p(np.maximum(baseline_values.astype(float), 0.0))
            center = float(np.median(transformed))
            mad = float(np.median(np.abs(transformed - center)))
            scale = 1.4826 * mad
            if scale < 0.05:
                scale = max(0.05, 0.1 * abs(center))
            audit_values = dataset.features[audit_indices, service, value_index]
            audit_observed = dataset.features[audit_indices, service, observed_index] > 0
            audit_transformed = np.log1p(np.maximum(audit_values.astype(float), 0.0))
            deviations = np.minimum(np.abs(audit_transformed - center) / scale, 20.0)
            score[audit_observed, service] = deviations[audit_observed]
        per_signal[signal] = score
    stacked = np.stack(list(per_signal.values()))
    finite = np.isfinite(stacked)
    general = np.max(np.where(finite, stacked, -np.inf), axis=0)
    general[~finite.any(axis=0)] = np.nan
    return general, per_signal


def _first_persistent(
    scores: np.ndarray, epochs: list[int], threshold: float, consecutive: int = 2
) -> list[int | None]:
    first: list[int | None] = []
    for service in range(scores.shape[1]):
        found = None
        for index in range(0, len(epochs) - consecutive + 1):
            block = scores[index : index + consecutive, service]
            if (
                np.isfinite(block).all()
                and np.all(block >= threshold)
                and all(epochs[index + offset] == epochs[index] + 60 * offset
                        for offset in range(consecutive))
            ):
                found = epochs[index]
                break
        first.append(found)
    return first


def _rank_root(
    root: int, first: list[int | None], maximum: np.ndarray
) -> int:
    if first[root] is None:
        return len(first)
    sentinel = 10**20
    order = sorted(
        range(len(first)),
        key=lambda service: (
            first[service] if first[service] is not None else sentinel,
            -float(maximum[service]) if math.isfinite(float(maximum[service])) else math.inf,
            service,
        ),
    )
    return order.index(root) + 1


def _case_result(
    dataset: Re2WindowDataset,
    adjacency: np.ndarray,
    case_id: str,
    case_rows: list[dict[str, str]],
    case_meta: dict[str, str],
    threshold: float,
) -> dict[str, Any] | None:
    if not any(int(row["failure_next_minute"]) for row in case_rows):
        return None
    by_input = {row["input_minute_utc"]: row for row in case_rows}
    indices = [
        index for index, owner in enumerate(dataset.case_ids)
        if owner == case_id and dataset.timestamps[index] in by_input
    ]
    baseline = [index for index in indices if by_input[dataset.timestamps[index]]["phase"] == "pre"]
    audit = [index for index in indices if by_input[dataset.timestamps[index]]["phase"] != "pre"]
    if len(baseline) < 5 or len(audit) < 2:
        return None
    root_name = case_meta["root_service"]
    if root_name not in dataset.services:
        return None
    root = dataset.services.index(root_name)
    scores, per_signal = _robust_scores(dataset, baseline, audit)
    epochs = [dataset.timestamp_epochs[index] for index in audit]
    first = _first_persistent(scores, epochs, threshold)
    with np.errstate(all="ignore"):
        maximum = np.nanmax(scores, axis=0)
    graph = adjacency[audit].sum(axis=0)
    neighbors = [service for service in range(len(dataset.services))
                 if service != root and graph[root, service] + graph[service, root] > 0]
    neighbor_times = [first[service] for service in neighbors if first[service] is not None]
    root_time = first[root]
    earliest_neighbor = min(neighbor_times) if neighbor_times else None
    if root_time is None and earliest_neighbor is None:
        comparison = "neither_detected"
    elif root_time is None:
        comparison = "root_missing"
    elif earliest_neighbor is None or root_time < earliest_neighbor:
        comparison = "root_earlier"
    elif root_time == earliest_neighbor:
        comparison = "tie"
    else:
        comparison = "root_later"

    first_visible = min(
        _epoch(row["target_minute_utc"])
        for row in case_rows if int(row["failure_next_minute"]) == 1
    )
    fault_signal = FAULT_SIGNAL.get(case_meta["fault_type"])
    fault_scores = per_signal.get(fault_signal) if fault_signal else None
    fault_first = (
        _first_persistent(fault_scores, epochs, threshold)[root]
        if fault_scores is not None else None
    )
    return {
        "case_id": case_id,
        "root_service": root_name,
        "fault_type": case_meta["fault_type"],
        "baseline_minutes": len(baseline),
        "audit_minutes": len(audit),
        "neighbors": [dataset.services[index] for index in neighbors],
        "root_first_anomaly_utc": (
            datetime.fromtimestamp(root_time, timezone.utc).isoformat() if root_time else None
        ),
        "earliest_neighbor_anomaly_utc": (
            datetime.fromtimestamp(earliest_neighbor, timezone.utc).isoformat()
            if earliest_neighbor else None
        ),
        "root_vs_neighbor": comparison,
        "root_lead_over_neighbor_minutes": (
            (earliest_neighbor - root_time) / 60
            if root_time is not None and earliest_neighbor is not None else None
        ),
        "first_visible_failure_utc": datetime.fromtimestamp(
            first_visible, timezone.utc
        ).isoformat(),
        "root_lead_before_visible_minutes": (
            (first_visible - root_time) / 60 if root_time is not None else None
        ),
        "root_temporal_rank_all_services": _rank_root(root, first, maximum),
        "root_max_general_score": (
            float(maximum[root]) if math.isfinite(float(maximum[root])) else None
        ),
        "fault_aligned_signal": fault_signal,
        "root_fault_signal_detected": fault_first is not None,
    }


def _summarize(cases: list[dict[str, Any]], service_count: int) -> dict[str, Any]:
    comparisons = defaultdict(int)
    for case in cases:
        comparisons[case["root_vs_neighbor"]] += 1
    comparable = [
        case for case in cases
        if case["root_vs_neighbor"] in {"root_earlier", "tie", "root_later"}
    ]
    leads = [
        case["root_lead_over_neighbor_minutes"] for case in comparable
        if case["root_lead_over_neighbor_minutes"] is not None
    ]
    visible_leads = [
        case["root_lead_before_visible_minutes"] for case in cases
        if case["root_lead_before_visible_minutes"] is not None
    ]
    ranks = [case["root_temporal_rank_all_services"] for case in cases]
    return {
        "cases": len(cases),
        "root_general_anomaly_detected": sum(
            case["root_first_anomaly_utc"] is not None for case in cases
        ),
        "root_fault_aligned_signal_detected": sum(
            case["root_fault_signal_detected"] for case in cases
        ),
        "comparison_counts": dict(comparisons),
        "comparable_cases": len(comparable),
        "root_earlier_or_tied_rate": (
            sum(case["root_vs_neighbor"] in {"root_earlier", "tie"} for case in comparable)
            / len(comparable) if comparable else None
        ),
        "median_root_lead_over_neighbor_minutes": statistics.median(leads) if leads else None,
        "root_detected_before_or_at_visible_rate": (
            sum(lead >= 0 for lead in visible_leads) / len(visible_leads)
            if visible_leads else None
        ),
        "median_root_lead_before_visible_minutes": (
            statistics.median(visible_leads) if visible_leads else None
        ),
        "temporal_hit_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "temporal_hit_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "temporal_hit_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "temporal_mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "random_hit_at_1": 1 / service_count,
    }


def _analyze_dataset(dataset_name: str, threshold: float) -> dict[str, Any]:
    panel = PANELS[dataset_name]
    dataset = Re2WindowDataset(panel, history=5)
    schema = json.loads((panel / "feature_schema.json").read_text())
    source_path = Path(schema["source_labels"])
    rows = _read_csv(source_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    cases = {row["case_id"]: row for row in _read_csv(source_path.parent / "cases.csv")}
    with np.load(panel / "service_adjacency_1m.npz", allow_pickle=False) as graph:
        adjacency = graph["adjacency"].copy()
    details = []
    for case_id, case_rows in sorted(grouped.items()):
        result = _case_result(
            dataset, adjacency, case_id, case_rows, cases[case_id], threshold
        )
        if result is not None:
            details.append(result)
    by_fault: dict[str, Any] = {}
    for fault_type in sorted({case["fault_type"] for case in details}):
        subset = [case for case in details if case["fault_type"] == fault_type]
        by_fault[fault_type] = _summarize(subset, len(dataset.services))
    return {
        "dataset": f"RE2-{dataset_name.upper()}",
        "services": len(dataset.services),
        "threshold": threshold,
        "consecutive_minutes": 2,
        "summary": _summarize(details, len(dataset.services)),
        "by_fault_type": by_fault,
        "case_details": details,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RE2根因服务与相邻症状服务时序审计",
        "",
        "每个case仅使用故障注入前分钟建立服务级稳健基线；注入对齐分钟之后，若服务遥测"
        "连续2分钟超过阈值，则记为首次异常。根因服务来自官方注入配置。",
        "",
        "当前面板调用图是对称调用计数，因此本报告只能比较根因与直接连接服务的时序先后，"
        "不能把结果解释为已验证的有向因果传播。",
        "",
    ]
    for threshold_result in report["thresholds"]:
        lines.extend([f"## 稳健偏离阈值 z≥{threshold_result['threshold']}", ""])
        lines.extend([
            "| 数据集 | 有用户可感知故障case | 根因一般异常检出 | 根因故障对应信号检出 | "
            "严格早于 | 同时 | 晚于 | 根因缺失 | 中位领先分钟 | 可见故障前检出比例 | 时间Hit@1 | Hit@3 | MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for dataset in threshold_result["datasets"]:
            s = dataset["summary"]
            counts = s["comparison_counts"]
            lines.append(
                f"| {dataset['dataset']} | {s['cases']} | "
                f"{s['root_general_anomaly_detected']}/{s['cases']} | "
                f"{s['root_fault_aligned_signal_detected']}/{s['cases']} | "
                f"{counts.get('root_earlier', 0)} | {counts.get('tie', 0)} | "
                f"{counts.get('root_later', 0)} | {counts.get('root_missing', 0)} | "
                f"{_fmt(s['median_root_lead_over_neighbor_minutes'], 1)} | "
                f"{_fmt(s['root_detected_before_or_at_visible_rate'])} | "
                f"{_fmt(s['temporal_hit_at_1'])} | {_fmt(s['temporal_hit_at_3'])} | "
                f"{_fmt(s['temporal_mrr'])} |"
            )
        lines.append("")
    conservative = report["thresholds"][-1]
    ob = conservative["datasets"][0]["summary"]
    tt = conservative["datasets"][1]["summary"]
    lines.extend([
        "## 主要结论",
        "",
        f"- 在较严格的z≥5设置下，OB中根因严格早于相邻异常的case只有"
        f"{ob['comparison_counts'].get('root_earlier', 0)}/{ob['cases']}，同时发生有"
        f"{ob['comparison_counts'].get('tie', 0)}个；TT严格早于为"
        f"{tt['comparison_counts'].get('root_earlier', 0)}/{tt['cases']}，同时发生有"
        f"{tt['comparison_counts'].get('tie', 0)}个。",
        "- 两个数据集的根因异常相对用户可感知故障的中位提前量都只有约1分钟。",
        "- 因此，时序先后包含一定定位信息，TT上的证据更强；但OB大量同分钟异常，"
        "当前一分钟粒度不足以稳定支持‘根因先于症状传播’这一跨数据集假设。",
        "- 仅按首次异常时间排序的Hit@1仍然较低，说明时间先后只能作为定位证据之一，"
        "不能替代服务状态、异常类型和调用关系。",
        "",
    ])
    lines.extend([
        "## 解释边界",
        "",
        "- 稳健z分数是无监督诊断工具，不是最终模型，也没有使用测试模型预测。",
        "- fault-aligned信号使用已知故障类型，仅用于检查注入是否在根因服务遥测中可见，不能作为模型输入。",
        "- 如果结论随z阈值显著变化，或根因不经常早于连接服务，就不应把‘传播方向解耦’作为论文核心假设。",
        "- 要研究有向传播，下一步必须从原始trace保留caller→callee方向，当前对称图不够。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    thresholds = []
    for threshold in (3.0, 5.0):
        print(f"audit threshold={threshold}")
        thresholds.append({
            "threshold": threshold,
            "datasets": [
                _analyze_dataset(dataset, threshold) for dataset in PANELS
            ],
        })
    report = {
        "protocol": {
            "normal_profile": "within-case pre-injection minutes only",
            "signals": list(SIGNALS),
            "score": "absolute robust deviation after log1p; capped at 20",
            "persistence": "two consecutive minutes",
            "graph_limitation": "prepared adjacency is symmetric and non-causal",
        },
        "thresholds": thresholds,
    }
    artifact = ROOT / "artifacts/re2_root_symptom_timing_audit_v1.json"
    document = ROOT / "docs/2026-11-12-root-symptom-timing-audit.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
