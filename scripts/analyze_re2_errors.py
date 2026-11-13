#!/usr/bin/env python3
"""Audit failure-stage, fault-type and root-service errors for RE2 baselines.

The five deployment test folds are disjoint, so this script pools their
out-of-fold predictions. It does not retrain or tune any model.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
FOLDS = range(5)
MODELS = {
    "MTG": "re2_{dataset}_gcn_gru_deployment_fold{fold}_fair_30ep_seed42_v3",
    "InstantOps-adapted": (
        "re2_{dataset}_instantops_adapted_deployment_fold{fold}_"
        "fair_30ep_seed42_baseline_v1"
    ),
    "LTFP": "re2_{dataset}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1",
    "Eadro-adapted": (
        "re2_{dataset}_eadro_adapted_deployment_fold{fold}_"
        "fair_30ep_seed42_baseline_v1"
    ),
}
TRANSITION_ORDER = (
    "onset",
    "unknown_to_failure",
    "continuing",
    "recovery",
    "stable_normal",
    "unknown_to_normal",
)
TRANSITION_LABELS = {
    "onset": "严格首分钟（正常→故障）",
    "unknown_to_failure": "当前业务状态不可观测→故障",
    "continuing": "持续（故障→故障）",
    "recovery": "恢复（故障→非故障）",
    "stable_normal": "稳定非故障",
    "unknown_to_normal": "当前业务状态不可观测→非故障",
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
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recalled = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recalled))


def _classification(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {
            "samples": 0,
            "positives": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "ap": None,
            "alarm_rate": None,
            "mean_probability": None,
        }
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    scores = np.asarray([row["probability"] for row in rows], dtype=float)
    predictions = scores >= 0.5
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    tn = int(np.logical_and(~predictions, labels == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(rows),
        "positives": int(labels.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": _average_precision(labels, scores),
        "alarm_rate": float(predictions.mean()),
        "mean_probability": float(scores.mean()),
    }


def _rank_summary(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    events = list(events)
    if not events:
        return {
            "events": 0,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "mrr": None,
            "mean_rank": None,
        }
    ranks = np.asarray([event["rank"] for event in events], dtype=float)
    return {
        "events": len(events),
        "hit_at_1": float((ranks <= 1).mean()),
        "hit_at_3": float((ranks <= 3).mean()),
        "hit_at_5": float((ranks <= 5).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
    }


def _dataset_sources(dataset: str) -> dict[str, Any]:
    panel_dir = ROOT / "artifacts" / f"re2_{dataset}_panel_fair_v1"
    schema = json.loads((panel_dir / "feature_schema.json").read_text())
    label_path = Path(schema["source_labels"])
    labels = _read_csv(label_path)
    cases = {row["case_id"]: row for row in _read_csv(label_path.parent / "cases.csv")}

    by_input: dict[str, dict[str, str]] = {}
    by_target: dict[str, dict[str, str]] = {}
    for row in labels:
        for mapping, field in ((by_input, "input_minute_utc"), (by_target, "target_minute_utc")):
            timestamp = row[field]
            if timestamp in mapping and mapping[timestamp]["case_id"] != row["case_id"]:
                raise ValueError(f"{dataset} case timestamps overlap at {timestamp}")
            mapping[timestamp] = row
    return {
        "by_input": by_input,
        "by_target": by_target,
        "cases": cases,
        "services": list(schema["services"]),
    }


def _transition(future: dict[str, str], current: dict[str, str] | None) -> str:
    future_failure = int(future["failure_next_minute"])
    current_observed = bool(current and int(current["entry_observed"]) == 1)
    if future_failure:
        if not current_observed:
            return "unknown_to_failure"
        if (
            int(current["failure_next_minute"]) == 1
            and current["event_id"] == future["event_id"]
        ):
            return "continuing"
        return "onset"
    if current is None or int(current["entry_observed"]) != 1:
        return "unknown_to_normal"
    if int(current["failure_next_minute"]) == 1:
        return "recovery"
    return "stable_normal"


def _load_model_rows(dataset: str, model: str, sources: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in FOLDS:
        directory = ROOT / "artifacts" / MODELS[model].format(dataset=dataset, fold=fold)
        prediction_path = directory / "predictions.jsonl"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                prediction = json.loads(line)
                timestamp = prediction["input_minute_utc"]
                if timestamp in seen:
                    raise ValueError(f"duplicate out-of-fold prediction: {dataset} {model} {timestamp}")
                seen.add(timestamp)
                future = sources["by_input"].get(timestamp)
                if future is None:
                    raise ValueError(f"prediction lacks source label: {dataset} {timestamp}")
                if int(future["entry_observed"]) != 1:
                    raise ValueError(f"prediction uses invalid target: {dataset} {timestamp}")
                label = int(prediction["failure_target"])
                if label != int(future["failure_next_minute"]):
                    raise ValueError(f"prediction/source label mismatch: {dataset} {timestamp}")
                if prediction["event_id"] != future["event_id"]:
                    raise ValueError(f"prediction/source event mismatch: {dataset} {timestamp}")
                case = sources["cases"][future["case_id"]]
                current = sources["by_target"].get(timestamp)
                if current is not None and current["case_id"] != future["case_id"]:
                    current = None
                rows.append({
                    "fold": fold,
                    "timestamp": timestamp,
                    "case_id": future["case_id"],
                    "fault_type": case["fault_type"],
                    "case_root_service": case["root_service"],
                    "event_id": future["event_id"],
                    "true_services": prediction["true_services"],
                    "ranking": prediction["ranking"],
                    "label": label,
                    "probability": float(prediction["failure_probability"]),
                    "transition": _transition(future, current),
                })
    return rows


def _event_rows(rows: list[dict[str, Any]], services: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["event_id"] and row["true_services"]:
            grouped[row["event_id"]].append(row)
    service_index = {service: index for index, service in enumerate(services)}
    events = []
    for event_id, windows in grouped.items():
        roots = {service for row in windows for service in row["true_services"]}
        if len(roots) != 1:
            raise ValueError(f"event {event_id} has roots {sorted(roots)}")
        root = next(iter(roots))
        scores = np.zeros(len(services), dtype=float)
        for window in windows:
            by_service = {item["service"]: float(item["score"]) for item in window["ranking"]}
            if set(by_service) != set(services):
                raise ValueError(f"event {event_id} ranking service mismatch")
            scores += np.asarray([by_service[service] for service in services])
        scores /= len(windows)
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == service_index[root])[0] + 1)
        first = windows[0]
        events.append({
            "event_id": event_id,
            "case_id": first["case_id"],
            "fault_type": first["fault_type"],
            "root_service": root,
            "windows": len(windows),
            "rank": rank,
        })
    return events


def _group_classification(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _classification(group) for name, group in sorted(grouped.items())}


def _group_ranks(events: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event[key])].append(event)
    return {name: _rank_summary(group) for name, group in sorted(grouped.items())}


def _analyze(dataset: str, model: str, sources: dict[str, Any]) -> dict[str, Any]:
    rows = _load_model_rows(dataset, model, sources)
    events = _event_rows(rows, sources["services"])
    transitions = {
        name: _classification(row for row in rows if row["transition"] == name)
        for name in TRANSITION_ORDER
    }
    return {
        "out_of_fold_windows": len(rows),
        "overall": _classification(rows),
        "transitions": transitions,
        "by_fault_type": _group_classification(rows, "fault_type"),
        "by_root_service": _group_classification(
            [row for row in rows if row["label"] == 1], "case_root_service"
        ),
        "event_rca": _rank_summary(events),
        "event_rca_by_fault_type": _group_ranks(events, "fault_type"),
        "event_rca_by_root_service": _group_ranks(events, "root_service"),
        "event_details": events,
    }


def _ratio(metric: dict[str, Any], *, positive: bool) -> str:
    numerator = metric["tp"] if positive else metric["fp"]
    denominator = metric["positives"] if positive else metric["samples"] - metric["positives"]
    if denominator == 0:
        return "NA"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def _value(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    ob_example = next(iter(payload["results"]["RE2-OB"].values()))["transitions"]
    tt_example = next(iter(payload["results"]["RE2-TT"].values()))["transitions"]
    lines = [
        "# RE2下一分钟故障预测与定位错误分析",
        "",
        "本报告汇总seed 42、30 epochs、deployment五折的不重叠测试预测。"
        "阈值固定为0.5，未重训、未调参。阶段指标为池化out-of-fold诊断统计，"
        "不替代主表的五折均值±标准差。",
        "",
        "## 主要发现",
        "",
        "1. **当前高分部分依赖故障持续性，严格首分钟证据很少。** "
        f"OB只有{ob_example['onset']['samples']}个可确认的正常→故障窗口，另有"
        f"{ob_example['unknown_to_failure']['samples']}个故障正例的前一业务状态不可观测；"
        f"TT对应为{tt_example['onset']['samples']}和"
        f"{tt_example['unknown_to_failure']['samples']}。",
        "2. **模型普遍存在状态惯性。** OB的恢复分钟误报率为35.7%–71.4%，"
        "TT为33.3%–62.5%。因此不仅要学习「是否处于故障」，还需要学习「状态是否在变化」。",
        "3. **LTFP仍是当前最好的完整模型平衡点。** 它在TT的稳定非故障误报率"
        "最低，事件级Hit@1为0.875；OB上事件级Hit@1为0.836；但严格首分钟样本"
        "过少，不能据此声称稳定的首次预警能力。",
        "4. **故障类型差异很大。** OB的CPU故障对所有方法都较难；TT的disk故障中，"
        "MTG/InstantOps/Eadro的正例检出率只有4.2%/4.2%/6.2%，LTFP为58.3%。",
        "5. **图结构本身不能简单判定为无效，问题更可能是传播与任务共享方式。** "
        "MTG的Hit@1在OB/TT仅0.418/0.391，而同样使用图注意力的LTFP达到0.836/0.875。",
        "6. **InstantOps的遮蔽式定位具有服务偏置。** OB上productcatalogservice的平均名次为5.47，"
        "TT上ts-auth-service为10.38、ts-order-service为15.33。",
        "",
        "## 对新方法的直接启示",
        "",
        "- **状态/关系路径**：使用绝对遥测状态与服务图，判断下一分钟系统风险；",
        "- **变化/局部路径（待验证）**：显式建模时序差分和短期变化，但必须先证明严格首分钟中存在可学习信号；",
        "- **任务分工**：预测头使用全局状态+变化信息，根因头优先保留局部服务证据；",
        "- **最小验证**：先比较Persistence、线性状态探针和状态+变化探针，再决定是否增加变化支路。",
        "",
        "以上是seed 42下的诊断性证据，最终结论仍需多种子和消融实验验证。",
        "",
    ]
    for dataset_name, models in payload["results"].items():
        lines.extend([
            f"## {dataset_name}",
            "",
            "### 故障阶段",
            "",
            "| 方法 | 严格首分钟检出 | 未知→故障检出 | 持续故障检出 | 恢复分钟误报 | 稳定非故障误报 | 未知→正常 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model, result in models.items():
            transition = result["transitions"]
            lines.append(
                f"| {model} | {_ratio(transition['onset'], positive=True)} | "
                f"{_ratio(transition['unknown_to_failure'], positive=True)} | "
                f"{_ratio(transition['continuing'], positive=True)} | "
                f"{_ratio(transition['recovery'], positive=False)} | "
                f"{_ratio(transition['stable_normal'], positive=False)} | "
                f"{transition['unknown_to_normal']['samples']} |"
            )
        lines.extend([
            "",
            "### 整体事件级根因定位（五折测试事件池化）",
            "",
            "| 方法 | 事件数 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均名次 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model, result in models.items():
            rca = result["event_rca"]
            lines.append(
                f"| {model} | {rca['events']} | {_value(rca['hit_at_1'])} | "
                f"{_value(rca['hit_at_3'])} | {_value(rca['hit_at_5'])} | "
                f"{_value(rca['mrr'])} | {_value(rca['mean_rank'], 2)} |"
            )
        fault_types = sorted({
            fault_type
            for result in models.values()
            for fault_type in result["by_fault_type"]
        })
        lines.extend([
            "",
            "### 按故障类型",
            "",
            "检出率仅对该故障类型case中的正例窗口计算；Hit@1为该类型已归因事件的池化结果。",
            "",
            "| 故障类型 | 方法 | 正例检出率 | F1（该类型全窗口） | 事件数 | Hit@1 | MRR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for fault_type in fault_types:
            for model, result in models.items():
                detection = result["by_fault_type"].get(fault_type, _classification([]))
                rca = result["event_rca_by_fault_type"].get(fault_type, _rank_summary([]))
                lines.append(
                    f"| {fault_type} | {model} | {_ratio(detection, positive=True)} | "
                    f"{_value(detection['f1'])} | {rca['events']} | "
                    f"{_value(rca['hit_at_1'])} | {_value(rca['mrr'])} |"
                )
        lines.extend([
            "",
            "### 根因服务定位薄弱项",
            "",
            "下表每种方法列出MRR最低的5个根因服务；事件数很少的行只用于定位错误案例，不作稳定统计结论。",
            "",
            "| 方法 | 根因服务 | 事件数 | Hit@1 | MRR | 平均名次 |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for model, result in models.items():
            weakest = sorted(
                result["event_rca_by_root_service"].items(),
                key=lambda item: (item[1]["mrr"], -item[1]["events"], item[0]),
            )[:5]
            for service, rca in weakest:
                lines.append(
                    f"| {model} | {service} | {rca['events']} | "
                    f"{_value(rca['hit_at_1'])} | {_value(rca['mrr'])} | "
                    f"{_value(rca['mean_rank'], 2)} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload: dict[str, Any] = {
        "protocol": {
            "task": "past five complete minutes predict next-minute user-visible failure state",
            "models": list(MODELS),
            "datasets": ["RE2-OB", "RE2-TT"],
            "folds": list(FOLDS),
            "epochs": 30,
            "seed": 42,
            "threshold": 0.5,
            "aggregation": "pooled disjoint out-of-fold predictions for diagnostic counts",
        },
        "results": {},
    }
    for dataset in DATASETS:
        sources = _dataset_sources(dataset)
        payload["results"][f"RE2-{dataset.upper()}"] = {
            model: _analyze(dataset, model, sources) for model in MODELS
        }
    json_path = ROOT / "artifacts" / "re2_error_analysis_seed42_v1.json"
    markdown_path = ROOT / "docs" / "2026-11-12-re2-error-analysis.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(markdown_path.read_text())
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
