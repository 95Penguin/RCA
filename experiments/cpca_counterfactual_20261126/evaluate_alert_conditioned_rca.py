#!/usr/bin/env python3
"""Evaluate forecast-gated CPCA without changing either method.

The saved OCSID row at minute ``t`` uses observations through ``t`` and its
label describes ``t+1``.  The CPCA audit row with the same case/minute uses
only requests completed by ``t``.  This script joins those rows and reports:

* ordinary alert precision/recall using each fold's validation-selected gate;
* RCA conditional on a true-positive alert;
* end-to-end RCA where a missed alert or missing attribution receives 0 RR;
* event onset alerting, first-alert delay, first-alert RCA, and retrospective
  aggregation over only the alerted windows of an event.

The last aggregation is explicitly retrospective and is not an onset result.
All RCA ranks use the full service universe, zero-fill absent services, assign
the worst rank to ties, and treat no positive evidence as abstention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scores(window: dict | None, method: str) -> dict[str, float]:
    if not window:
        return {}
    by_method = window.get("scores_by_method", {})
    if method in by_method:
        return by_method[method]
    if method == "cpca":
        return window.get("cpca_scores", {})
    return {}


def _rank(scores: dict[str, float], root: str, services: list[str]) -> int | None:
    values = [float(scores.get(service, 0.0)) for service in services]
    if max(values, default=0.0) <= 0.0:
        return None
    root_value = float(scores.get(root, 0.0))
    return sum(value >= root_value for value in values)


def _metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if not n:
        return {
            "n": 0, "hit_at_1": None, "hit_at_3": None, "hit_at_5": None,
            "mrr": None, "abstentions": 0,
        }
    return {
        "n": n,
        "hit_at_1": sum(rank == 1 for rank in ranks) / n,
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / n,
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "mrr": sum(1.0 / rank for rank in ranks if rank is not None) / n,
        "abstentions": sum(rank is None for rank in ranks),
    }


def _mean_scores(rows: list[dict[str, float]], services: list[str]) -> dict[str, float]:
    return {
        service: sum(scores.get(service, 0.0) for scores in rows) / len(rows)
        for service in services
    }


def _audit_index(audit: dict) -> tuple[dict[tuple[str, int], dict], dict]:
    windows: dict[tuple[str, int], dict] = {}
    diagnostics = defaultdict(int)
    positive_minutes = 0
    scored_windows = 0
    for case in audit["cases"].values():
        positive_minutes += int(case.get("positive_minutes", 0))
        scored_windows += int(case.get("scored_windows", 0))
        for name, value in case.get("tree_diagnostics", {}).items():
            diagnostics[name] += int(value)
        for window in case.get("windows", []):
            key = (case["case_id"], int(window["minute"]))
            if key in windows:
                raise ValueError(f"duplicate audit window {key}")
            windows[key] = window
    examined = diagnostics["positive_requests_examined"]
    invalid = diagnostics["invalid_positive_trees"]
    coverage = {
        "positive_label_minutes": positive_minutes,
        "scored_windows": scored_windows,
        "window_coverage": scored_windows / positive_minutes if positive_minutes else None,
        **dict(diagnostics),
        "valid_entry_request_fraction": (
            (examined - invalid) / examined if examined else None
        ),
        "note": (
            "excluded_non_entry_requests counts trace IDs without a recognized business "
            "entry and is not included in the valid-entry request denominator; current "
            "artifacts do not split invalid trees by rejection reason"
        ),
    }
    return windows, coverage


def _evaluate_method(
    predictions: list[dict], windows: dict, services: list[str], method: str,
) -> dict:
    positives = [
        row for row in predictions
        if int(row["failure_target"]) == 1 and row.get("root") and row.get("event")
    ]
    seen = set()
    offline_ranks = []
    conditional_ranks = []
    end_to_end_ranks = []
    event_rows = defaultdict(list)
    covered_positive = 0
    covered_alerted = 0
    for row in positives:
        key = (row["case"], int(row["minute"]))
        if key in seen:
            raise ValueError(f"duplicate prediction window {key}")
        seen.add(key)
        window = windows.get(key)
        scores = _scores(window, method)
        rank = _rank(scores, row["root"], services)
        covered_positive += int(window is not None)
        offline_ranks.append(rank)
        if row["alert"]:
            covered_alerted += int(window is not None)
            conditional_ranks.append(rank)
            end_to_end_ranks.append(rank)
        else:
            end_to_end_ranks.append(None)
        event_rows[row["event"]].append({**row, "scores": scores, "rank": rank})

    detected_event_ranks = []
    end_to_end_event_ranks = []
    first_alert_ranks = []
    onset_alerts = 0
    detected_events = 0
    offsets = []
    event_details = {}
    for event, rows in event_rows.items():
        rows.sort(key=lambda row: int(row["minute"]))
        roots = {row["root"] for row in rows}
        if len(roots) != 1:
            raise ValueError(f"event {event!r} has inconsistent roots")
        root = rows[0]["root"]
        alerted = [row for row in rows if row["alert"]]
        onset_alerts += int(rows[0]["alert"])
        if alerted:
            detected_events += 1
            first_index = next(index for index, row in enumerate(rows) if row["alert"])
            offsets.append(first_index)
            first_rank = alerted[0]["rank"]
            aggregate_rank = _rank(
                _mean_scores([row["scores"] for row in alerted], services), root, services
            )
            first_alert_ranks.append(first_rank)
            detected_event_ranks.append(aggregate_rank)
            end_to_end_event_ranks.append(aggregate_rank)
        else:
            first_index = None
            first_rank = None
            aggregate_rank = None
            first_alert_ranks.append(None)
            end_to_end_event_ranks.append(None)
        event_details[event] = {
            "root": root,
            "positive_windows": len(rows),
            "alerted_windows": len(alerted),
            "onset_alert": bool(rows[0]["alert"]),
            "first_alert_offset_windows": first_index,
            "first_alert_rank": first_rank,
            "alerted_window_mean_rank": aggregate_rank,
        }

    events_n = len(event_rows)
    return {
        "method": method,
        "positive_windows": len(positives),
        "covered_positive_windows": covered_positive,
        "alerted_positive_windows": sum(row["alert"] for row in positives),
        "covered_alerted_positive_windows": covered_alerted,
        "offline_all_positive_windows": _metrics(offline_ranks),
        "conditional_on_true_positive_alert_windows": _metrics(conditional_ranks),
        "end_to_end_all_positive_windows_missed_alert_is_zero": _metrics(end_to_end_ranks),
        "events": {
            "n": events_n,
            "detected": detected_events,
            "detected_fraction": detected_events / events_n if events_n else None,
            "onset_alerts": onset_alerts,
            "onset_alert_fraction": onset_alerts / events_n if events_n else None,
            "first_alert_offset_windows": {
                "mean": statistics.mean(offsets) if offsets else None,
                "median": statistics.median(offsets) if offsets else None,
                "max": max(offsets) if offsets else None,
            },
            "first_alert_rca_all_events_missed_alert_is_zero": _metrics(first_alert_ranks),
            "retrospective_alerted_window_mean_detected_events_only": _metrics(
                detected_event_ranks
            ),
            "retrospective_alerted_window_mean_all_events_missed_alert_is_zero": _metrics(
                end_to_end_event_ranks
            ),
        },
        "event_details": event_details,
    }


def _evaluate_prediction(path: Path, audit: dict, methods: list[str]) -> dict:
    cohort = _load(path)
    if cohort["dataset"] != audit["dataset"]:
        raise ValueError(f"dataset mismatch: {path}")
    services = list(cohort["folds"][0]["services"])
    if any(list(fold["services"]) != services for fold in cohort["folds"]):
        raise ValueError(f"service order differs across folds: {path}")
    predictions = []
    tp = fp = tn = fn = 0
    for fold in cohort["folds"]:
        threshold = float(fold["calibration"]["threshold"])
        for source in fold["test_predictions"]:
            row = dict(source)
            row["fold"] = int(fold["fold"])
            row["threshold"] = threshold
            row["alert"] = float(row["probability"]) >= threshold
            target = int(row["failure_target"])
            tp += int(row["alert"] and target == 1)
            fp += int(row["alert"] and target == 0)
            tn += int(not row["alert"] and target == 0)
            fn += int(not row["alert"] and target == 1)
            predictions.append(row)
    windows, coverage = _audit_index(audit)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "source": str(path),
        "sha256": _sha256(path),
        "seed": cohort.get("config", {}).get("seed"),
        "implementation_version": cohort.get("implementation_version"),
        "alert_gate": "per-fold validation-selected raw-probability threshold",
        "classification": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        },
        "audit_coverage": coverage,
        "methods": {
            method: _evaluate_method(predictions, windows, services, method)
            for method in methods
        },
    }


def _summary(results: list[dict], method: str) -> dict:
    paths = {
        "alert_recall": ("classification", "recall"),
        "alert_precision": ("classification", "precision"),
        "onset_alert_fraction": ("methods", method, "events", "onset_alert_fraction"),
        "event_detected_fraction": ("methods", method, "events", "detected_fraction"),
        "first_alert_event_mrr": (
            "methods", method, "events",
            "first_alert_rca_all_events_missed_alert_is_zero", "mrr",
        ),
        "end_to_end_window_mrr": (
            "methods", method,
            "end_to_end_all_positive_windows_missed_alert_is_zero", "mrr",
        ),
    }
    summary = {}
    for name, keys in paths.items():
        values = []
        for result in results:
            value = result
            for key in keys:
                value = value[key]
            values.append(float(value))
        summary[name] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def _markdown(report: dict, method: str) -> str:
    lines = [
        f"# {report['dataset'].upper()} 告警门控后的 CPCA 审计",
        "",
        "预测行 minute=t 使用截至 t 的五分钟观测并预测 t+1；同键 CPCA 使用截至 t 完成的请求。",
        "每折使用验证集选择的原始概率阈值。漏报告警、缺失归因或无正分数在端到端指标中计 0。",
        "",
        "| seed | 告警 Precision | 告警 Recall | 事件检出 | onset 告警 | 首次告警事件 MRR | 端到端窗口 MRR |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["runs"]:
        cls = result["classification"]
        item = result["methods"][method]
        events = item["events"]
        lines.append(
            f"| {result['seed']} | {_pct(cls['precision'])} | {_pct(cls['recall'])} | "
            f"{_pct(events['detected_fraction'])} | {_pct(events['onset_alert_fraction'])} | "
            f"{events['first_alert_rca_all_events_missed_alert_is_zero']['mrr']:.4f} | "
            f"{item['end_to_end_all_positive_windows_missed_alert_is_zero']['mrr']:.4f} |"
        )
    lines.extend([
        "",
        "说明：首次告警事件 MRR 只使用每个事件第一次触发告警时的 CPCA 排名，并把从未告警的事件计 0；",
        "事件内所有已告警窗口的均值结果仅是回顾性指标，不能写成首次预警定位。",
        "",
    ])
    coverage = report["runs"][0]["audit_coverage"]
    lines.extend([
        "## 当前树覆盖审计",
        "",
        f"- 正例标签窗口：{coverage['positive_label_minutes']}；有 CPCA 输出：{coverage['scored_windows']}（{_pct(coverage['window_coverage'])}）。",
        f"- 检查的业务入口请求：{coverage['positive_requests_examined']}；无效树：{coverage['invalid_positive_trees']}；有效入口请求比例：{_pct(coverage['valid_entry_request_fraction'])}。",
        f"- entry-component 省略的 span 数：{coverage['component_omitted_spans']}。当前产物没有按并行、缺父、重复 ID、时钟越界等原因拆分，也没有省略 span 的比例分母。",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+",
        default=["cpca_uniform"],
        help=(
            "audit score fields to gate; cpca_uniform is the frozen main median "
            "score. Mechanism controls should use FINAL_RESULTS.json because older "
            "audit field names retain historical hard-operation semantics"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    audit_path = args.audit or HERE / "attribution" / args.dataset / "audit.json"
    for destination in (args.output, args.markdown):
        if destination is not None and destination.exists():
            parser.error(f"output exists: {destination}")
    audit = _load(audit_path)
    if audit["dataset"] != args.dataset:
        parser.error("audit dataset does not match --dataset")
    runs = [_evaluate_prediction(path, audit, args.methods) for path in args.predictions]
    report = {
        "dataset": args.dataset,
        "protocol": (
            "forecast-gated RCA; validation-selected per-fold alert threshold; "
            "same case/minute join; full service universe; zero fill; worst ties; "
            "missed alert or no evidence is 0 RR"
        ),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "runs": runs,
        "summary": {method: _summary(runs, method) for method in args.methods},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report, "cpca_uniform"), encoding="utf-8")
    print(json.dumps(report["summary"]["cpca_uniform"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
