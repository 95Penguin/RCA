#!/usr/bin/env python3
"""Gate A for OC-SID: audit operation-conditioned request-path evidence.

The audit streams raw RE2 traces case by case, reconstructs each completed
request DAG, and compares three label-free service scores on positive input
minutes:

* global_mean_duration: current service-level mean span duration;
* operation_inclusive: p95 inclusive latency within the currently riskiest
  entry operation;
* operation_exclusive: p95 exclusive latency within that operation.

Every reference distribution uses only the ten complete pre-injection minutes.
Operation selection and service scores use spans completed by the input minute;
the next-minute target is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.trace_paths import Span, entry_span, service_latencies  # noqa: E402
from training.metrics import rca_metrics  # noqa: E402


ARCHIVES = {
    "ob": ROOT / "data/RCAEval/RE2-OB.zip",
    "tt": ROOT / "data/RCAEval/RE2-TT.zip",
}
PANELS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}
SOURCE_LABELS = {
    "ob": ROOT / "data/processed/re2_ob_labels_v1/next_minute_labels.csv",
    "tt": ROOT / "data/processed/re2_tt_labels_v1/next_minute_labels.csv",
}
MIN_ENTRY_SPANS = {"ob": 100, "tt": 20}


@dataclass(frozen=True)
class Target:
    case_id: str
    input_epoch: int
    root_service: str
    event_id: str
    fault_type: str


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def _minute(microseconds: float) -> int:
    return math.floor(microseconds / 60_000_000) * 60


def _service(raw: str) -> str:
    name = raw.strip()
    return "frontendservice" if name in {"frontend", "frontend-external"} else name


def _case_member(case_id: str) -> str:
    dataset, remainder = case_id.split("_", 1)
    service_fault, repetition = remainder.rsplit("_", 1)
    return f"{dataset}/{service_fault}/{repetition}"


def _targets(dataset: str) -> dict[str, list[Target]]:
    grouped: dict[str, list[Target]] = defaultdict(list)
    with SOURCE_LABELS[dataset].open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["failure_next_minute"] != "1" or row["root_label_valid"] != "1":
                continue
            case_id = row["case_id"]
            root = row.get("root_cause_service", "")
            if not root:
                continue
            fault = case_id.rsplit("_", 2)[-2]
            grouped[case_id].append(Target(
                case_id=case_id,
                input_epoch=_epoch(row["input_minute_utc"]),
                root_service=root,
                event_id=row["event_id"],
                fault_type=fault,
            ))
    return grouped


def _percentile(values: list[float], quantile: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(math.ceil(quantile * len(ordered)) - 1, 0)])


def _robust_deviation(value: float | None, baseline: list[float], absolute: bool = True) -> float:
    if value is None or len(baseline) < 3:
        return 0.0
    transformed = np.log1p(np.maximum(np.asarray(baseline, dtype=float), 0.0))
    center = float(np.median(transformed))
    mad = float(np.median(np.abs(transformed - center))) * 1.4826
    scale = mad if mad >= 0.05 else max(0.05, 0.1 * abs(center))
    deviation = (math.log1p(max(value, 0.0)) - center) / scale
    return float(min(abs(deviation) if absolute else max(deviation, 0.0), 20.0))


def _read_case(
    archive: zipfile.ZipFile,
    dataset: str,
    case_id: str,
    targets: list[Target],
    services: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    member = _case_member(case_id)
    injection = int(archive.read(f"{member}/inject_time.txt").decode().strip())
    injection_minute = injection // 60 * 60
    baseline_minutes = set(range(injection_minute - 600, injection_minute, 60))
    input_minutes = {target.input_epoch for target in targets}
    desired_minutes = baseline_minutes | input_minutes
    lower = min(desired_minutes) - 300
    upper = max(desired_minutes) + 60

    traces: dict[str, list[Span]] = defaultdict(list)
    global_duration: dict[tuple[int, str], list[float]] = defaultdict(list)
    rows_read = 0
    with archive.open(f"{member}/traces.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        required = {
            "traceID", "spanID", "parentSpanID", "serviceName",
            "operationName", "startTime", "duration",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{member}/traces.csv lacks required fields")
        for row in reader:
            rows_read += 1
            try:
                start = float(row["startTime"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                continue
            end = start + duration
            minute = _minute(end)
            if minute < lower or minute >= upper:
                continue
            service = _service(row["serviceName"])
            span = Span(
                trace_id=row["traceID"], span_id=row["spanID"],
                parent_span_id=row["parentSpanID"], service=service,
                operation=row["operationName"], start_us=start, end_us=end,
            )
            traces[span.trace_id].append(span)
            if minute in desired_minutes and service in services:
                global_duration[minute, service].append(duration / 1_000.0)

    entry_duration: dict[tuple[int, str], list[float]] = defaultdict(list)
    path_values: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    path_services: dict[tuple[int, str], set[str]] = defaultdict(set)
    completed_traces = 0
    for spans in traces.values():
        entry = entry_span(spans, dataset)
        if entry is None:
            continue
        minute = _minute(entry.end_us)
        if minute not in desired_minutes:
            continue
        operation = f"{entry.service}|{entry.operation}"
        entry_duration[minute, operation].append(entry.duration_us / 1_000.0)
        latencies = service_latencies(spans, set(services))
        if not latencies:
            continue
        completed_traces += 1
        path_services[minute, operation].update(latencies)
        for service, (inclusive, exclusive) in latencies.items():
            path_values[minute, operation, service, "inclusive"].append(inclusive / 1_000.0)
            path_values[minute, operation, service, "exclusive"].append(exclusive / 1_000.0)

    baseline_entry: dict[str, list[float]] = defaultdict(list)
    operations = {operation for minute, operation in entry_duration if minute in baseline_minutes}
    for operation in operations:
        for minute in sorted(baseline_minutes):
            values = entry_duration.get((minute, operation), [])
            if len(values) >= MIN_ENTRY_SPANS[dataset]:
                baseline_entry[operation].append(_percentile(values))

    global_reference: dict[str, list[float]] = defaultdict(list)
    for service in services:
        for minute in sorted(baseline_minutes):
            values = global_duration.get((minute, service), [])
            if values:
                global_reference[service].append(float(np.mean(values)))

    path_reference: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for operation in operations:
        for service in services:
            for kind in ("inclusive", "exclusive"):
                for minute in sorted(baseline_minutes):
                    values = path_values.get((minute, operation, service, kind), [])
                    if values:
                        path_reference[operation, service, kind].append(_percentile(values))

    output = []
    for target in targets:
        minute = target.input_epoch
        operation_scores = []
        for operation, reference in baseline_entry.items():
            current = entry_duration.get((minute, operation), [])
            if len(current) < MIN_ENTRY_SPANS[dataset]:
                continue
            score = _robust_deviation(_percentile(current), reference, absolute=False)
            operation_scores.append((score, len(current), operation))
        selected = max(operation_scores, default=(0.0, 0, ""))
        operation = selected[2]
        scores = {
            "global_mean_duration": [],
            "operation_inclusive": [],
            "operation_exclusive": [],
            "operation_exclusive_high": [],
        }
        for service in services:
            global_values = global_duration.get((minute, service), [])
            global_current = float(np.mean(global_values)) if global_values else None
            scores["global_mean_duration"].append(
                _robust_deviation(global_current, global_reference[service])
            )
            for kind, method, absolute in (
                ("inclusive", "operation_inclusive", True),
                ("exclusive", "operation_exclusive", True),
                ("exclusive", "operation_exclusive_high", False),
            ):
                values = path_values.get((minute, operation, service, kind), [])
                current = _percentile(values) if values else None
                reference = path_reference.get((operation, service, kind), [])
                scores[method].append(_robust_deviation(current, reference, absolute=absolute))
        # These fixed, untrained sums are exploratory complementarity checks,
        # not the pre-registered Gate-A primary comparison.  All terms are on
        # the same capped robust-z scale.
        global_score = np.asarray(scores["global_mean_duration"])
        inclusive_score = np.asarray(scores["operation_inclusive"])
        exclusive_score = np.asarray(scores["operation_exclusive"])
        scores["global_plus_inclusive"] = (global_score + inclusive_score).tolist()
        scores["global_plus_exclusive"] = (global_score + exclusive_score).tolist()
        scores["global_plus_inclusive_exclusive"] = (
            global_score + inclusive_score + exclusive_score
        ).tolist()
        output.append({
            "case_id": case_id,
            "event_id": target.event_id,
            "input_epoch": minute,
            "root_service": target.root_service,
            "fault_type": target.fault_type,
            "selected_operation": operation,
            "operation_risk_score": selected[0],
            "operation_trace_count": selected[1],
            "root_on_selected_path": bool(
                operation and target.root_service in path_services.get((minute, operation), set())
            ),
            "scores": scores,
        })
    return output, {
        "case_id": case_id,
        "raw_span_rows": rows_read,
        "retained_trace_ids": len(traces),
        "completed_entry_traces": completed_traces,
        "baseline_operations": len(baseline_entry),
    }


def _event_metrics(rows: list[dict[str, object]], method: str, services: tuple[str, ...]):
    index = {service: position for position, service in enumerate(services)}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["event_id"])].append(row)
    targets, scores = [], []
    for members in grouped.values():
        target = np.zeros(len(services), dtype=np.float32)
        target[index[str(members[0]["root_service"])]] = 1.0
        targets.append(target)
        scores.append(np.mean([member["scores"][method] for member in members], axis=0))
    result = rca_metrics(np.stack(targets), np.stack(scores))
    result["positive_events"] = result.pop("positive_windows")
    return result


def _window_metrics(rows: list[dict[str, object]], method: str, services: tuple[str, ...]):
    index = {service: position for position, service in enumerate(services)}
    targets = np.zeros((len(rows), len(services)), dtype=np.float32)
    scores = []
    for position, row in enumerate(rows):
        targets[position, index[str(row["root_service"])]] = 1.0
        scores.append(row["scores"][method])
    return rca_metrics(targets, np.asarray(scores, dtype=np.float32))


def _onset_metrics(rows: list[dict[str, object]], method: str, services: tuple[str, ...]):
    first = {}
    for row in rows:
        event = str(row["event_id"])
        if event not in first or int(row["input_epoch"]) < int(first[event]["input_epoch"]):
            first[event] = row
    return _window_metrics(list(first.values()), method, services)


def _summarize(dataset: str, rows: list[dict[str, object]], services: tuple[str, ...]):
    methods = next(iter(rows))["scores"]
    result = {
        "dataset": dataset,
        "positive_windows": len(rows),
        "positive_events": len({row["event_id"] for row in rows}),
        "selected_operation_rate": float(np.mean([bool(row["selected_operation"]) for row in rows])),
        "root_on_selected_path_rate": float(np.mean([row["root_on_selected_path"] for row in rows])),
        "methods": {},
        "by_fault_type": {},
    }
    for method in methods:
        result["methods"][method] = {
            "window": _window_metrics(rows, method, services),
            "event": _event_metrics(rows, method, services),
            "onset": _onset_metrics(rows, method, services),
        }
    for fault in sorted({str(row["fault_type"]) for row in rows}):
        subset = [row for row in rows if row["fault_type"] == fault]
        result["by_fault_type"][fault] = {
            method: _event_metrics(subset, method, services) for method in methods
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=["ob", "tt"])
    parser.add_argument("--max-cases", type=int, default=None, help="smoke-test prefix only")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/re2_operation_source_impact_gate_a_v1.json",
    )
    args = parser.parse_args()
    report = {
        "protocol": {
            "task": "OC-SID Gate A request-path evidence audit",
            "input_boundary": "only spans completed by the input minute",
            "normal_reference": "ten complete pre-injection minutes within each case",
            "operation_selection": "largest positive entry-p95 robust deviation at input minute",
            "primary_comparison": "operation_exclusive versus global_mean_duration",
            "exploratory_combinations": (
                "fixed sums on a common robust-z scale; inspected after the primary comparison"
            ),
        },
        "datasets": {},
    }
    for dataset in args.datasets:
        schema = json.loads((PANELS[dataset] / "feature_schema.json").read_text())
        services = tuple(schema["services"])
        targets_by_case = _targets(dataset)
        case_ids = sorted(targets_by_case)
        if args.max_cases is not None:
            case_ids = case_ids[: args.max_cases]
        rows, case_quality = [], []
        with zipfile.ZipFile(ARCHIVES[dataset]) as archive:
            for position, case_id in enumerate(case_ids, 1):
                print(f"{dataset.upper()} case {position}/{len(case_ids)} {case_id}", flush=True)
                case_rows, quality = _read_case(
                    archive, dataset, case_id, targets_by_case[case_id], services
                )
                rows.extend(case_rows)
                case_quality.append(quality)
        summary = _summarize(dataset, rows, services)
        report["datasets"][dataset] = {
            "summary": summary,
            "case_quality": case_quality,
            "rows": rows,
        }
        print(json.dumps(summary["methods"], indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
