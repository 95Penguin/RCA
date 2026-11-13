#!/usr/bin/env python3
"""Inspect RE2-TT root HTTP spans and candidate latency-impact labels."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-operation-spans", type=int, default=20)
    parser.add_argument("--latency-ratio", type=float, default=1.5)
    parser.add_argument("--min-latency-increase-seconds", type=float, default=0.1)
    args = parser.parse_args()
    operation_totals = Counter()
    root_services = Counter()
    all_services = Counter()
    status_codes = Counter()
    case_reports = []
    with zipfile.ZipFile(args.archive) as archive:
        cases = sorted({name.rsplit("/", 1)[0] for name in archive.namelist()
                        if name.endswith("/inject_time.txt")})
        for index, case in enumerate(cases, 1):
            injection = int(archive.read(case + "/inject_time.txt"))
            injection_minute = injection // 60 * 60
            operation_minutes: dict[tuple[str, str, int], list[float]] = defaultdict(list)
            with archive.open(case + "/traces.csv") as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                for row in reader:
                    service = row["serviceName"]
                    all_services[service] += 1
                    parent = row["parentSpanID"].strip().lower()
                    if parent and parent not in {"nan", "null", "none"}:
                        continue
                    operation = row["operationName"]
                    if not operation.startswith(("GET ", "POST ", "PUT ", "DELETE ", "PATCH ")):
                        continue
                    try:
                        duration = float(row["duration"]) / 1_000_000
                        completed = (int(row["startTime"]) + duration * 1_000_000) / 1_000_000
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(duration) or duration < 0:
                        continue
                    minute = math.floor(completed / 60) * 60
                    if not injection_minute - 720 <= minute < injection_minute + 720:
                        continue
                    operation_minutes[service, operation, minute].append(duration)
                    operation_totals[service + "|" + operation] += 1
                    root_services[service] += 1
                    status_codes[row["statusCode"]] += 1
            operations = sorted({(service, operation) for service, operation, _ in operation_minutes})
            qualified = []
            qualified_series = []
            highs_by_operation: dict[tuple[str, str], set[int]] = {}
            pre_high_by_operation: dict[tuple[str, str], set[int]] = {}
            pre_counts = []
            for service, operation in operations:
                baseline_p95 = []
                for minute in range(injection_minute - 600, injection_minute, 60):
                    values = operation_minutes.get((service, operation, minute), [])
                    if len(values) >= args.min_operation_spans:
                        baseline_p95.append(percentile(values, 0.95))
                if len(baseline_p95) < 5:
                    continue
                baseline = statistics.median(baseline_p95)
                threshold = max(args.latency_ratio * baseline,
                                baseline + args.min_latency_increase_seconds)
                high = set()
                pre_high = set()
                for minute in range(injection_minute - 600, injection_minute, 60):
                    values = operation_minutes.get((service, operation, minute), [])
                    if len(values) >= args.min_operation_spans and percentile(values, 0.95) >= threshold:
                        pre_high.add(minute)
                for minute in range(injection_minute + 60, injection_minute + 720, 60):
                    values = operation_minutes.get((service, operation, minute), [])
                    if len(values) >= args.min_operation_spans and percentile(values, 0.95) >= threshold:
                        high.add(minute)
                highs_by_operation[service, operation] = high
                pre_high_by_operation[service, operation] = pre_high
                qualified.append({"service": service, "operation": operation,
                                  "baseline_p95_seconds": round(baseline, 4),
                                  "threshold_seconds": round(threshold, 4),
                                  "pre_observed_minutes": len(baseline_p95),
                                  "high_post_minutes": len(high),
                                  "high_pre_minutes": len(pre_high)})
                qualified_series.append({
                    "service": service, "operation": operation,
                    "minutes": [
                        [minute, len(values), round(percentile(values, 0.95), 6)]
                        for (s, op, minute), values in sorted(operation_minutes.items())
                        if s == service and op == operation and values
                    ],
                })
                pre_counts.extend(len(operation_minutes.get((service, operation, m), []))
                                  for m in range(injection_minute - 600, injection_minute, 60))
            high_minutes = sorted(set().union(*highs_by_operation.values())
                                  if highs_by_operation else set())
            confirmed_minutes = set()
            for high in highs_by_operation.values():
                for minute in high:
                    if minute - 60 in high or minute + 60 in high:
                        confirmed_minutes.add(minute)
            sustained = []
            for minute in sorted(confirmed_minutes):
                if not sustained or minute != sustained[-1][-1] + 60:
                    sustained.append([])
                sustained[-1].append(minute)
            pre_high_minutes = set().union(*pre_high_by_operation.values()) if pre_high_by_operation else set()
            pre_confirmed_minutes = set()
            for high in pre_high_by_operation.values():
                for minute in high:
                    if minute - 60 in high or minute + 60 in high:
                        pre_confirmed_minutes.add(minute)
            case_reports.append({
                "case_id": case.replace("/", "_"),
                "root_service": case.split("/")[1].rsplit("_", 1)[0],
                "fault_type": case.split("/")[1].rsplit("_", 1)[1],
                "injection_epoch": injection,
                "root_http_spans": sum(len(values) for values in operation_minutes.values()),
                "root_http_operations": len(operations),
                "qualified_operations": qualified,
                "qualified_operation_series": qualified_series,
                "qualified_operation_count": len(qualified),
                "median_pre_operation_spans_per_minute": statistics.median(pre_counts) if pre_counts else 0,
                "high_post_minutes": len(high_minutes),
                "high_pre_minutes": len(pre_high_minutes),
                "confirmed_pre_minutes": len(pre_confirmed_minutes),
                "sustained_events": len(sustained),
                "sustained_impact_minutes": sum(map(len, sustained)),
                "event_start_epochs": [run[0] for run in sustained],
            })
            if index % 15 == 0:
                print(f"Scanned {index}/{len(cases)} TT cases", flush=True)
    report = {
        "archive": str(args.archive), "cases": len(case_reports),
        "candidate_rule": {"min_operation_spans": args.min_operation_spans,
                           "latency_ratio": args.latency_ratio,
                           "min_latency_increase_seconds": args.min_latency_increase_seconds,
                           "consecutive_minutes": 2,
                           "note": "exploratory root HTTP operation p95; not final labels"},
        "root_services": root_services.most_common(),
        "all_trace_service_count": len(all_services),
        "top_operations": operation_totals.most_common(25),
        "root_span_status_codes": status_codes.most_common(),
        "case_reports": case_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cases": report["cases"], "root_services": report["root_services"],
                      "all_trace_service_count": report["all_trace_service_count"],
                      "top_operations": report["top_operations"][:8],
                      "impact_cases": sum(x["sustained_events"] > 0 for x in case_reports),
                      "events": sum(x["sustained_events"] for x in case_reports)}, indent=2))


if __name__ == "__main__":
    main()
