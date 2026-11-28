"""Run the author ShapleyIQ core algorithm on the fixed RE2 forecast cohort.

The upstream attribution code is copied unmodified to third_party/.  A tiny
LogServiceClient adapter maps the RE2 CSV span schema to its expected tree
schema.  We report both upstream's no-cache default (normal duration = 0) and
the same algorithm with a causal first-10-minute mean-duration reference.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import importlib.util
import io
import json
import logging
import math
import sys
import time
import types
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/95penguin/project/NextMinuteRCA")
OUT = Path(__file__).resolve().parent
VALIDATION = Path("/Users/95penguin/MyDocs/项目/NextMinuteRCA-validation-20261125")
sys.path.insert(0, str(ROOT / "src"))
from ocsid import cpca  # noqa: E402
from ocsid.evaluation import evaluate_attribution  # noqa: E402
from ocsid.audit import ARCHIVES, PREFIX  # noqa: E402

SERVICES = {
    "ob": ("adservice", "cartservice", "checkoutservice", "currencyservice",
           "emailservice", "frontendservice", "paymentservice", "productcatalogservice",
           "recommendationservice", "redis", "shippingservice"),
    "tt": ("ts-admin-basic-info-service", "ts-admin-travel-service", "ts-assurance-service",
           "ts-auth-service", "ts-basic-service", "ts-config-service", "ts-consign-price-service",
           "ts-consign-service", "ts-contacts-service", "ts-food-map-service", "ts-food-service",
           "ts-inside-payment-service", "ts-order-other-service", "ts-order-service",
           "ts-payment-service", "ts-preserve-other-service", "ts-preserve-service",
           "ts-price-service", "ts-route-service", "ts-seat-service", "ts-security-service",
           "ts-station-service", "ts-ticketinfo-service", "ts-train-service",
           "ts-travel-service", "ts-travel2-service", "ts-user-service"),
}


def fast_merge_timelines(self, timelines):
    """Exact search-range optimization of ShapleyIQ's upstream merge loop.

    The upstream code scans every later interval at every step. Since those
    intervals are sorted by integer start time, its strict overlap predicate
    is exactly the prefix ``start <= end + ceil(duration * threshold) - 1``.
    The integer form avoids precision loss for epoch-microsecond timestamps. This preserves
    candidate order, max-end tie behavior, recycled members and contributions.
    """
    remaining = sorted(timelines, key=lambda item: item[0])
    merged = []
    while remaining:
        starts = [item[0] for item in remaining]
        last_j = 0
        recycled = {0}
        big_ids = []
        big_start = remaining[0][0]
        while True:
            current = remaining[last_j]
            max_gap = math.ceil((current[1] - current[0]) * self.sync_overlap_threshold) - 1
            boundary = current[1] + max_gap
            limit = bisect_right(starts, boundary, lo=last_j + 1)
            synced = list(range(last_j + 1, limit))
            max_end_idx = last_j
            if synced:
                candidate = max(synced, key=lambda idx: remaining[idx][1])
                if remaining[candidate][1] > current[1]:
                    max_end_idx = candidate
            big_end = remaining[max_end_idx][1]
            big_ids.append(current[2])
            if not synced or last_j == max_end_idx:
                break
            recycled.update(synced)
            last_j = max_end_idx

        total_diff = sum(remaining[idx][3] for idx in recycled)
        merged.append((big_start, big_end, "~~".join(big_ids), total_diff))
        remaining = [item for idx, item in enumerate(remaining) if idx not in recycled]
    return merged


def load_upstream_class():
    """Import the downloaded upstream file with an in-memory RE2 tree adapter."""
    package_names = (
        "rca4tracing", "rca4tracing.datasources", "rca4tracing.datasources.logstore",
        "rca4tracing.rca", "rca4tracing.common",
    )
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    class LogServiceClient:
        def build_span_tree(self, trace):
            nodes, children, roots = {}, defaultdict(list), []
            for span in trace:
                sid = str(span["spanID"])
                nodes[sid] = {
                    "ServiceName": span["serviceName"],
                    "OperationName": span["rpc"],
                    "Duration": span["Duration"],
                    "TimeStamp": span["TimeStamp"],
                }
            for span in trace:
                sid, parent = str(span["spanID"]), str(span.get("parentSpanID", ""))
                if parent in nodes and parent != sid:
                    children[parent].append(sid)
                else:
                    roots.append(sid)
            for parent, child_ids in children.items():
                nodes[parent]["children"] = child_ids
            return roots, nodes

    log_client = types.ModuleType("rca4tracing.datasources.logstore.log_service_client")
    log_client.LogServiceClient = LogServiceClient
    sys.modules[log_client.__name__] = log_client
    utils = types.ModuleType("rca4tracing.rca.utils")
    utils.contribution_to_prob = lambda values: values
    sys.modules[utils.__name__] = utils
    logger = types.ModuleType("rca4tracing.common.logger")
    logger.setup_logger = lambda *args, **kwargs: logging.getLogger("shapleyiq")
    sys.modules[logger.__name__] = logger

    source = OUT / "third_party" / "shapley_value_rca.py"
    spec = importlib.util.spec_from_file_location("shapleyiq_upstream", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShapleyValueRCA


def parse_spans(rows):
    return [cpca.Span(sid, parent, service, operation, start, duration)
            for sid, parent, service, operation, start, duration in rows]


def completion_minute(spans):
    entry = next((span for span in spans if cpca.is_entry(span, DATASET)), None)
    end = entry.end if entry is not None else max(span.end for span in spans)
    return int(end // 60_000_000) * 60


def upstream_trace(rows):
    return [{"spanID": span.sid, "parentSpanID": span.parent,
             "serviceName": span.service, "rpc": span.operation,
             "Duration": int(span.duration), "TimeStamp": int(span.start)}
            for span in rows]


def evaluate_case(dataset, case_name, positive_minutes, cohort_case_rows, max_requests):
    archive_path = ARCHIVES[dataset]
    prefix = PREFIX[dataset]
    archive_case = case_name[len(prefix) + 1:] if case_name.startswith(prefix + "_") else case_name
    # Forecast labels flatten the final archive directory separator:
    # checkoutservice_delay/1 -> RE2-OB_checkoutservice_delay_1.
    if "_" in archive_case:
        fault, repetition = archive_case.rsplit("_", 1)
        if repetition.isdigit():
            archive_case = f"{fault}/{repetition}"
    member = f"{prefix}/{archive_case}"
    with zipfile.ZipFile(archive_path) as zf:
        inject_time = int(zf.read(f"{member}/inject_time.txt").decode().strip())
        traces = defaultdict(list)
        with zf.open(f"{member}/traces.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            needed = {"traceID", "spanID", "parentSpanID", "serviceName", "startTime", "duration", "operationName"}
            if not needed.issubset(reader.fieldnames or ()):
                raise ValueError(f"{member}/traces.csv missing fields {sorted(needed - set(reader.fieldnames or ())) }")
            for row in reader:
                try:
                    start, duration = int(row["startTime"]), float(row["duration"])
                except (TypeError, ValueError):
                    continue
                if duration < 0 or not math.isfinite(duration):
                    continue
                service = cpca.normalize_service(row["serviceName"])
                if service not in SERVICES[dataset]:
                    continue
                traces[row["traceID"]].append((
                    row["spanID"], row["parentSpanID"], service, row["operationName"],
                    float(start), duration,
                ))

    spans_by_trace = {trace_id: parse_spans(rows) for trace_id, rows in traces.items()}
    complete_minutes = [completion_minute(spans) for spans in spans_by_trace.values() if spans]
    case_start = min(complete_minutes) if complete_minutes else inject_time // 1_000_000
    warmup_cutoff = case_start + 10 * 60

    # Healthy reference: only complete, entry-rooted, valid requests in the
    # same fixed 10-minute warm-up used by the forecast protocol.
    durations = defaultdict(list)
    for spans in spans_by_trace.values():
        if not spans:
            continue
        minute = completion_minute(spans)
        if minute >= warmup_cutoff:
            continue
        component = cpca.entry_component(spans, dataset)
        if component is None:
            continue
        parsed = cpca.decompose_tree(component, dataset, clock_tolerance_us=1000.0)
        if parsed is None:
            continue
        by_id = parsed[0]
        for span in component:
            if span.sid in by_id:
                durations[(span.service, span.operation)].append(span.duration)
    normal_means = {f"{service}:{operation}": sum(values) / len(values)
                    for (service, operation), values in durations.items() if values}

    ShapleyValueRCA = load_upstream_class()
    raw_engine = ShapleyValueRCA(using_cache=False)
    warm_engine = ShapleyValueRCA(using_cache=False)
    raw_engine.merge_timelines = types.MethodType(fast_merge_timelines, raw_engine)
    warm_engine.merge_timelines = types.MethodType(fast_merge_timelines, warm_engine)
    # This hook only replaces the upstream Redis lookup. All timeline splitting,
    # synchronous merge and Shapley allocation remain the author implementation.
    warm_engine.get_normal_duration = lambda node_id: normal_means.get(node_id)

    scores = {"shapleyiq_default": {}, "shapleyiq_warmup10": {}}
    request_counts = defaultdict(int)
    request_trace_cache = []
    invalid = 0
    for trace_id, spans in spans_by_trace.items():
        if not spans:
            continue
        entry_spans = [span for span in spans if cpca.is_entry(span, dataset)]
        if len(entry_spans) != 1:
            continue
        entry = entry_spans[0]
        minute = int(entry.end // 60_000_000) * 60
        if minute not in positive_minutes or minute < warmup_cutoff:
            continue
        component = cpca.entry_component(spans, dataset)
        if component is None:
            invalid += 1
            continue
        parsed = cpca.decompose_tree(component, dataset, clock_tolerance_us=1000.0)
        if parsed is None:
            invalid += 1
            continue
        op_key = (entry.service, entry.operation)
        if request_counts[(minute, op_key)] >= max_requests:
            continue
        request_counts[(minute, op_key)] += 1
        request_trace_cache.append((minute, upstream_trace(component)))

    by_minute = defaultdict(list)
    for minute, trace in request_trace_cache:
        by_minute[minute].append(trace)
        for method, engine in (("shapleyiq_default", raw_engine), ("shapleyiq_warmup10", warm_engine)):
            contribution = engine.analyze_trace(trace)
            bucket = scores[method].setdefault(minute, defaultdict(float))
            for operation_id, value in contribution.items():
                service = operation_id.split(":", 1)[0]
                if service in SERVICES[dataset]:
                    bucket[service] += float(value)
    for method in scores:
        for minute, values in scores[method].items():
            count = len(by_minute[minute])
            scores[method][minute] = {service: value / count for service, value in values.items()}

    windows = []
    for minute in sorted(positive_minutes):
        method_scores = {method: score_map.get(minute, {}) for method, score_map in scores.items()}
        windows.append({"minute": minute, "scores_by_method": method_scores,
                        "trace_count": len(by_minute.get(minute, []))})
    return {"case_id": case_name, "injection_time": inject_time,
            "windows": windows, "diagnostics": {
                "raw_traces": len(spans_by_trace),
                "accepted_requests": sum(len(items) for items in by_minute.values()),
                "invalid_or_ambiguous_candidate_traces": invalid,
                "normal_operations_with_warmup_reference": len(normal_means),
            }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--limit-cases", type=int, default=0,
                        help="smoke-run only the first N positive cases")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    global DATASET
    DATASET = args.dataset
    cohort_path = VALIDATION / f"prediction_{args.dataset}_corrected_v2_seed42.json"
    cohort = json.loads(cohort_path.read_text())
    services = list(cohort["folds"][0]["services"])
    if services != list(SERVICES[args.dataset]):
        raise ValueError("Service universe differs from the fixed forecast cohort")
    positives = defaultdict(dict)
    for fold in cohort["folds"]:
        for row in fold["test_predictions"]:
            if row["failure_target"] and row["root"]:
                positives[row["case"]][int(row["minute"])] = row
    case_names = sorted(positives)
    if args.limit_cases:
        case_names = case_names[:args.limit_cases]
    output = args.output or OUT / "shapleyiq" / args.dataset / (
        f"smoke_{args.limit_cases}.json" if args.limit_cases else "results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    eval_cohort = cohort
    if args.limit_cases:
        chosen = set(case_names)
        eval_cohort = dict(cohort)
        eval_cohort["folds"] = [
            {**fold, "test_predictions": [row for row in fold["test_predictions"]
                                           if row["case"] in chosen]}
            for fold in cohort["folds"]
        ]
    report = {"dataset": args.dataset, "implementation_version": "author-core-with-re2-adapter-v1",
              "protocol": "fixed corrected-v2 forecast positive windows; first-10-minute causal normal means; max 100 valid requests per entry operation/minute; mean per trace; zero-fill/full-service evaluation",
              "source": {"repository": "https://github.com/lonyle/ShapleyIQ",
                        "file": "rca4tracing/rca/shapley_value_rca.py",
                        "sha256": hashlib.sha256((OUT / "third_party/shapley_value_rca.py").read_bytes()).hexdigest()},
              "config": {"max_requests": args.max_requests, "limit_cases": args.limit_cases,
                         "positive_cases": len(case_names),
                         "positive_windows": sum(len(positives[case]) for case in case_names)},
              "cases": {}}
    start = time.time()
    for index, case_name in enumerate(case_names, 1):
        print(f"[{args.dataset} {index}/{len(case_names)}] {case_name}", flush=True)
        report["cases"][case_name] = evaluate_case(
            args.dataset, case_name, set(positives[case_name]), positives[case_name], args.max_requests)
        report["elapsed_seconds"] = round(time.time() - start, 2)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  accepted={report['cases'][case_name]['diagnostics']['accepted_requests']} "
              f"invalid={report['cases'][case_name]['diagnostics']['invalid_or_ambiguous_candidate_traces']} "
              f"elapsed={report['elapsed_seconds']}s", flush=True)

    # Build the same explicit cohort audit contract used by the existing RCA evaluator.
    audit = {"implementation_version": report["implementation_version"], "cases": report["cases"]}
    for method in ("shapleyiq_default", "shapleyiq_warmup10"):
        evaluated = evaluate_attribution(audit, eval_cohort, method)
        report.setdefault("metrics", {})[method] = evaluated
    report["elapsed_seconds"] = round(time.time() - start, 2)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "elapsed_seconds": report["elapsed_seconds"],
                      "metrics": {k: v["event"] for k, v in report["metrics"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
