"""Build per-minute, per-entry-operation features for the OCSID predictor.

This is the data layer of OCSID-V1.  It streams each RE2 case's traces.csv,
groups spans by traceID (they are interleaved in the raw file), selects the
entry span with the dataset's label policy, and aggregates completed requests
by (minute, entry operation):

* request count;
* p50 / p95 / p99 / mean / max entry duration;
* total span count and error count;
* number of distinct services touched;
* the first-10-minute causal baseline for p95 and count.

Output: one CSV per dataset under artifacts/, plus a JSON summary.  The
predictor joins it with the panel's next-minute labels by input minute.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_ocsid_operation_features.py \
  --dataset tt --output artifacts/re2_tt_ocsid_operation_features_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


from .paths import ROOT
from ocsid.cpca import Span, is_entry, normalize_service  # noqa: E402

ARCHIVES = {
    "ob": ROOT / "data/RCAEval/RE2-OB.zip",
    "tt": ROOT / "data/RCAEval/RE2-TT.zip",
}
PREFIX = {"ob": "RE2-OB", "tt": "RE2-TT"}
FIELDS = [
    "dataset", "case_id", "minute_epoch", "input_minute_utc",
    "entry_service", "entry_operation",
    "count", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms",
    "span_count", "error_count", "n_services",
    "baseline_p95_ms", "baseline_count",
]


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _case_members(zf: zipfile.ZipFile, dataset: str) -> list[str]:
    prefix = PREFIX[dataset] + "/"
    members = []
    for name in zf.namelist():
        if name.endswith("/traces.csv") and name.startswith(prefix):
            members.append(name[len(prefix):-len("/traces.csv")])
    return sorted(members)


def _case_id(dataset: str, member: str) -> str:
    return f"{PREFIX[dataset]}_{member.replace('/', '_')}"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return float(ordered[max(math.ceil(q * len(ordered)) - 1, 0)])


def build_case(zf: zipfile.ZipFile, dataset: str, member: str) -> list[dict]:
    case_prefix = f"{PREFIX[dataset]}/{member}"
    with zf.open(f"{case_prefix}/traces.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        grouped: dict[str, list[tuple]] = defaultdict(list)
        trace_errors: dict[str, int] = defaultdict(int)
        for row in reader:
            try:
                start = int(row["startTime"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue
            if duration < 0 or not math.isfinite(duration):
                continue
            try:
                status = float(row.get("statusCode") or 0)
            except (TypeError, ValueError):
                status = 0.0
            if status != 0:
                trace_errors[row["traceID"]] += 1
            grouped[row["traceID"]].append((
                row["spanID"], row["parentSpanID"],
                normalize_service(row["serviceName"]), row["operationName"],
                float(start), duration,
            ))

    # (minute, entry_service, entry_operation) -> values
    buckets: dict[tuple[int, str, str], dict] = defaultdict(
        lambda: {"dur": [], "spans": 0, "errors": 0, "services": set()}
    )
    completions: list[int] = []
    for trace_id, rows in grouped.items():
        spans = [Span(*row) for row in rows]
        if not spans:
            continue
        # Prediction statistics require an entry span, not a serial replay.
        # Preserve the historical root selection even for unsupported RCA trees.
        by_id = {span.sid: span for span in spans}
        roots = [span for span in spans if span.parent not in by_id]
        if not roots:
            continue
        entries = [span for span in roots if is_entry(span, dataset)]
        root = (min(entries, key=lambda span: (span.start, -span.duration))
                if entries else max(roots, key=lambda span: span.duration))
        if not is_entry(root, dataset):
            continue
        completed = int(max(span.end for span in spans) // 60_000_000) * 60
        completions.append(completed)
        key = (completed, root.service, root.operation)
        item = buckets[key]
        item["dur"].append(root.duration / 1000.0)
        item["spans"] += len(spans)
        item["errors"] += trace_errors.get(trace_id, 0)
        item["services"].update(span.service for span in spans)

    if not completions:
        return []
    start = min(completions)
    cutoff = start + 600
    # Causal first-10-minute baseline per operation.
    baseline: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"p95": [], "count": []}
    )
    per_minute_count: dict[tuple[int, str, str], int] = defaultdict(int)
    for (minute, service, operation), item in buckets.items():
        per_minute_count[minute, service, operation] = len(item["dur"])
    for (minute, service, operation), count in per_minute_count.items():
        if minute < cutoff:
            baseline[(service, operation)]["p95"].append(
                _percentile(buckets[minute, service, operation]["dur"], 0.95)
            )
            baseline[(service, operation)]["count"].append(float(count))

    rows = []
    for (minute, service, operation), item in buckets.items():
        durations = item["dur"]
        if not durations:
            continue
        base = baseline.get((service, operation), {"p95": [1.0], "count": [1.0]})
        base_p95 = statistics.median(base["p95"]) if base["p95"] else 1.0
        base_count = statistics.median(base["count"]) if base["count"] else 1.0
        rows.append({
            "dataset": PREFIX[dataset],
            "case_id": _case_id(dataset, member),
            "minute_epoch": minute,
            "input_minute_utc": _iso(minute),
            "entry_service": service,
            "entry_operation": operation,
            "count": len(durations),
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "p99_ms": _percentile(durations, 0.99),
            "mean_ms": sum(durations) / len(durations),
            "max_ms": max(durations),
            "span_count": item["spans"],
            "error_count": item["errors"],
            "n_services": len(item["services"]),
            "baseline_p95_ms": max(base_p95, 1e-6),
            "baseline_count": max(base_count, 1e-6),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ARCHIVES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N cases")
    args = parser.parse_args(argv)

    with zipfile.ZipFile(ARCHIVES[args.dataset]) as zf:
        members = _case_members(zf, args.dataset)
        if args.limit:
            members = members[: args.limit]
        all_rows = []
        for index, member in enumerate(members, 1):
            rows = build_case(zf, args.dataset, member)
            all_rows.extend(rows)
            if index % 10 == 0 or index == len(members):
                print(f"[{args.dataset}] {index}/{len(members)} cases, {len(all_rows)} rows",
                      flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "dataset": args.dataset,
        "cases": len(members),
        "rows": len(all_rows),
        "operations": len({(r["entry_service"], r["entry_operation"]) for r in all_rows}),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
