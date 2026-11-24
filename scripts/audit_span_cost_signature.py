#!/usr/bin/env python3
"""Audit where an injected fault's latency actually lands in the span tree.

This is the minimal evidence probe behind CPCA (docs/2026-11-23-critical-path-
counterfactual-design.md, section 3.1.1).  It does not score services and does
not train anything.  For one case and one target service it compares, before vs
after injection:

* the target span's own duration;
* the gap immediately before the call (request path);
* the gap immediately after the call (return path);
* the caller's self time (duration minus the union of its direct children).

The point is to check the attribution rule: compute/IO faults should raise the
target's duration and self time, while delay faults should raise the target
duration and the caller's *post-gap*, not its pre-gap.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_span_cost_signature.py \
  --dataset tt --case ts-train-service_delay/1 --service ts-train-service
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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "ob": ROOT / "data/RCAEval/RE2-OB.zip",
    "tt": ROOT / "data/RCAEval/RE2-TT.zip",
}
PREFIX = {"ob": "RE2-OB", "tt": "RE2-TT"}


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Length of a union of half-open intervals."""
    total = 0.0
    current_end: float | None = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _read_spans(zf: zipfile.ZipFile, member: str):
    traces: dict[str, list[tuple[str, str, str, int, float]]] = defaultdict(list)
    with zf.open(member) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        for row in reader:
            try:
                start = int(row["startTime"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue
            if duration < 0 or not math.isfinite(duration):
                continue
            traces[row["traceID"]].append(
                (
                    row["spanID"],
                    row["parentSpanID"],
                    row["serviceName"].strip(),
                    start,
                    duration,
                )
            )
    return traces


def _collect(traces, target: str) -> tuple[list, list]:
    """Return edge samples and root-span samples touching ``target``."""
    edges: list[tuple[int, float, float, float, float]] = []
    roots: list[tuple[int, float, float]] = []
    for spans in traces.values():
        by_id = {span[0]: span for span in spans}
        for span in spans:
            if span[2] != target:
                continue
            parent = by_id.get(span[1])
            if parent is None:
                children = [c for c in spans if c[1] == span[0]]
                child_union = _union_length(
                    [(c[3], c[3] + c[4]) for c in children]
                )
                roots.append(
                    (
                        span[3] // 60_000_000 * 60,
                        span[4],
                        max(span[4] - child_union, 0.0),
                    )
                )
                continue
            if parent[2] == target:
                continue
            children = [c for c in spans if c[1] == parent[0]]
            intervals = [(c[3], c[3] + c[4]) for c in children]
            child_union = _union_length(intervals)
            start, end = span[3], span[3] + span[4]
            previous_end = parent[3]
            for child_start, child_end in sorted(intervals):
                if child_end <= start:
                    previous_end = max(previous_end, child_end)
            edges.append(
                (
                    start // 60_000_000 * 60,
                    span[4],
                    max(start - previous_end, 0.0),
                    max(parent[3] + parent[4] - end, 0.0),
                    max(parent[4] - child_union, 0.0),
                )
            )
    return edges, roots


def _split(samples, inject: int):
    before = [s for s in samples if s[0] < inject]
    after = [s for s in samples if s[0] >= inject + 60]
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ARCHIVES), required=True)
    parser.add_argument("--case", required=True, help="e.g. ts-train-service_delay/1")
    parser.add_argument("--service", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    prefix = PREFIX[args.dataset]
    member = f"{prefix}/{args.case.strip('/')}"
    with zipfile.ZipFile(ARCHIVES[args.dataset]) as zf:
        inject = int(zf.read(f"{member}/inject_time.txt").decode().strip())
        traces = _read_spans(zf, f"{member}/traces.csv")

    edges, roots = _collect(traces, args.service)
    edge_before, edge_after = _split(edges, inject)
    root_before, root_after = _split(roots, inject)

    def paired(before, after, indices):
        out = {"before_samples": len(before), "after_samples": len(after)}
        for name, index in indices.items():
            out[name] = {
                "before": _median([s[index] for s in before]),
                "after": _median([s[index] for s in after]),
            }
        return out

    report = {
        "dataset": args.dataset,
        "case": args.case,
        "service": args.service,
        "inject_epoch": inject,
        "traces": len(traces),
        "edge": paired(
            edge_before,
            edge_after,
            {
                "child_duration_us": 1,
                "pre_gap_us": 2,
                "post_gap_us": 3,
                "caller_self_us": 4,
            },
        ),
        "root_span": paired(
            root_before,
            root_after,
            {"duration_us": 1, "self_us": 2},
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
