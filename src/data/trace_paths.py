"""Request-level trace path utilities used by operation-conditioned RCA audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str
    service: str
    operation: str
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return max(self.end_us - self.start_us, 0.0)


def interval_union_length(
    intervals: Iterable[tuple[float, float]],
    lower: float,
    upper: float,
) -> float:
    """Length of the clipped interval union, including overlapping children once."""
    clipped = sorted(
        (max(start, lower), min(end, upper))
        for start, end in intervals
        if min(end, upper) > max(start, lower)
    )
    if not clipped:
        return 0.0
    total = 0.0
    left, right = clipped[0]
    for start, end in clipped[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def service_latencies(
    spans: Iterable[Span],
    scored_services: set[str],
) -> dict[str, tuple[float, float]]:
    """Return per-service (inclusive, exclusive) latency in microseconds.

    Exclusive latency subtracts the union of direct-child intervals from each
    parent.  Children from unscored infrastructure services still count when
    subtracting wait time, but only scored application services are returned.
    """
    unique = {(span.trace_id, span.span_id): span for span in spans}
    child_intervals: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for span in unique.values():
        parent = span.parent_span_id.strip()
        if not parent or parent.lower() in {"nan", "none", "null"}:
            continue
        child_intervals.setdefault((span.trace_id, parent), []).append(
            (span.start_us, span.end_us)
        )
    totals: dict[str, list[float]] = {}
    for key, span in unique.items():
        if span.service not in scored_services:
            continue
        inclusive = span.duration_us
        child_time = interval_union_length(
            child_intervals.get(key, ()), span.start_us, span.end_us
        )
        exclusive = max(inclusive - child_time, 0.0)
        pair = totals.setdefault(span.service, [0.0, 0.0])
        pair[0] += inclusive
        pair[1] += exclusive
    return {service: (values[0], values[1]) for service, values in totals.items()}


def entry_span(spans: Iterable[Span], dataset: str) -> Span | None:
    """Choose the observable entry span using the existing RE2 label policy."""
    candidates = []
    for span in spans:
        parent = span.parent_span_id.strip().lower()
        parent_blank = not parent or parent in {"nan", "none", "null"}
        if dataset == "ob":
            is_entry = span.service == "frontendservice" and span.operation == "frontend"
        elif dataset == "tt":
            is_entry = parent_blank and span.operation.startswith(
                ("GET ", "POST ", "PUT ", "DELETE ", "PATCH ")
            )
        else:
            raise ValueError(f"unsupported RE2 dataset: {dataset}")
        if is_entry:
            candidates.append(span)
    if not candidates:
        return None
    # The outermost request begins first.  Prefer the longer span only when
    # instrumentation emitted multiple roots with the same start timestamp.
    return min(candidates, key=lambda span: (span.start_us, -span.duration_us))

