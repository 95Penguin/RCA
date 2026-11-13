"""Auditable next-minute outcome labels for RCAEval RE2-OB cases.

The injection record supplies a candidate root and a reference time. A case is
positive only when completed frontend requests show sustained latency impact.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _utc(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _case_names(archive: zipfile.ZipFile) -> list[str]:
    cases = {
        name.rsplit("/", 1)[0]
        for name in archive.namelist()
        if name.endswith("/inject_time.txt") and name.split("/")[-2].isdigit()
    }
    if not cases:
        raise ValueError("no numbered RE2-OB cases with injection times were found")
    return sorted(cases)


def _frontend_minutes(archive: zipfile.ZipFile, case: str) -> dict[int, tuple[int, float]]:
    durations: dict[int, list[float]] = defaultdict(list)
    with archive.open(f"{case}/traces.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        needed = {"serviceName", "operationName", "startTime", "duration"}
        if not needed.issubset(reader.fieldnames or ()):
            raise ValueError(f"{case}/traces.csv is missing {sorted(needed)}")
        for row in reader:
            if row["serviceName"] != "frontendservice" or row["operationName"] != "frontend":
                continue
            try:
                duration_us = float(row["duration"])
                completed = (int(row["startTime"]) + duration_us) / 1_000_000
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid frontend span in {case}") from exc
            if not math.isfinite(duration_us) or duration_us < 0:
                raise ValueError(f"invalid frontend duration in {case}")
            durations[math.floor(completed / 60) * 60].append(duration_us / 1_000_000)
    return {minute: (len(values), _p95(values)) for minute, values in durations.items()}


def build_re2_ob_labels(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    min_entry_spans: int = 100,
    latency_ratio: float = 1.5,
    min_latency_increase_seconds: float = 0.1,
    consecutive_minutes: int = 2,
) -> dict[str, object]:
    """Export case-minute labels and events; require a clean 10-minute baseline.

    The minute containing injection is excluded. Baseline uses the ten complete
    pre-injection minutes, and labels use only completed requests. Positive
    events must start after injection and contain consecutive threshold minutes.
    """
    if min_entry_spans < 1 or latency_ratio <= 1 or min_latency_increase_seconds < 0:
        raise ValueError("invalid minimum count or latency threshold")
    if consecutive_minutes < 2:
        raise ValueError("consecutive_minutes must be at least 2")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_path = output / "next_minute_labels.csv"
    event_path = output / "events.csv"
    case_path = output / "cases.csv"
    schema_path = output / "label_schema.json"
    if any(path.exists() for path in (label_path, event_path, case_path, schema_path)):
        raise FileExistsError("label audit output exists; choose a fresh output directory")

    labels: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        cases = _case_names(archive)
        for case in cases:
            injection_epoch = int(archive.read(f"{case}/inject_time.txt").strip())
            injection_minute = (injection_epoch // 60) * 60
            post_start = injection_minute + 60
            evidence = {
                minute: value for minute, value in _frontend_minutes(archive, case).items()
                if injection_minute - 720 <= minute < injection_minute + 720
            }
            normal = [
                evidence[minute][1]
                for minute in range(injection_minute - 600, injection_minute, 60)
                if minute in evidence and evidence[minute][0] >= min_entry_spans
            ]
            if len(normal) < 5:
                raise ValueError(f"{case} has fewer than five observed baseline minutes")
            baseline = statistics.median(normal)
            threshold = max(
                latency_ratio * baseline, baseline + min_latency_increase_seconds
            )
            high = {
                minute for minute, (count, p95) in evidence.items()
                if minute >= post_start and count >= min_entry_spans and p95 >= threshold
            }
            runs: list[list[int]] = []
            for minute in sorted(high):
                if not runs or minute != runs[-1][-1] + 60:
                    runs.append([])
                runs[-1].append(minute)
            service_fault = case.split("/")[1]
            root_service, fault_type = service_fault.rsplit("_", 1)
            case_id = case.replace("/", "_")
            event_by_minute: dict[int, str] = {}
            for run in runs:
                if len(run) < consecutive_minutes:
                    continue
                event_id = f"{case_id}_event_{len(events) + 1}"
                events.append({
                    "event_id": event_id,
                    "case_id": case_id,
                    "root_service": root_service,
                    "fault_type": fault_type,
                    "injection_utc": _utc(injection_epoch),
                    "impact_start_utc": _utc(run[0]),
                    "impact_end_exclusive_utc": _utc(run[-1] + 60),
                    "duration_minutes": len(run),
                    "onset_lag_seconds": run[0] - injection_epoch,
                })
                event_by_minute.update({minute: event_id for minute in run})
            first, last = min(evidence), max(evidence)
            for target in range(first + 60, last + 60, 60):
                # This minute mixes pre- and post-injection requests.
                count, p95 = evidence.get(target, (0, 0.0))
                phase = ("mixed" if target == injection_minute else
                         "pre" if target < injection_minute else "post")
                event_id = event_by_minute.get(target, "")
                # A confirmed event needs no additional lookahead at its end.
                # An isolated high minute needs future evidence before it can
                # safely be called a negative under the two-minute rule.
                confirmation_observed = all(
                    evidence.get(target + step * 60, (0, 0.0))[0] >= min_entry_spans
                    for step in range(consecutive_minutes)
                )
                valid = (
                    phase != "mixed"
                    and count >= min_entry_spans
                    and (bool(event_id) or target not in high or confirmation_observed)
                )
                labels.append({
                    "case_id": case_id,
                    "input_minute_utc": _utc(target - 60),
                    "target_minute_utc": _utc(target),
                    "phase": phase,
                    "entry_span_count": count,
                    "entry_p95_latency_seconds": round(p95, 6),
                    "baseline_p95_latency_seconds": round(baseline, 6),
                    "latency_threshold_seconds": round(threshold, 6),
                    "entry_observed": int(valid),
                    "failure_next_minute": int(bool(event_id)),
                    "root_cause_service": root_service if event_id else "",
                    "root_label_valid": int(valid),
                    "event_id": event_id,
                })
            case_rows.append({
                "case_id": case_id, "root_service": root_service,
                "fault_type": fault_type, "injection_utc": _utc(injection_epoch),
                "baseline_minutes": len(normal),
                "baseline_p95_latency_seconds": round(baseline, 6),
                "latency_threshold_seconds": round(threshold, 6),
                "entry_minutes": len(evidence),
                "impact_events": sum(event["case_id"] == case_id for event in events),
                "impact_minutes": sum(minute in event_by_minute for minute in evidence),
            })

    for path, rows in ((label_path, labels), (event_path, events), (case_path, case_rows)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary: dict[str, object] = {
        "dataset": "RE2-OB", "case_count": len(case_rows),
        "cases_with_observed_impact": sum(bool(row["impact_events"]) for row in case_rows),
        "event_count": len(events),
        "positive_minutes": sum(int(row["failure_next_minute"]) for row in labels),
        "valid_target_minutes": sum(int(row["entry_observed"]) for row in labels),
        "valid_positive_minutes": sum(
            int(row["failure_next_minute"]) * int(row["entry_observed"])
            for row in labels
        ),
        "cases_by_fault_type": dict(Counter(row["fault_type"] for row in case_rows)),
        "impact_cases_by_fault_type": dict(Counter(
            row["fault_type"] for row in case_rows if row["impact_events"]
        )),
        "prediction_horizon_seconds": 60, "history_minutes": 5,
        "entry_service": "frontendservice", "entry_operation": "frontend",
        "timestamp_rule": "completed span minute; injection minute excluded",
        "baseline_rule": "median per-minute frontend p95 over 10 complete pre-injection minutes",
        "min_entry_spans": min_entry_spans,
        "latency_ratio": latency_ratio,
        "min_latency_increase_seconds": min_latency_increase_seconds,
        "consecutive_minutes": consecutive_minutes,
        "failure_target": "sustained observable frontend p95 latency degradation",
        "root_target": "injected service only for post-injection observed impact events",
        "split_unit": "whole injected case; no case may cross train/validation/test",
        "source_archive": str(Path(archive_path).resolve()),
    }
    schema_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
