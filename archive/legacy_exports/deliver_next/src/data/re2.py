"""RE2-OB telemetry adapter for the shared five-to-one-minute graph model."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.dataset import GraphWindowDataset


SERVICES = (
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontendservice", "paymentservice",
    "productcatalogservice", "recommendationservice", "redis", "shippingservice",
)
METRIC_NAMES = ("cpu", "mem", "diskio", "socket")
FEATURE_COLUMNS = (
    *(f"metric_{name}" for name in METRIC_NAMES),
    *(f"observed_{name}" for name in METRIC_NAMES),
    "trace_span_count", "trace_error_count", "trace_duration_mean_ms",
    "log_count", "log_error_count", "log_observed",
)


def _minute(epoch: float) -> int:
    return math.floor(epoch / 60) * 60


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _service(raw: str) -> str:
    name = raw.strip()
    if name in {"frontend", "frontend-external"}:
        return "frontendservice"
    return name


def _read_source_labels(path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[int, str]]:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    case_by_minute: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "input_minute_utc", "target_minute_utc",
                    "failure_next_minute", "root_cause_service", "entry_observed"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"source labels lack {sorted(required)}")
        for row in reader:
            case_id = row["case_id"]
            by_case[case_id].append(row)
            for key in ("input_minute_utc", "target_minute_utc"):
                epoch = int(datetime.fromisoformat(row[key]).timestamp())
                previous = case_by_minute.setdefault(epoch, case_id)
                if previous != case_id:
                    raise ValueError(f"case telemetry overlaps at {_iso(epoch)}")
    return by_case, case_by_minute


def _aggregate_metrics(archive: zipfile.ZipFile, case: str, minutes: set[int]):
    sums: dict[tuple[int, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    with archive.open(f"{case}/simple_metrics.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        columns = [
            (column, _service(column.rsplit("_", 1)[0]), column.rsplit("_", 1)[1])
            for column in (reader.fieldnames or ())
            if column.rsplit("_", 1)[-1] in METRIC_NAMES
        ]
        for row in reader:
            try:
                minute = _minute(float(row["time"]))
            except (ValueError, TypeError):
                continue
            if minute not in minutes:
                continue
            for column, service, metric in columns:
                if service not in SERVICES or not row[column]:
                    continue
                try:
                    value = float(row[column])
                except ValueError:
                    continue
                if math.isfinite(value):
                    pair = sums[minute, service, metric]
                    pair[0] += value
                    pair[1] += 1
    return sums


def _aggregate_logs(archive: zipfile.ZipFile, case: str, minutes: set[int]):
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    with archive.open(f"{case}/logs.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        needed = {"timestamp", "container_name", "level"}
        if not needed.issubset(reader.fieldnames or ()):
            raise ValueError(f"{case}/logs.csv lacks log fields")
        for row in reader:
            try:
                minute = _minute(int(row["timestamp"]) / 1_000_000_000)
            except (ValueError, TypeError):
                continue
            service = _service(row["container_name"])
            if minute not in minutes or service not in SERVICES:
                continue
            pair = counts[minute, service]
            pair[0] += 1
            pair[1] += row["level"].lower() in {"error", "fatal", "critical"}
    return counts


def _aggregate_traces(archive: zipfile.ZipFile, case: str, minutes: set[int]):
    observations: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    spans: dict[tuple[str, str], tuple[str, int]] = {}
    children: list[tuple[str, str, str, int]] = []
    with archive.open(f"{case}/traces.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        needed = {"traceID", "spanID", "parentSpanID", "serviceName",
                  "startTime", "duration", "statusCode"}
        if not needed.issubset(reader.fieldnames or ()):
            raise ValueError(f"{case}/traces.csv lacks span fields")
        for row in reader:
            service = _service(row["serviceName"])
            if service not in SERVICES:
                continue
            try:
                duration_us = float(row["duration"])
                completed = (int(row["startTime"]) + duration_us) / 1_000_000
            except (ValueError, TypeError):
                continue
            if not math.isfinite(duration_us) or duration_us < 0:
                continue
            minute = _minute(completed)
            if minute not in minutes:
                continue
            key = (row["traceID"], row["spanID"])
            spans[key] = (service, minute)
            status = row["statusCode"]
            try:
                error = bool(status and float(status) != 0)
            except ValueError:
                error = False
            values = observations[minute, service]
            values[0] += 1
            values[1] += error
            values[2] += duration_us / 1_000  # microseconds to milliseconds
            parent = row["parentSpanID"]
            if parent and parent.lower() not in {"nan", "none", "null"}:
                children.append((row["traceID"], parent, service, minute))
    adjacency: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((len(SERVICES), len(SERVICES)), dtype=np.float32)
    )
    service_index = {name: i for i, name in enumerate(SERVICES)}
    resolved = 0
    for trace_id, parent_id, child_service, child_minute in children:
        parent = spans.get((trace_id, parent_id))
        if parent is None:
            continue
        parent_service, parent_minute = parent
        if parent_service == child_service:
            continue
        # Both spans must have completed before the edge becomes observable.
        minute = max(parent_minute, child_minute)
        if minute in minutes:
            i, j = service_index[parent_service], service_index[child_service]
            adjacency[minute][i, j] += 1
            adjacency[minute][j, i] += 1
            resolved += 1
    return observations, adjacency, len(children), resolved


def prepare_re2_ob(
    archive_path: str | Path, source_label_csv: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Create a source-aligned service panel and dynamic call graph.

    All features are attached to their observation minute. Train-only feature
    scaling happens later in the shared training engine.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_files = [output / name for name in (
        "service_features_1m.csv", "next_minute_labels.csv", "case_minutes.csv",
        "service_adjacency_1m.npz", "feature_schema.json", "quality_report.json",
        "label_schema.json",
    )]
    if any(path.exists() for path in output_files):
        raise FileExistsError("RE2 panel output exists; use a fresh output directory")
    by_case, case_by_minute = _read_source_labels(Path(source_label_csv))
    features: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    graphs: dict[int, np.ndarray] = {}
    quality: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for number, (case_id, rows) in enumerate(sorted(by_case.items()), 1):
            # RE2-OB_<service>_<fault>_<rep> -> RE2-OB/<service>_<fault>/<rep>
            prefix, repetition = case_id.rsplit("_", 1)
            case = prefix.replace("RE2-OB_", "RE2-OB/", 1) + "/" + repetition
            if f"{case}/traces.csv" not in archive.namelist():
                raise ValueError(f"missing RE2 trace case {case}")
            minutes = {minute for minute, owner in case_by_minute.items() if owner == case_id}
            metrics = _aggregate_metrics(archive, case, minutes)
            logs = _aggregate_logs(archive, case, minutes)
            traces, edges, candidates, resolved = _aggregate_traces(archive, case, minutes)
            for (minute, service, metric), (total, count) in metrics.items():
                values = features[minute, service]
                values[f"metric_{metric}"] = total / count
                values[f"observed_{metric}"] = 1.0
            for (minute, service), (count, errors) in logs.items():
                values = features[minute, service]
                values["log_count"] = float(count)
                values["log_error_count"] = float(errors)
                values["log_observed"] = 1.0
            for (minute, service), (count, errors, duration_sum) in traces.items():
                values = features[minute, service]
                values["trace_span_count"] = count
                values["trace_error_count"] = errors
                values["trace_duration_mean_ms"] = duration_sum / count
            graphs.update(edges)
            quality[case_id] = {
                "minutes": len(minutes), "trace_parent_candidates": candidates,
                "resolved_cross_service_calls": resolved,
                "minutes_with_edges": sum(bool(matrix.any()) for matrix in edges.values()),
            }
            if number % 30 == 0:
                print(f"Prepared {number}/{len(by_case)} RE2-OB cases", flush=True)
    with output_files[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "service", "telemetry_valid", *FEATURE_COLUMNS])
        writer.writeheader()
        for minute in sorted(case_by_minute):
            for service in SERVICES:
                values = features[minute, service]
                valid = any(values.get(f"observed_{name}", 0) for name in METRIC_NAMES)
                valid |= bool(values.get("trace_span_count", 0) or values.get("log_observed", 0))
                writer.writerow({"timestamp_utc": _iso(minute), "service": service,
                                 "telemetry_valid": int(valid),
                                 **{key: values.get(key, 0.0) for key in FEATURE_COLUMNS}})
    with output_files[1].open("w", newline="", encoding="utf-8") as handle:
        fields = ["input_minute_utc", "target_minute_utc", "service",
                  "failure_next_minute", "root_cause_next_minute", "entry_observed",
                  "event_id", "root_label_valid"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rows in by_case.values():
            for row in rows:
                for service in SERVICES:
                    writer.writerow({
                        "input_minute_utc": row["input_minute_utc"],
                        "target_minute_utc": row["target_minute_utc"],
                        "service": service,
                        "failure_next_minute": row["failure_next_minute"],
                        "root_cause_next_minute": int(
                            row["failure_next_minute"] == "1"
                            and row["root_cause_service"] == service
                        ),
                        "entry_observed": row["entry_observed"],
                        "event_id": row["event_id"],
                        "root_label_valid": row["root_label_valid"],
                    })
    with output_files[2].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "case_id"])
        writer.writeheader()
        for minute, case_id in sorted(case_by_minute.items()):
            writer.writerow({"timestamp_utc": _iso(minute), "case_id": case_id})
    epochs = np.asarray(sorted(case_by_minute), dtype=np.int64)
    adjacency = np.stack([
        graphs.get(minute, np.zeros((len(SERVICES), len(SERVICES)), dtype=np.float32))
        for minute in epochs
    ])
    np.savez_compressed(output_files[3], timestamps=epochs,
                        services=np.asarray(SERVICES), adjacency=adjacency)
    output_files[4].write_text(json.dumps({
        "dataset": "RE2-OB", "services": SERVICES, "feature_columns": FEATURE_COLUMNS,
        "graph_policy": "child/parent completed before edge observation; symmetric counts",
        "source_labels": str(Path(source_label_csv).resolve()),
    }, indent=2), encoding="utf-8")
    report: dict[str, object] = {
        "cases": len(by_case), "minutes": len(case_by_minute),
        "service_minute_rows": len(case_by_minute) * len(SERVICES),
        "minutes_with_edges": int((adjacency.sum(axis=(1, 2)) > 0).sum()),
        "case_quality": quality,
    }
    output_files[5].write_text(json.dumps(report, indent=2), encoding="utf-8")
    source_schema = Path(source_label_csv).parent / "label_schema.json"
    if not source_schema.is_file():
        raise ValueError(f"missing RE2 label schema: {source_schema}")
    output_files[6].write_bytes(source_schema.read_bytes())
    return {key: value for key, value in report.items() if key != "case_quality"}


class Re2WindowDataset(GraphWindowDataset):
    """Shared graph windows with case-boundary and repetition-held-out splits."""

    def __init__(self, data_dir: str | Path, history: int = 5):
        directory = Path(data_dir)
        with (directory / "feature_schema.json").open() as handle:
            schema = json.load(handle)
        with np.load(directory / "service_adjacency_1m.npz", allow_pickle=False) as graph:
            graph_times = graph["timestamps"].copy()
            graph_services = tuple(graph["services"].tolist())
            adjacency = graph["adjacency"].copy()
        if graph_services != tuple(schema["services"]):
            raise ValueError("graph service order differs from features")
        super().__init__(
            directory / "service_features_1m.csv", directory / "next_minute_labels.csv",
            np.eye(len(graph_services), dtype=np.float32), graph_services,
            schema["feature_columns"], history=history, coverage_policy="available",
        )
        if not np.array_equal(graph_times, self.timestamp_epochs):
            raise ValueError("graph timestamps differ from service panel")
        self.adjacency = torch.from_numpy(adjacency.astype(np.float32))
        case_by_time: dict[str, str] = {}
        with (directory / "case_minutes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                case_by_time[row["timestamp_utc"]] = row["case_id"]
        self.case_ids = [case_by_time[stamp] for stamp in self.timestamps]
        self.sample_indices = [
            end for end in self.sample_indices
            if len(set(self.case_ids[end - history + 1:end + 1])) == 1
            and end + 1 < len(self.timestamps)
            and self.timestamp_epochs[end + 1] == self.timestamp_epochs[end] + 60
            and self.case_ids[end] == self.case_ids[end + 1]
        ]
        splits = {"train": [], "validation": [], "test": []}
        for end in self.sample_indices:
            repetition = self.case_ids[end].rsplit("_", 1)[-1]
            name = {"1": "train", "2": "validation", "3": "test"}.get(repetition)
            if name is None:
                raise ValueError(f"unexpected RE2 repetition {repetition}")
            splits[name].append(end)
        if any(not part for part in splits.values()):
            raise ValueError("RE2 repetition split has an empty partition")
        self.predefined_splits = tuple(splits.values())
        self.split_report = {
            "policy": "case_repetition_holdout", "train_repetition": 1,
            "validation_repetition": 2, "test_repetition": 3,
            "case_counts": {name: len({self.case_ids[end] for end in part})
                            for name, part in splits.items()},
        }
