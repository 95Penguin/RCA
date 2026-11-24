"""RE2-OB and RE2-TT telemetry adapters for the shared graph model."""

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

import numpy as np
import torch

from data.dataset import GraphWindowDataset


SERVICES = (
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontendservice", "paymentservice",
    "productcatalogservice", "recommendationservice", "redis", "shippingservice",
)
TT_SERVICES = (
    "ts-admin-basic-info-service", "ts-admin-travel-service", "ts-assurance-service",
    "ts-auth-service", "ts-basic-service", "ts-config-service",
    "ts-consign-price-service", "ts-consign-service", "ts-contacts-service",
    "ts-food-map-service", "ts-food-service", "ts-inside-payment-service",
    "ts-order-other-service", "ts-order-service", "ts-payment-service",
    "ts-preserve-other-service", "ts-preserve-service", "ts-price-service",
    "ts-route-service", "ts-seat-service", "ts-security-service",
    "ts-station-service", "ts-ticketinfo-service", "ts-train-service",
    "ts-travel-service", "ts-travel2-service", "ts-user-service",
)
METRIC_NAMES = ("cpu", "mem", "diskio", "socket")
FEATURE_COLUMNS = (
    *(f"metric_{name}" for name in METRIC_NAMES),
    *(f"observed_{name}" for name in METRIC_NAMES),
    "trace_span_count", "trace_error_count", "trace_duration_mean_ms",
    "log_count", "log_error_count", "log_observed",
    "entry_span_count", "entry_p95_ms", "observed_entry_p95",
    "observed_entry_baseline",
    "entry_degradation_score_rolling",
)
GRAPH_DIRECTIONS = ("symmetric", "callee-to-caller", "caller-to-callee")
GRAPH_ARRAY_KEYS = {
    "symmetric": "adjacency",
    # The attention layer aggregates columns into each row.  A caller-row
    # matrix therefore sends information from the callee to the caller.
    "callee-to-caller": "caller_row_adjacency",
    "caller-to-callee": "callee_row_adjacency",
}


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


def _aggregate_metrics(archive: zipfile.ZipFile, case: str, minutes: set[int],
                       services: tuple[str, ...]):
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
                if service not in services or not row[column]:
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


def _aggregate_logs(archive: zipfile.ZipFile, case: str, minutes: set[int],
                    services: tuple[str, ...], dataset_name: str):
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
            if minute not in minutes or service not in services:
                continue
            pair = counts[minute, service]
            pair[0] += 1
            error = row["level"].lower() in {"error", "fatal", "critical"}
            if dataset_name == "RE2-TT":
                message = row.get("message", "").upper()
                error |= " ERROR " in message or " EXCEPTION" in message or " FATAL " in message
            pair[1] += error
    return counts


def _aggregate_traces(archive: zipfile.ZipFile, case: str, minutes: set[int],
                      services: tuple[str, ...], dataset_name: str,
                      min_entry_spans: int, latency_ratio: float,
                      min_latency_increase_ms: float):
    observations: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    entry_durations: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    spans: dict[tuple[str, str], tuple[str, int]] = {}
    children: list[tuple[str, str, str, int]] = []
    with archive.open(f"{case}/traces.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        needed = {"traceID", "spanID", "parentSpanID", "serviceName",
                  "startTime", "duration", "statusCode", "operationName"}
        if not needed.issubset(reader.fieldnames or ()):
            raise ValueError(f"{case}/traces.csv lacks span fields")
        for row in reader:
            service = _service(row["serviceName"])
            if service not in services:
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
            operation = row["operationName"]
            parent_blank = not parent or parent.lower() in {"nan", "none", "null"}
            is_entry = (
                (dataset_name == "RE2-OB" and service == "frontendservice"
                 and operation == "frontend")
                or (dataset_name == "RE2-TT" and parent_blank
                    and operation.startswith(("GET ", "POST ", "PUT ", "DELETE ", "PATCH ")))
            )
            if is_entry:
                entry_durations[service, operation, minute].append(duration_us / 1_000)
            if parent and parent.lower() not in {"nan", "none", "null"}:
                children.append((row["traceID"], parent, service, minute))

    # The baseline for minute t uses only t-5 through t-1 within this case.
    # This does not use injection time, later spans, or target-minute labels.
    entry_features: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    entry_p95: dict[tuple[str, str, int], tuple[int, float]] = {}
    for key, durations in entry_durations.items():
        if len(durations) < min_entry_spans:
            continue
        ordered = sorted(durations)
        entry_p95[key] = (len(durations), ordered[math.ceil(0.95 * len(ordered)) - 1])
    for (service, operation, minute), (count, p95) in entry_p95.items():
        features = entry_features[minute, service]
        features["entry_span_count"] = features.get("entry_span_count", 0.0) + count
        features["entry_p95_ms"] = max(features.get("entry_p95_ms", 0.0), p95)
        features["observed_entry_p95"] = 1.0
        prior = [entry_p95[service, operation, minute - offset * 60][1]
                 for offset in range(1, 6)
                 if (service, operation, minute - offset * 60) in entry_p95]
        if len(prior) < 3:
            continue
        baseline = statistics.median(prior)
        score = min(
            p95 / max(latency_ratio * baseline, 1e-9),
            (p95 - baseline) / max(min_latency_increase_ms, 1e-9),
        )
        features["observed_entry_baseline"] = 1.0
        features["entry_degradation_score_rolling"] = max(
            features.get("entry_degradation_score_rolling", 0.0), max(score, 0.0)
        )
    caller_row_adjacency: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((len(services), len(services)), dtype=np.float32)
    )
    service_index = {name: i for i, name in enumerate(services)}
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
            caller_row_adjacency[minute][i, j] += 1
            resolved += 1
    return (
        observations,
        caller_row_adjacency,
        len(children),
        resolved,
        entry_features,
    )


def _prepare_re2(
    archive_path: str | Path, source_label_csv: str | Path, output_dir: str | Path,
    dataset_name: str, services: tuple[str, ...],
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
        "label_schema.json", "case_groups.csv",
    )]
    if any(path.exists() for path in output_files):
        raise FileExistsError("RE2 panel output exists; use a fresh output directory")
    by_case, case_by_minute = _read_source_labels(Path(source_label_csv))
    source_schema = Path(source_label_csv).parent / "label_schema.json"
    if not source_schema.is_file():
        raise ValueError(f"missing RE2 label schema: {source_schema}")
    label_definition = json.loads(source_schema.read_text())
    min_entry_spans = int(label_definition.get(
        "min_entry_spans", label_definition.get("min_operation_spans", 0)))
    if min_entry_spans < 1:
        raise ValueError("RE2 label schema lacks a positive entry count minimum")
    features: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    caller_row_graphs: dict[int, np.ndarray] = {}
    quality: dict[str, dict[str, int]] = {}
    case_signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for number, (case_id, rows) in enumerate(sorted(by_case.items()), 1):
            # RE2-XX_<service>_<fault>_<rep> -> RE2-XX/<service>_<fault>/<rep>
            prefix, repetition = case_id.rsplit("_", 1)
            case = prefix.replace(dataset_name + "_", dataset_name + "/", 1) + "/" + repetition
            if f"{case}/traces.csv" not in archive.namelist():
                raise ValueError(f"missing RE2 trace case {case}")
            minutes = {minute for minute, owner in case_by_minute.items() if owner == case_id}
            with archive.open(f"{case}/pod-node-1.csv") as raw:
                pods = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
            if not pods or not {"POD", "NODE_NAME"}.issubset(pods[0]):
                raise ValueError(f"missing pod/node snapshot in {case}")
            case_signatures[case_id] = (
                tuple(sorted(row["POD"] for row in pods)),
                tuple(sorted({row["NODE_NAME"] for row in pods})),
            )
            metrics = _aggregate_metrics(archive, case, minutes, services)
            logs = _aggregate_logs(archive, case, minutes, services, dataset_name)
            traces, directed_edges, candidates, resolved, entry = _aggregate_traces(
                archive, case, minutes, services, dataset_name, min_entry_spans,
                float(label_definition["latency_ratio"]),
                1000 * float(label_definition["min_latency_increase_seconds"]),
            )
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
            for (minute, service), entry_values in entry.items():
                features[minute, service].update(entry_values)
            caller_row_graphs.update(directed_edges)
            quality[case_id] = {
                "minutes": len(minutes), "trace_parent_candidates": candidates,
                "resolved_cross_service_calls": resolved,
                "minutes_with_edges": sum(
                    bool(matrix.any()) for matrix in directed_edges.values()
                ),
            }
            if number % 30 == 0:
                print(f"Prepared {number}/{len(by_case)} {dataset_name} cases", flush=True)
    with output_files[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "service", "telemetry_valid", *FEATURE_COLUMNS])
        writer.writeheader()
        for minute in sorted(case_by_minute):
            for service in services:
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
                for service in services:
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
    caller_row_adjacency = np.stack([
        caller_row_graphs.get(
            minute, np.zeros((len(services), len(services)), dtype=np.float32)
        )
        for minute in epochs
    ])
    callee_row_adjacency = caller_row_adjacency.transpose(0, 2, 1)
    adjacency = caller_row_adjacency + callee_row_adjacency
    np.savez_compressed(output_files[3], timestamps=epochs,
                        services=np.asarray(services), adjacency=adjacency,
                        caller_row_adjacency=caller_row_adjacency,
                        callee_row_adjacency=callee_row_adjacency)
    output_files[4].write_text(json.dumps({
        "dataset": dataset_name, "services": services, "feature_columns": FEATURE_COLUMNS,
        "graph_policy": (
            "completed parent/child calls; symmetric, caller-row and callee-row "
            "weighted adjacency arrays"
        ),
        "entry_feature_policy": (
            "completed entry spans at observation minute; root-operation p95; "
            "rolling baseline uses 3+ observed minutes among prior 5, within case"
        ),
        "source_labels": str(Path(source_label_csv).resolve()),
    }, indent=2), encoding="utf-8")
    report: dict[str, object] = {
        "cases": len(by_case), "minutes": len(case_by_minute),
        "service_minute_rows": len(case_by_minute) * len(services),
        "minutes_with_edges": int((adjacency.sum(axis=(1, 2)) > 0).sum()),
        "case_quality": quality,
    }
    output_files[5].write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_files[6].write_bytes(source_schema.read_bytes())
    deployment_signatures = sorted(set(case_signatures.values()))
    session_signatures = sorted({signature[1] for signature in case_signatures.values()})
    deployment_ids = {signature: f"{dataset_name}-deployment-{index + 1:02d}"
                      for index, signature in enumerate(deployment_signatures)}
    session_ids = {signature: f"{dataset_name}-session-{index + 1:02d}"
                   for index, signature in enumerate(session_signatures)}
    with output_files[7].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "deployment_id", "session_id"])
        writer.writeheader()
        for case_id, signature in sorted(case_signatures.items()):
            writer.writerow({"case_id": case_id,
                             "deployment_id": deployment_ids[signature],
                             "session_id": session_ids[signature[1]]})
    report["deployment_groups"] = len(deployment_signatures)
    report["session_groups"] = len(session_signatures)
    return {key: value for key, value in report.items() if key != "case_quality"}


def prepare_re2_ob(
    archive_path: str | Path, source_label_csv: str | Path, output_dir: str | Path
) -> dict[str, object]:
    return _prepare_re2(archive_path, source_label_csv, output_dir, "RE2-OB", SERVICES)


def prepare_re2_tt(
    archive_path: str | Path, source_label_csv: str | Path, output_dir: str | Path
) -> dict[str, object]:
    return _prepare_re2(archive_path, source_label_csv, output_dir, "RE2-TT", TT_SERVICES)


class Re2WindowDataset(GraphWindowDataset):
    """Point-in-time failure state forecasts with case-safe data splits.

    A horizon of ``h`` predicts the application state and root service at the
    exact target minute ``t + h``. The default ``h=1`` preserves the original
    next-minute task.
    """

    def __init__(self, data_dir: str | Path, history: int = 5,
                 split_policy: str = "repetition", split_fold: int = 0,
                 horizon_minutes: int = 1,
                 graph_direction: str = "symmetric"):
        if horizon_minutes < 1:
            raise ValueError("horizon_minutes must be positive")
        if split_policy not in {"repetition", "configuration", "deployment", "session"}:
            raise ValueError("split_policy must be repetition, configuration, deployment or session")
        if graph_direction not in GRAPH_DIRECTIONS:
            raise ValueError(
                f"graph_direction must be one of {', '.join(GRAPH_DIRECTIONS)}"
            )
        if not 0 <= split_fold <= 4 or (split_policy == "repetition" and split_fold):
            raise ValueError("split_fold must be 0-4 for grouped splits, 0 for repetition")
        directory = Path(data_dir)
        with (directory / "feature_schema.json").open() as handle:
            schema = json.load(handle)
        with np.load(directory / "service_adjacency_1m.npz", allow_pickle=False) as graph:
            graph_times = graph["timestamps"].copy()
            graph_services = tuple(graph["services"].tolist())
            graph_key = GRAPH_ARRAY_KEYS[graph_direction]
            if graph_key not in graph.files:
                raise ValueError(
                    f"graph artifact lacks {graph_key!r} for {graph_direction!r}"
                )
            adjacency = graph[graph_key].copy()
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
        self.graph_direction = graph_direction
        case_by_time: dict[str, str] = {}
        with (directory / "case_minutes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                case_by_time[row["timestamp_utc"]] = row["case_id"]
        self.case_ids = [case_by_time[stamp] for stamp in self.timestamps]
        self.fault_type_names = tuple(sorted({
            case_id.rsplit("_", 2)[-2] for case_id in self.case_ids
        }))
        fault_type_index = {
            name: index for index, name in enumerate(self.fault_type_names)
        }
        self.fault_type_targets = np.asarray([
            fault_type_index[case_id.rsplit("_", 2)[-2]]
            for case_id in self.case_ids
        ], dtype=np.int64)
        self.sample_indices = [
            end for end in self.sample_indices
            if len(set(self.case_ids[end - history + 1:end + 1])) == 1
            and end + 1 < len(self.timestamps)
            and self.timestamp_epochs[end + 1] == self.timestamp_epochs[end] + 60
            and self.case_ids[end] == self.case_ids[end + 1]
        ]
        self._apply_point_horizon(horizon_minutes)
        splits = {"train": [], "validation": [], "test": []}
        grouped: dict[str, list[str]] = {}
        group_by_case: dict[str, str] = {}
        group_fold: dict[str, int] = {}
        if split_policy == "configuration":
            configurations = sorted({case.rsplit("_", 1)[0] for case in self.case_ids})
            for config in configurations:
                fault_type = config.rsplit("_", 1)[-1]
                grouped.setdefault(fault_type, []).append(config)
            if any(len(configs) != 5 for configs in grouped.values()):
                raise ValueError("configuration split requires five services per fault type")
            config_split = {}
            for fault_index, fault_type in enumerate(sorted(grouped)):
                for index, config in enumerate(grouped[fault_type]):
                    config_split[config] = (
                        "validation" if index == (fault_index + split_fold) % 5 else
                        "test" if index == (fault_index + split_fold + 2) % 5 else "train"
                    )
        elif split_policy in {"deployment", "session"}:
            group_file = directory / "case_groups.csv"
            if not group_file.is_file():
                raise ValueError(f"{split_policy} split requires {group_file}")
            field = f"{split_policy}_id"
            with group_file.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    group_by_case[row["case_id"]] = row[field]
            if any(case not in group_by_case for case in self.case_ids):
                raise ValueError(f"{split_policy} group metadata misses a panel case")
            # Balance by raw case count, without consulting labels or the
            # number of target-valid windows in a case.
            group_sizes = Counter(group_by_case[case] for case in set(self.case_ids))
            if len(group_sizes) < 5:
                raise ValueError(f"{split_policy} split needs at least five groups")
            fold_sizes = [0] * 5
            for group_id, size in sorted(group_sizes.items(), key=lambda item: (-item[1], item[0])):
                fold = min(range(5), key=lambda index: (fold_sizes[index], index))
                group_fold[group_id] = fold
                fold_sizes[fold] += size
        for end in self.sample_indices:
            case_id = self.case_ids[end]
            if split_policy == "configuration":
                name = config_split[case_id.rsplit("_", 1)[0]]
            elif split_policy in {"deployment", "session"}:
                fold = group_fold[group_by_case[case_id]]
                name = ("test" if fold == split_fold else
                        "validation" if fold == (split_fold + 1) % 5 else "train")
            else:
                repetition = case_id.rsplit("_", 1)[-1]
                name = {"1": "train", "2": "validation", "3": "test"}.get(repetition)
                if name is None:
                    raise ValueError(f"unexpected RE2 repetition {repetition}")
            splits[name].append(end)
        if any(not part for part in splits.values()):
            raise ValueError("RE2 split has an empty partition")
        self.predefined_splits = tuple(splits.values())
        self.split_report = {
            "policy": ("case_repetition_holdout" if split_policy == "repetition" else
                       f"{split_policy}_holdout"),
            "case_counts": {name: len({self.case_ids[end] for end in part})
                            for name, part in splits.items()},
        }
        if split_policy == "configuration":
            self.split_report["configuration_counts"] = {
                name: len({self.case_ids[end].rsplit("_", 1)[0] for end in part})
                for name, part in splits.items()
            }
            self.split_report["fold"] = split_fold
            self.split_report["allocation"] = (
                "within each fault type, sorted service configurations: "
                "validation index=(fault_index+fold) mod 5; "
                "test index=(fault_index+fold+2) mod 5"
            )
        elif split_policy in {"deployment", "session"}:
            self.split_report.update({
                "fold": split_fold,
                "group_field": f"{split_policy}_id",
                "group_counts": {name: len({group_by_case[self.case_ids[end]] for end in part})
                                 for name, part in splits.items()},
                "group_assignment": group_fold,
                "allocation": "greedy balance by raw case count; test fold=N, validation fold=(N+1)%5",
            })
        else:
            self.split_report.update({"train_repetition": 1,
                                      "validation_repetition": 2,
                                      "test_repetition": 3})
        self.split_report.update({
            "task": "failure_state",
            "horizon_minutes": horizon_minutes,
            "target_semantics": f"exact state at t+{horizon_minutes}m",
        })

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """Add case fault type for positive-window auxiliary supervision only."""
        sample = super().__getitem__(item)
        end = self.sample_indices[item]
        sample["fault_type_target"] = torch.tensor(
            self.fault_type_targets[end], dtype=torch.long
        )
        sample["fault_type_valid"] = torch.tensor(bool(self.failure_targets[end]))
        return sample

    def _apply_point_horizon(self, horizon_minutes: int) -> None:
        """Shift next-minute panel labels to an exact future target minute."""
        self.forecast_horizon_minutes = horizon_minutes
        self.forecast_mode = "point"
        if horizon_minutes == 1:
            self.forecast_end_timestamps = {
                end: self.label_target_times[self.timestamps[end]]
                for end in self.sample_indices
            }
            return

        index_by_epoch = {epoch: index for index, epoch in enumerate(self.timestamp_epochs)}
        original_failure = self.failure_targets.copy()
        original_nodes = self.node_targets.copy()
        original_events = list(self.event_ids)
        original_root_valid = dict(self._root_label_valid)
        valid_ends: list[int] = []
        forecast_ends: dict[int, str] = {}
        source_offset = (horizon_minutes - 1) * self.bucket_seconds
        target_offset = horizon_minutes * self.bucket_seconds
        for end in self.sample_indices:
            decision_epoch = self.timestamp_epochs[end]
            source = index_by_epoch.get(decision_epoch + source_offset)
            if source is None:
                continue
            source_timestamp = self.timestamps[source]
            target_timestamp = self.label_target_times.get(source_timestamp)
            if target_timestamp is None or not self._label_valid.get(source_timestamp, True):
                continue
            target_epoch = self._parse_timestamp(target_timestamp)
            target = index_by_epoch.get(target_epoch)
            if target is None or target_epoch - decision_epoch != target_offset:
                continue
            if not (
                self.case_ids[end] == self.case_ids[source] == self.case_ids[target]
            ):
                continue

            decision_timestamp = self.timestamps[end]
            self.failure_targets[end] = original_failure[source]
            self.node_targets[end] = original_nodes[source]
            self.event_ids[end] = original_events[source]
            self._root_label_valid[decision_timestamp] = original_root_valid.get(
                source_timestamp,
                not bool(original_failure[source]) or bool(original_nodes[source].any()),
            )
            valid_ends.append(end)
            forecast_ends[end] = target_timestamp
        self.sample_indices = valid_ends
        self.forecast_end_timestamps = forecast_ends


class Re2FirstVisibleWindowDataset(Re2WindowDataset):
    """LTFP-style first user-visible failure target on the audited RE2 panel.

    Each physical case has at most one failure reference time: the first target
    minute whose audited business-degradation label is positive. Inputs at or
    after that reference time are excluded. Pre-injection inputs are explicit
    healthy controls; post-injection inputs are positive iff the first visible
    failure lies in ``(t, t + horizon]``. Cases without an observed failure are
    censored and retained only when the complete horizon is observable.
    """

    def __init__(
        self,
        data_dir: str | Path,
        history: int = 5,
        split_policy: str = "repetition",
        split_fold: int = 0,
        horizon_minutes: int = 1,
        graph_direction: str = "symmetric",
    ) -> None:
        if horizon_minutes < 1:
            raise ValueError("horizon_minutes must be positive")
        super().__init__(
            data_dir,
            history,
            split_policy,
            split_fold,
            horizon_minutes=1,
            graph_direction=graph_direction,
        )
        directory = Path(data_dir)
        schema = json.loads((directory / "feature_schema.json").read_text())
        source_path = Path(schema["source_labels"])
        if not source_path.is_file():
            raise ValueError(f"missing source labels for first-visible task: {source_path}")

        source_rows: dict[tuple[str, str], dict[str, str]] = {}
        onset_by_case: dict[str, int] = {}
        censor_by_case: dict[str, int] = {}
        root_by_case: dict[str, str] = {}
        root_valid_by_case: dict[str, bool] = {}
        with source_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {
                "case_id", "input_minute_utc", "target_minute_utc", "phase",
                "failure_next_minute", "root_cause_service", "root_label_valid",
            }
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("source labels lack fields required by first-visible protocol")
            for row in reader:
                case_id = row["case_id"]
                source_rows[case_id, row["input_minute_utc"]] = row
                target_epoch = self._parse_timestamp(row["target_minute_utc"])
                censor_by_case[case_id] = max(censor_by_case.get(case_id, target_epoch), target_epoch)
                if row["failure_next_minute"] == "1":
                    onset_by_case[case_id] = min(
                        onset_by_case.get(case_id, target_epoch), target_epoch
                    )
                    root = row["root_cause_service"]
                    if root:
                        previous = root_by_case.setdefault(case_id, root)
                        if previous != root:
                            raise ValueError(f"root service changes inside case {case_id}")
                    root_valid_by_case[case_id] = row["root_label_valid"] == "1"

        service_index = {service: index for index, service in enumerate(self.services)}
        horizon_seconds = horizon_minutes * 60
        original_splits = self.predefined_splits
        valid_ends: list[int] = []
        positives_by_split: dict[str, int] = {}
        self.failure_targets[:] = 0
        self.node_targets[:] = 0
        self.event_ids = ["" for _ in self.event_ids]
        for end in list(self.sample_indices):
            case_id = self.case_ids[end]
            timestamp = self.timestamps[end]
            decision_epoch = self.timestamp_epochs[end]
            source = source_rows.get((case_id, timestamp))
            if source is None:
                continue
            onset = onset_by_case.get(case_id)
            if onset is not None and decision_epoch >= onset:
                continue  # strictly no post-failure inputs
            if onset is None and decision_epoch + horizon_seconds > censor_by_case[case_id]:
                continue  # incomplete negative forecast horizon

            post_injection = source["phase"] == "post"
            positive = bool(
                post_injection
                and onset is not None
                and 0 < onset - decision_epoch <= horizon_seconds
            )
            self.failure_targets[end] = int(positive)
            if positive:
                root = root_by_case.get(case_id)
                if root in service_index:
                    self.node_targets[end, service_index[root]] = 1.0
                self.event_ids[end] = f"{case_id}_first_visible"
                self._root_label_valid[timestamp] = bool(
                    root_valid_by_case.get(case_id, False) and root in service_index
                )
            else:
                self._root_label_valid[timestamp] = True
            valid_ends.append(end)

        valid_set = set(valid_ends)
        self.sample_indices = valid_ends
        self.predefined_splits = tuple(
            [end for end in split if end in valid_set] for split in original_splits
        )
        if any(not split for split in self.predefined_splits):
            raise ValueError("first-visible protocol produced an empty split")
        for name, split in zip(("train", "validation", "test"), self.predefined_splits):
            positives_by_split[name] = int(self.failure_targets[split].sum())
            if positives_by_split[name] == 0:
                raise ValueError(f"first-visible {name} split has no positive windows")
        self.split_report = {
            **self.split_report,
            "task": "first_user_visible_failure",
            "horizon_minutes": horizon_minutes,
            "target_semantics": f"first failure onset within (t, t+{horizon_minutes}m]",
            "failure_reference": "first audited business-degradation target minute per case",
            "post_failure_inputs": "excluded",
            "pre_injection_policy": "healthy control",
            "positive_windows": positives_by_split,
            "cases_with_failure_reference": len(onset_by_case),
        }
        self.forecast_horizon_minutes = horizon_minutes
        self.forecast_mode = "window"
        self.forecast_end_timestamps = {
            end: datetime.fromtimestamp(
                self.timestamp_epochs[end] + horizon_seconds,
                tz=timezone.utc,
            ).isoformat()
            for end in valid_ends
        }
        self.first_visible_horizon_minutes = horizon_minutes
