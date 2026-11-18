from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import infer_service_name


TIME_COLUMN = "time"
DEFAULT_INTERVAL_SECONDS = 15

_SYSTEM_NAMES = {
    "OB": "online-boutique",
    "SS": "sock-shop",
    "TT": "train-ticket",
}


@dataclass(frozen=True)
class TelemetryCase:
    """One telemetry episode with stable identity and event-time metadata.

    ``name`` is retained for callers written against the original interface.
    RCAEval loaders set it to the globally stable ``case_id`` so existing
    case-level splitting code automatically stops grouping unrelated repeat
    directories named only ``1``, ``2``, etc.

    ``incident_time`` remains the backwards-compatible target-time field.
    New code should use ``failure_time`` for an observed degradation onset and
    ``inject_time`` for the intervention timestamp.  When no onset manifest is
    supplied, the RCAEval loader keeps the historical behavior and exposes the
    injection timestamp through ``incident_time``.
    """

    name: str
    frame: pd.DataFrame
    incident_time: float | None = None
    root_label: str | None = None
    fault_type: str | None = None
    metadata: dict | None = None
    case_id: str | None = None
    scenario_id: str | None = None
    system: str | None = None
    inject_time: float | None = None
    failure_time: float | None = None
    censor_time: float | None = None
    topology: dict | None = None

    def __post_init__(self) -> None:
        case_id = str(self.case_id or self.name)
        scenario_id = self.scenario_id or case_id.rpartition("/")[0] or case_id
        inject_time = self.inject_time if self.inject_time is not None else self.incident_time
        incident_time = self.incident_time
        if incident_time is None:
            if self.failure_time is not None:
                incident_time = self.failure_time
            elif self.censor_time is None:
                # No onset/censor label is attached, so preserve the legacy
                # convention that incident_time denotes the injection proxy.
                incident_time = inject_time

        topology = dict(self.topology or {})
        metadata = dict(self.metadata or {})
        metadata.setdefault("case_id", case_id)
        metadata.setdefault("scenario_id", scenario_id)
        if self.system is not None:
            metadata.setdefault("system", self.system)
        if inject_time is not None:
            metadata.setdefault("inject_time", float(inject_time))
        if self.failure_time is not None:
            metadata.setdefault("failure_time", float(self.failure_time))
        if self.censor_time is not None:
            metadata.setdefault("censor_time", float(self.censor_time))
        if topology:
            metadata.setdefault("topology", topology)
            if topology.get("edges"):
                metadata.setdefault("edges", topology["edges"])

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "inject_time", inject_time)
        object.__setattr__(self, "incident_time", incident_time)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "topology", topology)

    @property
    def event_observed(self) -> bool:
        return self.failure_time is not None

    @property
    def target_time(self) -> float | None:
        """Return the onset target when available, else the legacy target."""

        return self.failure_time if self.failure_time is not None else self.incident_time


@dataclass(frozen=True)
class WindowDataset:
    X: np.ndarray
    Y: np.ndarray
    horizons: tuple[float, ...]
    feature_names: tuple[str, ...]
    raw_metric_names: tuple[str, ...]
    case_names: tuple[str, ...]
    end_times: np.ndarray
    lead_times: np.ndarray
    root_labels: tuple[str | None, ...]
    fault_types: tuple[str | None, ...]

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)


def read_metric_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return clean_metric_frame(frame)


def clean_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "time.1" in frame.columns:
        frame = frame.drop(columns=["time.1"])
    if TIME_COLUMN not in frame.columns:
        frame.insert(0, TIME_COLUMN, np.arange(len(frame), dtype=float))
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # Preserve latency quantile identity.  Renaming p90 to a generic latency
    # name can create duplicate columns and makes a registered p95 -> p90 ->
    # generic SLI preference impossible to audit.
    frame[TIME_COLUMN] = pd.to_numeric(frame[TIME_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN).reset_index(drop=True)

    numeric_cols = [TIME_COLUMN]
    for col in frame.columns:
        if col == TIME_COLUMN:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            frame[col] = values
            numeric_cols.append(col)
    frame = frame[numeric_cols]
    value_cols = [col for col in numeric_cols if col != TIME_COLUMN]
    if value_cols:
        # Sorting before forward fill is essential: filling in source-file
        # order can copy a later observation into an earlier timestamp.
        frame[value_cols] = frame[value_cols].ffill().fillna(0)
    return frame.reset_index(drop=True)


def read_metrics_json(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    frame = _json_to_frame(obj)
    return clean_metric_frame(frame)


def _json_to_frame(obj) -> pd.DataFrame:
    if isinstance(obj, list):
        return pd.json_normalize(obj)

    if not isinstance(obj, dict):
        raise ValueError("metrics.json must contain a JSON object or list")

    if "metrics" in obj:
        return _json_to_frame(obj["metrics"])

    if "data" in obj:
        return _json_to_frame(obj["data"])

    series_by_name = {}
    time_values = None
    for key, value in obj.items():
        if key in {TIME_COLUMN, "timestamp", "timestamps"}:
            time_values = value
            continue
        if isinstance(value, dict):
            index = list(value.keys())
            vals = list(value.values())
            if time_values is None:
                time_values = index
            series_by_name[key] = vals
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                nested = pd.json_normalize(value)
                for nested_col in nested.columns:
                    series_by_name[f"{key}_{nested_col}"] = nested[nested_col].to_numpy()
            else:
                series_by_name[key] = value

    if series_by_name:
        min_len = min(len(v) for v in series_by_name.values())
        data = {name: np.asarray(values[:min_len]) for name, values in series_by_name.items()}
        if time_values is not None:
            data[TIME_COLUMN] = np.asarray(time_values[:min_len])
        return pd.DataFrame(data)

    return pd.json_normalize(obj)


def parse_case_name(case_dir: Path) -> tuple[str | None, str | None]:
    if case_dir.name.isdigit():
        return parse_case_name(case_dir.parent)
    name = case_dir.name
    parts = name.split("_")
    # Some exports encode the repetition in the scenario directory itself,
    # e.g. ``ts-route-service_f3_1`` rather than a numeric leaf directory.
    if len(parts) >= 3 and parts[-1].isdigit():
        name = "_".join(parts[:-1])
    if "_" in name:
        root_label, fault_type = name.rsplit("_", 1)
        return root_label or None, fault_type or None
    return None, None


def infer_case_identity(case_dir: str | Path, dataset_root: str | Path | None = None) -> tuple[str, str, str | None]:
    """Infer stable case/scenario identifiers and the benchmark system.

    RCAEval leaf directories are repeat numbers, so using only
    ``case_path.name`` aliases many unrelated fault scenarios.  IDs are rooted
    at the nearest dataset directory such as ``RE2-SS`` whenever possible.
    """

    case_path = Path(case_dir)
    dataset_index = None
    dataset_id = None
    system = None
    for idx, part in enumerate(case_path.parts):
        match = re.fullmatch(r"RE\d+-([A-Za-z]+)", part, flags=re.IGNORECASE)
        if match:
            dataset_index = idx
            dataset_id = part
            system = _SYSTEM_NAMES.get(match.group(1).upper(), match.group(1).lower())

    if dataset_index is not None:
        relative_parts = case_path.parts[dataset_index:]
    elif dataset_root is not None:
        root_path = Path(dataset_root)
        try:
            relative_parts = (root_path.name, *case_path.relative_to(root_path).parts)
        except ValueError:
            relative_parts = (case_path.name,)
    else:
        relative_parts = (case_path.name,)

    case_id = "/".join(str(part) for part in relative_parts)
    if len(relative_parts) > 1:
        scenario_id = "/".join(str(part) for part in relative_parts[:-1])
    else:
        scenario_id = case_id
    if dataset_id is None and dataset_root is not None:
        root_name = Path(dataset_root).name
        match = re.fullmatch(r"RE\d+-([A-Za-z]+)", root_name, flags=re.IGNORECASE)
        if match:
            system = _SYSTEM_NAMES.get(match.group(1).upper(), match.group(1).lower())
    return case_id, scenario_id, system


def _normalise_topology(payload) -> dict:
    if not payload:
        return {}
    directed = False
    raw_edges = payload
    raw_nodes = []
    if isinstance(payload, Mapping):
        directed = bool(payload.get("directed", False))
        raw_edges = payload.get("edges", payload.get("links", payload.get("dependencies", [])))
        raw_nodes = payload.get("nodes", payload.get("services", []))

    edges: set[tuple[str, str]] = set()
    for edge in raw_edges if isinstance(raw_edges, Sequence) and not isinstance(raw_edges, (str, bytes)) else []:
        if isinstance(edge, Mapping):
            source = edge.get("source", edge.get("src", edge.get("from")))
            target = edge.get("target", edge.get("dst", edge.get("to")))
        elif isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) >= 2:
            source, target = edge[0], edge[1]
        else:
            continue
        if source is None or target is None:
            continue
        edges.add((str(source), str(target)))

    nodes = {str(node.get("id", node.get("name"))) if isinstance(node, Mapping) else str(node) for node in raw_nodes}
    nodes.discard("None")
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)
    return {
        "directed": directed,
        "nodes": sorted(nodes),
        "edges": [list(edge) for edge in sorted(edges)],
    }


def load_topology_metadata(case_dir: str | Path) -> dict:
    """Load optional explicit service topology without inferring causal edges.

    Supported files contain either an edge list or a mapping with ``edges`` /
    ``links`` / ``dependencies``.  Correlation edges are deliberately not
    labeled as topology here.
    """

    case_path = Path(case_dir)
    search_roots = [case_path]
    search_roots.extend(list(case_path.parents[:3]))
    for root in search_roots:
        for filename in ("topology.json", "service_graph.json", "dependencies.json"):
            path = root / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            topology = _normalise_topology(payload)
            if topology:
                topology["source"] = str(path)
                return topology
    return {}


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _manifest_records(manifest: str | Path | Mapping | None) -> dict[str, Mapping]:
    if manifest is None:
        return {}
    if isinstance(manifest, (str, Path)):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    else:
        payload = manifest
    if not isinstance(payload, Mapping):
        raise ValueError("label manifest must be a mapping")
    cases = payload.get("cases", payload)
    if isinstance(cases, Mapping):
        return {str(case_id): record for case_id, record in cases.items() if isinstance(record, Mapping)}
    if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
        records: dict[str, Mapping] = {}
        for record in cases:
            if not isinstance(record, Mapping) or record.get("case_id") is None:
                continue
            case_id = str(record["case_id"])
            if case_id in records:
                raise ValueError(f"duplicate case_id in label manifest: {case_id}")
            records[case_id] = record
        return records
    raise ValueError("label manifest cases must be a list or mapping")


def load_rcaeval_case(
    case_dir: str | Path,
    dataset_root: str | Path | None = None,
    label_record: Mapping | None = None,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    include_auxiliary: bool = True,
    prefer_simple_metrics: bool = False,
) -> TelemetryCase:
    case_path = Path(case_dir)
    if label_record is not None:
        audit_status = str(label_record.get("audit_status") or "")
        status = str(label_record.get("status") or "")
        if audit_status.startswith("candidate_") or status == "pending_review":
            raise ValueError(
                f"case {label_record.get('case_id', case_path)!r} has only a pending-review onset candidate; "
                "final reviewed labels are required for dataset loading"
            )
        final_audit_statuses = {
            "accepted_onset",
            "censored_no_onset",
            "censored_missing_telemetry",
            "censored_ambiguous_timing",
        }
        if audit_status not in final_audit_statuses:
            raise ValueError(
                f"case {label_record.get('case_id', case_path)!r} has unsupported final audit_status "
                f"{audit_status!r}"
            )
        if label_record.get("review_status") not in (None, "complete"):
            raise ValueError(f"case {label_record.get('case_id', case_path)!r} has incomplete review")
    if prefer_simple_metrics:
        metric_candidates = (
            case_path / "simple_data.csv",
            case_path / "simple_metrics.csv",
            case_path / "metrics.json",
            case_path / "data.csv",
            case_path / "metrics.csv",
        )
    else:
        metric_candidates = (
            case_path / "metrics.json",
            case_path / "data.csv",
            case_path / "simple_data.csv",
            case_path / "simple_metrics.csv",
            case_path / "metrics.csv",
        )
    metric_path = next((path for path in metric_candidates if path.exists()), None)
    if metric_path is None:
        raise FileNotFoundError(f"missing RCAEval metric file in {case_path}")
    if metric_path.suffix == ".json":
        frame = read_metrics_json(metric_path)
    else:
        frame = read_metric_csv(metric_path)

    modalities = {"metrics"}
    for filename, prefix, raw_filename, raw_mode, modality_name in (
        ("logts.csv", "log", "logs.csv", "log", "logs"),
        ("tracets_err.csv", "trace_err", "traces.csv", "err", "traces"),
        ("tracets_lat.csv", "trace_lat", "traces.csv", "lat", "traces"),
    ):
        if not include_auxiliary:
            break
        aux_path = case_path / filename
        raw_path = case_path / raw_filename
        # Prefer raw events so aggregation timestamps can unambiguously denote
        # the end of a completed right-closed bucket.  Pre-aggregated RCAEval
        # files use left-edge timestamps and are shifted to their availability
        # time when raw events are unavailable.
        if raw_path.exists() and raw_filename == "logs.csv":
            aux = aggregate_raw_logs(raw_path, interval_seconds=interval_seconds)
        elif raw_path.exists() and raw_filename == "traces.csv":
            aux = aggregate_raw_traces(raw_path, mode=raw_mode, interval_seconds=interval_seconds)
        elif aux_path.exists():
            aux = read_interval_aggregate_csv(aux_path, interval_seconds=interval_seconds)
        else:
            continue
        if aux.empty:
            continue
        aux = prefix_aux_columns(aux, prefix)
        frame = merge_causal_time(frame, aux)
        modalities.add(modality_name)

    inject_time = None
    inject_path = case_path / "inject_time.txt"
    if inject_path.exists():
        text = inject_path.read_text(encoding="utf-8").strip()
        inject_time = float(text)
    manifest_inject_time = _optional_float(label_record.get("inject_time")) if label_record is not None else None
    if inject_time is not None and manifest_inject_time is not None and not math.isclose(
        inject_time, manifest_inject_time, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(
            f"inject_time.txt and final label manifest disagree for {case_path}: "
            f"{inject_time} != {manifest_inject_time}"
        )
    if manifest_inject_time is not None:
        inject_time = manifest_inject_time

    failure_time = _optional_float(label_record.get("failure_time")) if label_record is not None else None
    censor_time = _optional_float(label_record.get("censor_time")) if label_record is not None else None
    if label_record is not None:
        if inject_time is None or censor_time is None:
            raise ValueError(f"final label record for {case_path} needs inject_time and censor_time")
        if censor_time < inject_time:
            raise ValueError(f"final label record for {case_path} has censor_time before inject_time")
        if failure_time is not None and not (inject_time < failure_time <= censor_time):
            raise ValueError(
                f"final label record for {case_path} must satisfy inject_time < failure_time <= censor_time"
            )
        audit_status = str(label_record.get("audit_status"))
        if audit_status == "accepted_onset" and failure_time is None:
            raise ValueError(f"accepted onset record for {case_path} has no failure_time")
        if audit_status.startswith("censored_") and failure_time is not None:
            raise ValueError(f"censored record for {case_path} unexpectedly sets failure_time")
    # A supplied onset manifest changes the legacy target to the inferred
    # degradation onset.  Censored cases intentionally have no incident time.
    incident_time = failure_time if label_record is not None else inject_time

    root_label, fault_type = parse_case_name(case_path)
    case_id, scenario_id, system = infer_case_identity(case_path, dataset_root=dataset_root)
    if label_record is not None:
        if label_record.get("case_id") is not None and str(label_record["case_id"]) != case_id:
            raise ValueError(f"label case_id does not match physical path: {label_record['case_id']!r} != {case_id!r}")
        if label_record.get("scenario_id") is not None and str(label_record["scenario_id"]) != scenario_id:
            raise ValueError(
                f"label scenario_id does not match physical path: {label_record['scenario_id']!r} != {scenario_id!r}"
            )
        if label_record.get("system") is not None and str(label_record["system"]) != str(system):
            raise ValueError(f"label system does not match physical path: {label_record['system']!r} != {system!r}")
    topology = load_topology_metadata(case_path)
    if label_record is not None and isinstance(label_record.get("topology"), Mapping):
        topology = _normalise_topology(label_record["topology"])
    metadata = {
        "case_dir": str(case_path),
        "metric_path": str(metric_path),
        "modalities": sorted(modalities),
        "legacy_name": case_path.name,
        "inject_time": inject_time,
        "failure_time": failure_time,
        "censor_time": censor_time,
    }
    if label_record is not None:
        metadata["label_version"] = label_record.get("label_version")
        metadata["label_status"] = label_record.get("status")
        metadata["audit_status"] = label_record.get("audit_status")
        metadata["review_status"] = label_record.get("review_status")
        metadata["adjudicated"] = bool(label_record.get("adjudicated", False))
        metadata["sli"] = label_record.get("sli", [])
    return TelemetryCase(
        name=case_id,
        frame=frame,
        incident_time=incident_time,
        root_label=root_label,
        fault_type=fault_type,
        metadata=metadata,
        case_id=case_id,
        scenario_id=scenario_id,
        system=system,
        inject_time=inject_time,
        failure_time=failure_time,
        censor_time=censor_time,
        topology=topology,
    )


def discover_rcaeval_cases(root: str | Path) -> list[Path]:
    root_path = Path(root)
    case_dirs = {path.parent for path in root_path.rglob("metrics.json")}
    for filename in ("data.csv", "simple_data.csv", "simple_metrics.csv", "metrics.csv"):
        case_dirs.update(path.parent for path in root_path.rglob(filename))
    return sorted(case_dirs)


def load_rcaeval_cases(
    root: str | Path,
    limit: int | None = None,
    label_manifest: str | Path | Mapping | None = None,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    include_auxiliary: bool = True,
    prefer_simple_metrics: bool = False,
) -> list[TelemetryCase]:
    case_dirs = discover_rcaeval_cases(root)
    if limit is not None:
        case_dirs = case_dirs[:limit]
    records = _manifest_records(label_manifest)
    cases = []
    for path in case_dirs:
        case_id, _, _ = infer_case_identity(path, dataset_root=root)
        if label_manifest is not None and case_id not in records:
            raise KeyError(
                f"label manifest has no record for discovered case {case_id!r}; "
                "refusing to mix failure-onset targets with injection-time proxies"
            )
        cases.append(
            load_rcaeval_case(
                path,
                dataset_root=root,
                label_record=records.get(case_id) if label_manifest is not None else None,
                interval_seconds=interval_seconds,
                include_auxiliary=include_auxiliary,
                prefer_simple_metrics=prefer_simple_metrics,
            )
        )
    return cases


def load_sample_csv_case(
    csv_path: str | Path,
    incident_time: float | None,
    root_label: str | None = None,
    fault_type: str | None = None,
) -> TelemetryCase:
    frame = read_metric_csv(csv_path)
    case_id = Path(csv_path).stem
    return TelemetryCase(
        name=case_id,
        frame=frame,
        incident_time=incident_time,
        root_label=root_label,
        fault_type=fault_type,
        metadata={"csv_path": str(csv_path)},
        case_id=case_id,
        scenario_id=case_id,
        inject_time=incident_time,
    )


def load_multisource_sample_case(
    sample_dir: str | Path,
    incident_time: float | None = None,
    root_label: str | None = None,
    fault_type: str | None = None,
) -> TelemetryCase:
    sample_path = Path(sample_dir)
    metrics_path = sample_path / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv in {sample_path}")
    frame = read_metric_csv(metrics_path)

    for filename, prefix in (
        ("logts.csv", "log"),
        ("tracets_lat.csv", "trace_lat"),
        ("tracets_err.csv", "trace_err"),
    ):
        aux_path = sample_path / filename
        if not aux_path.exists():
            continue
        aux = prefix_aux_columns(read_interval_aggregate_csv(aux_path), prefix)
        frame = merge_causal_time(frame, aux)

    if incident_time is None:
        inject_path = sample_path / "inject_time.txt"
        if inject_path.exists():
            incident_time = float(inject_path.read_text(encoding="utf-8").strip())

    case_id = sample_path.name
    return TelemetryCase(
        name=case_id,
        frame=clean_metric_frame(frame),
        incident_time=incident_time,
        root_label=root_label,
        fault_type=fault_type,
        metadata={"sample_dir": str(sample_path), "modalities": ["metrics", "logs", "traces"]},
        case_id=case_id,
        scenario_id=case_id,
        inject_time=incident_time,
    )


def read_interval_aggregate_csv(
    path: str | Path,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    timestamps_are_left_edges: bool = True,
) -> pd.DataFrame:
    """Read an interval aggregate and timestamp it when the bucket is known.

    RCAEval's ``logts`` / ``tracets`` files use the left edge of a 15-second
    aggregation interval.  Adding the interval moves each record to the right
    edge, preventing the aggregate from becoming visible before all events in
    that interval have occurred.
    """

    frame = read_metric_csv(path)
    if timestamps_are_left_edges and not frame.empty:
        frame = frame.copy()
        frame[TIME_COLUMN] = frame[TIME_COLUMN] + float(interval_seconds)
    return frame


def merge_causal_time(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance_seconds: float | None = None,
) -> pd.DataFrame:
    """Backward-align completed auxiliary observations to a primary timeline.

    Every joined right-hand timestamp is guaranteed to satisfy
    ``right.time <= left.time``.  Missing/stale auxiliary values are set to
    zero instead of being copied from a future observation.
    """

    left = left.sort_values(TIME_COLUMN)
    right = right.sort_values(TIME_COLUMN)
    left = left.copy()
    right = right.copy()
    left[TIME_COLUMN] = pd.to_numeric(left[TIME_COLUMN], errors="coerce").astype(float)
    right[TIME_COLUMN] = pd.to_numeric(right[TIME_COLUMN], errors="coerce").astype(float)
    tolerance = tolerance_seconds
    if tolerance is None:
        tolerance = max(infer_step_seconds(left), infer_step_seconds(right)) * 2
    merged = pd.merge_asof(
        left,
        right,
        on=TIME_COLUMN,
        direction="backward",
        tolerance=tolerance,
    )
    return merged.fillna(0)


def merge_nearest_time(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the now-causal alignment operation."""

    return merge_causal_time(left, right)


def prefix_aux_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Preserve service as the leading token while marking auxiliary modality.

    `log_frontend_1` would be grouped as service `log`; `frontend_log_frontend_1`
    stays grouped under service `frontend`, which is what the graph model needs.
    """

    return frame.rename(
        columns={
            col: f"{infer_service_name(str(col))}_{prefix}_{col}"
            for col in frame.columns
            if col != TIME_COLUMN
        }
    )


def _event_seconds(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame:
        values = pd.to_numeric(frame["timestamp"], errors="coerce")
        if values.notna().any():
            median = values.dropna().median()
            if median > 1e15:
                return values / 1e9
            if median > 1e12:
                return values / 1e3
            return values
    if "startTimeMillis" in frame:
        return pd.to_numeric(frame["startTimeMillis"], errors="coerce") / 1e3
    if "startTime" in frame:
        values = pd.to_numeric(frame["startTime"], errors="coerce")
        if values.notna().any():
            return values / 1e6 if values.dropna().median() > 1e14 else values
    return pd.Series(np.arange(len(frame), dtype=float), index=frame.index)


def right_closed_bin_end(values, interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> np.ndarray:
    """Map timestamps to right-closed buckets ``(end-interval, end]``."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    seconds = np.asarray(values, dtype=float)
    return np.ceil(seconds / float(interval_seconds)) * float(interval_seconds)


def right_closed_resample(
    frame: pd.DataFrame,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    agg: str = "mean",
    fill_empty: float | None = None,
) -> pd.DataFrame:
    """Causally resample numeric telemetry onto right-edge timestamps.

    A row stamped ``t`` summarizes only source rows in ``(t-I, t]`` where
    ``I`` is ``interval_seconds``.  Empty buckets remain missing unless
    ``fill_empty`` is supplied; no backward fill is ever performed.
    """

    clean = clean_metric_frame(frame)
    value_cols = [column for column in clean.columns if column != TIME_COLUMN]
    if clean.empty or not value_cols:
        return pd.DataFrame(columns=[TIME_COLUMN, *value_cols])
    work = clean.copy()
    work["_bin_end"] = right_closed_bin_end(work[TIME_COLUMN], interval_seconds)
    grouped = work.groupby("_bin_end", sort=True)[value_cols].agg(agg)
    first_end = float(grouped.index.min())
    last_end = float(grouped.index.max())
    full_index = np.arange(first_end, last_end + interval_seconds * 0.5, interval_seconds, dtype=float)
    grouped = grouped.reindex(full_index)
    if fill_empty is not None:
        grouped = grouped.fillna(float(fill_empty))
    grouped.insert(0, TIME_COLUMN, grouped.index.astype(float))
    return grouped.reset_index(drop=True)


def _pivot_by_service(
    frame: pd.DataFrame,
    service_col: str,
    value_col: str,
    aggfunc,
    interval_seconds: int = 15,
) -> pd.DataFrame:
    if frame.empty or service_col not in frame:
        return pd.DataFrame()
    work = frame.copy()
    work["_seconds"] = _event_seconds(work)
    work = work.dropna(subset=["_seconds"])
    if work.empty:
        return pd.DataFrame()
    work["_bin"] = right_closed_bin_end(work["_seconds"], interval_seconds)
    # Hyphens are part of canonical service identities (for example
    # ``ts-route-service`` and ``front-end``).  Replacing them with underscores
    # makes downstream ``infer_service_name`` truncate Train Ticket services
    # to the ambiguous token ``ts``.
    work["_service"] = work[service_col].astype(str)
    pivot = work.pivot_table(index="_bin", columns="_service", values=value_col, aggfunc=aggfunc, fill_value=0)
    pivot = pivot.sort_index()
    if len(pivot.index) > 0:
        full_index = np.arange(
            float(pivot.index.min()),
            float(pivot.index.max()) + interval_seconds * 0.5,
            interval_seconds,
            dtype=float,
        )
        pivot = pivot.reindex(full_index, fill_value=0)
    pivot.columns = [f"{col}_1" for col in pivot.columns]
    pivot.insert(0, TIME_COLUMN, pivot.index.astype(float))
    return clean_metric_frame(pivot.reset_index(drop=True))


def aggregate_raw_logs(path: str | Path, interval_seconds: int = 15) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    service_col = "container_name" if "container_name" in frame else "serviceName" if "serviceName" in frame else None
    if service_col is None:
        return pd.DataFrame()
    frame["_count"] = 1
    return _pivot_by_service(frame, service_col, "_count", "sum", interval_seconds=interval_seconds)


def aggregate_raw_traces(path: str | Path, mode: str, interval_seconds: int = 15) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or "serviceName" not in frame:
        return pd.DataFrame()
    if mode == "err":
        status = pd.to_numeric(frame.get("statusCode", 0), errors="coerce").fillna(0)
        frame["_value"] = (status != 0).astype(float)
        return _pivot_by_service(frame, "serviceName", "_value", "sum", interval_seconds=interval_seconds)
    if mode == "lat":
        frame["_value"] = pd.to_numeric(frame.get("duration", 0), errors="coerce").fillna(0)
        return _pivot_by_service(frame, "serviceName", "_value", "mean", interval_seconds=interval_seconds)
    raise ValueError(f"unknown trace aggregation mode: {mode}")


def infer_step_seconds(frame: pd.DataFrame) -> float:
    times = frame[TIME_COLUMN].to_numpy(dtype=float)
    if len(times) < 2:
        return 1.0
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def build_window_dataset(
    cases: Sequence[TelemetryCase],
    horizons: Sequence[float],
    window_size: int = 20,
    stride: int = 1,
    pre_incident_only: bool = True,
    max_negative_per_case: int | None = None,
) -> WindowDataset:
    rows: list[np.ndarray] = []
    labels: list[list[int]] = []
    case_names: list[str] = []
    end_times: list[float] = []
    lead_times: list[float] = []
    root_labels: list[str | None] = []
    fault_types: list[str | None] = []
    feature_names: tuple[str, ...] | None = None
    raw_metric_names: tuple[str, ...] | None = None

    for case in cases:
        frame = clean_metric_frame(case.frame)
        case_metric_cols = [col for col in frame.columns if col != TIME_COLUMN]
        if raw_metric_names is None:
            raw_metric_names = tuple(case_metric_cols)
        metric_cols = list(raw_metric_names or ())
        if not metric_cols or len(frame) < window_size:
            continue
        values = frame.reindex(columns=metric_cols, fill_value=0.0).to_numpy(dtype=float)
        times = frame[TIME_COLUMN].to_numpy(dtype=float)
        case_negative_count = 0
        target_time = case.failure_time if case.failure_time is not None else case.incident_time

        for end_idx in range(window_size - 1, len(frame), stride):
            end_time = float(times[end_idx])
            if case.failure_time is None and case.censor_time is not None and end_time > case.censor_time:
                continue
            lead = math.inf
            if target_time is not None:
                lead = float(target_time - end_time)
                if pre_incident_only and lead <= 0:
                    continue

            y = [int(0 < lead <= horizon) for horizon in horizons]
            if max(y) == 0 and max_negative_per_case is not None:
                if case_negative_count >= max_negative_per_case:
                    continue
                case_negative_count += 1

            window = values[end_idx - window_size + 1 : end_idx + 1]
            x, names = window_to_features(window, metric_cols)
            if feature_names is None:
                feature_names = tuple(names)
            rows.append(x)
            labels.append(y)
            case_names.append(case.case_id or case.name)
            end_times.append(end_time)
            lead_times.append(lead)
            root_labels.append(case.root_label)
            fault_types.append(case.fault_type)

    if not rows:
        raise ValueError("no windows were generated; check data paths, incident times, and window size")

    return WindowDataset(
        X=np.vstack(rows),
        Y=np.asarray(labels, dtype=int),
        horizons=tuple(float(h) for h in horizons),
        feature_names=feature_names or tuple(),
        raw_metric_names=raw_metric_names or tuple(),
        case_names=tuple(case_names),
        end_times=np.asarray(end_times, dtype=float),
        lead_times=np.asarray(lead_times, dtype=float),
        root_labels=tuple(root_labels),
        fault_types=tuple(fault_types),
    )


def window_to_features(window: np.ndarray, metric_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    last = window[-1]
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    min_v = window.min(axis=0)
    max_v = window.max(axis=0)
    first = window[0]
    slope = (last - first) / max(len(window) - 1, 1)
    delta_mean = last - mean
    blocks = [
        ("last", last),
        ("mean", mean),
        ("std", std),
        ("min", min_v),
        ("max", max_v),
        ("slope", slope),
        ("delta_mean", delta_mean),
    ]
    values = []
    names = []
    for stat_name, stat_values in blocks:
        values.append(stat_values)
        names.extend([f"{metric}__{stat_name}" for metric in metric_names])
    return np.concatenate(values).astype(float), names


def split_by_case(
    dataset: WindowDataset,
    test_ratio: float = 0.3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_cases = np.asarray(sorted(set(dataset.case_names)))
    rng.shuffle(unique_cases)
    n_test = max(1, int(round(len(unique_cases) * test_ratio)))
    test_cases = set(unique_cases[:n_test])
    test_idx = np.asarray([name in test_cases for name in dataset.case_names])
    train_idx = ~test_idx
    if train_idx.sum() == 0 or test_idx.sum() == 0:
        return split_stratified(dataset.Y[:, -1], test_ratio=test_ratio, seed=seed)
    return np.flatnonzero(train_idx), np.flatnonzero(test_idx)


def split_stratified(y: np.ndarray, test_ratio: float = 0.3, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train_parts = []
    test_parts = []
    for value in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == value)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio))) if len(idx) > 1 else 0
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def iter_batches(indices: Iterable[int], batch_size: int) -> Iterable[np.ndarray]:
    batch = []
    for idx in indices:
        batch.append(idx)
        if len(batch) == batch_size:
            yield np.asarray(batch, dtype=int)
            batch = []
    if batch:
        yield np.asarray(batch, dtype=int)
