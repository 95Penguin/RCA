"""Train-fitted tensorization for fault-to-failure experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import TIME_COLUMN, TelemetryCase, clean_metric_frame, right_closed_resample
from .features import infer_service_name
from .protocol import TopologyManifest


MODALITIES = ("metric", "log", "trace")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    service: str
    modality: str
    slot: int


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    services: tuple[str, ...]
    modality_names: tuple[str, ...]
    modality_dims: tuple[int, ...]
    features: tuple[FeatureSpec, ...]
    max_features: int
    fitted_case_ids: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return int(sum(self.modality_dims))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["input_dim"] = self.input_dim
        return payload


@dataclass(frozen=True)
class FailureWindowDataset:
    X: np.ndarray
    Y: np.ndarray
    label_mask: np.ndarray
    availability_mask: np.ndarray
    feature_availability_mask: np.ndarray
    horizons: tuple[float, ...]
    case_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    systems: tuple[str, ...]
    end_times: np.ndarray
    inject_times: np.ndarray
    failure_times: np.ndarray
    censor_times: np.ndarray
    lead_times: np.ndarray
    root_targets: np.ndarray
    root_active: np.ndarray
    root_labels: tuple[str | None, ...]
    schema: FeatureSchema

    def subset(self, indices: np.ndarray) -> "FailureWindowDataset":
        idx = np.asarray(indices, dtype=int)
        return FailureWindowDataset(
            X=self.X[idx],
            Y=self.Y[idx],
            label_mask=self.label_mask[idx],
            availability_mask=self.availability_mask[idx],
            feature_availability_mask=self.feature_availability_mask[idx],
            horizons=self.horizons,
            case_ids=tuple(np.asarray(self.case_ids, dtype=object)[idx]),
            scenario_ids=tuple(np.asarray(self.scenario_ids, dtype=object)[idx]),
            systems=tuple(np.asarray(self.systems, dtype=object)[idx]),
            end_times=self.end_times[idx],
            inject_times=self.inject_times[idx],
            failure_times=self.failure_times[idx],
            censor_times=self.censor_times[idx],
            lead_times=self.lead_times[idx],
            root_targets=self.root_targets[idx],
            root_active=self.root_active[idx],
            root_labels=tuple(np.asarray(self.root_labels, dtype=object)[idx]),
            schema=self.schema,
        )


def infer_feature_modality(name: str) -> str:
    value = str(name)
    if "_trace_err_" in value or "_trace_lat_" in value:
        return "trace"
    if "_log_" in value:
        return "log"
    return "metric"


def _normalize(value: str) -> str:
    return str(value).strip().lower().replace("_", "-").replace(".", "-").replace("/", "-")


def resolve_service(name: str, topology: TopologyManifest) -> str | None:
    """Map a telemetry column to one explicit topology node."""

    inferred = infer_service_name(str(name))
    candidates = [inferred, str(name).split("__", 1)[0]]
    alias_norm = {_normalize(key): value for key, value in topology.aliases.items()}
    service_norm = {_normalize(service): service for service in topology.services}
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized in alias_norm:
            return alias_norm[normalized]
        if normalized in service_norm:
            return service_norm[normalized]
    # Prefer the longest service prefix so ``ts-travel2-service`` does not map
    # to ``ts-travel-service``.
    normalized_name = _normalize(str(name))
    matches = [service for key, service in service_norm.items() if normalized_name.startswith(key + "-")]
    if matches:
        return max(matches, key=len)
    return None


def fit_feature_schema(
    train_cases: Sequence[TelemetryCase],
    topology: TopologyManifest,
    max_features: int = 180,
    excluded_features: Sequence[str] = (),
    allowed_modalities: Sequence[str] = MODALITIES,
) -> FeatureSchema:
    """Fit a service/modality-balanced feature schema on train cases only."""

    excluded = {str(item) for item in excluded_features}
    allowed = {str(item) for item in allowed_modalities}
    unknown_modalities = allowed - set(MODALITIES)
    if unknown_modalities:
        raise ValueError(f"unknown modalities: {sorted(unknown_modalities)}")
    seen: set[str] = set()
    grouped: dict[tuple[str, str], list[str]] = {}
    for case in sorted(train_cases, key=lambda item: str(item.case_id)):
        for column in clean_metric_frame(case.frame).columns:
            name = str(column)
            if name == TIME_COLUMN or name in seen or name in excluded:
                continue
            service = resolve_service(name, topology)
            if service is None:
                continue
            modality = infer_feature_modality(name)
            if modality not in allowed:
                continue
            seen.add(name)
            grouped.setdefault((service, modality), []).append(name)

    if not grouped:
        raise ValueError("no train-fold telemetry columns map to the topology manifest")
    for values in grouped.values():
        values.sort()

    selected: list[tuple[str, str, str]] = []
    max_group = max(len(values) for values in grouped.values())
    for offset in range(max_group):
        for modality in MODALITIES:
            for service in topology.services:
                values = grouped.get((service, modality), ())
                if offset < len(values):
                    selected.append((values[offset], service, modality))
                    if len(selected) >= int(max_features):
                        break
            if len(selected) >= int(max_features):
                break
        if len(selected) >= int(max_features):
            break

    by_group: dict[tuple[str, str], list[str]] = {}
    for name, service, modality in selected:
        by_group.setdefault((service, modality), []).append(name)
    modality_names = tuple(modality for modality in MODALITIES if any(key[1] == modality for key in by_group))
    modality_dims = tuple(
        max((len(by_group.get((service, modality), ())) for service in topology.services), default=0)
        for modality in modality_names
    )
    slot_lookup: dict[tuple[str, str, str], int] = {}
    for (service, modality), names in by_group.items():
        for slot, name in enumerate(names):
            slot_lookup[(name, service, modality)] = slot
    specs = tuple(
        FeatureSpec(name=name, service=service, modality=modality, slot=slot_lookup[(name, service, modality)])
        for name, service, modality in selected
    )
    return FeatureSchema(
        version="failure-onset-feature-schema-v1",
        services=topology.services,
        modality_names=modality_names,
        modality_dims=modality_dims,
        features=specs,
        max_features=int(max_features),
        fitted_case_ids=tuple(sorted(str(case.case_id) for case in train_cases)),
    )


def _resolve_root_target(label: str | None, topology: TopologyManifest) -> int:
    if not label:
        return -1
    normalized = _normalize(label)
    aliases = {_normalize(key): value for key, value in topology.aliases.items()}
    if normalized in aliases:
        return topology.services.index(aliases[normalized])
    normalized_services = [_normalize(service) for service in topology.services]
    if normalized in normalized_services:
        return normalized_services.index(normalized)
    matches = [
        idx
        for idx, service in enumerate(normalized_services)
        if normalized.startswith(service) or service.startswith(normalized)
    ]
    return max(matches, key=lambda idx: len(normalized_services[idx])) if matches else -1


def build_failure_windows(
    cases: Sequence[TelemetryCase],
    schema: FeatureSchema,
    topology: TopologyManifest,
    horizons: Sequence[float] = (60.0, 180.0, 300.0),
    window_size: int = 24,
    stride: int = 2,
    grid_seconds: int = 15,
    require_all_horizons_observed: bool = True,
) -> FailureWindowDataset:
    if tuple(schema.services) != tuple(topology.services):
        raise ValueError("feature schema and topology service order differ")
    horizons = tuple(float(item) for item in horizons)
    if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique and strictly increasing")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    node_index = {service: idx for idx, service in enumerate(schema.services)}
    modality_index = {modality: idx for idx, modality in enumerate(schema.modality_names)}
    modality_offsets = {}
    offset = 0
    for modality, width in zip(schema.modality_names, schema.modality_dims):
        modality_offsets[modality] = offset
        offset += width

    rows = []
    labels = []
    label_masks = []
    modality_masks = []
    feature_masks = []
    case_ids: list[str] = []
    scenario_ids: list[str] = []
    systems: list[str] = []
    end_times = []
    inject_times = []
    failure_times = []
    censor_times = []
    lead_times = []
    root_targets = []
    root_active = []
    root_labels: list[str | None] = []

    for case in cases:
        if case.inject_time is None:
            raise ValueError(f"case {case.case_id} has no inject_time")
        if case.censor_time is None:
            raise ValueError(f"case {case.case_id} has no censor_time")
        frame = right_closed_resample(case.frame, interval_seconds=grid_seconds, agg="mean", fill_empty=None)
        if len(frame) < window_size:
            continue
        frame = frame.ffill().fillna(0.0)
        columns = set(str(column) for column in frame.columns)
        values_by_feature = {
            spec.name: frame[spec.name].to_numpy(dtype=float)
            for spec in schema.features
            if spec.name in columns
        }
        case_modality_mask = np.zeros((len(schema.services), len(schema.modality_names)), dtype=bool)
        case_feature_mask = np.zeros((len(schema.services), schema.input_dim), dtype=bool)
        for spec in schema.features:
            if spec.name not in values_by_feature:
                continue
            node = node_index[spec.service]
            modality = modality_index[spec.modality]
            position = modality_offsets[spec.modality] + spec.slot
            case_modality_mask[node, modality] = True
            case_feature_mask[node, position] = True

        times = frame[TIME_COLUMN].to_numpy(dtype=float)
        inject_time = float(case.inject_time)
        failure_time = float(case.failure_time) if case.failure_time is not None else math.nan
        censor_time = float(case.censor_time)
        target_index = _resolve_root_target(case.root_label, topology)
        for end_idx in range(window_size - 1, len(frame), stride):
            end_time = float(times[end_idx])
            if end_time > censor_time:
                continue
            if math.isfinite(failure_time) and end_time >= failure_time:
                # Strictly no post-failure inputs or labels.
                continue
            y = np.zeros(len(horizons), dtype=np.float32)
            valid = np.ones(len(horizons), dtype=bool)
            lead = failure_time - end_time if math.isfinite(failure_time) else math.inf
            if end_time >= inject_time:
                if math.isfinite(failure_time):
                    y[:] = [float(0.0 < lead <= horizon) for horizon in horizons]
                else:
                    # A censored trajectory is a known negative only when its
                    # complete forecast horizon was observed.
                    valid[:] = [end_time + horizon <= censor_time for horizon in horizons]
            # Pre-injection windows are explicitly healthy controls by design.
            if not valid.any() or (require_all_horizons_observed and not valid.all()):
                continue

            window = np.zeros(
                (len(schema.services), window_size, schema.input_dim),
                dtype=np.float32,
            )
            start_idx = end_idx - window_size + 1
            for spec in schema.features:
                series = values_by_feature.get(spec.name)
                if series is None:
                    continue
                node = node_index[spec.service]
                position = modality_offsets[spec.modality] + spec.slot
                window[node, :, position] = series[start_idx : end_idx + 1]

            rows.append(window)
            labels.append(y)
            label_masks.append(valid)
            modality_masks.append(case_modality_mask.copy())
            feature_masks.append(case_feature_mask.copy())
            case_ids.append(str(case.case_id))
            scenario_ids.append(str(case.scenario_id))
            systems.append(str(case.system or topology.system))
            end_times.append(end_time)
            inject_times.append(inject_time)
            failure_times.append(failure_time)
            censor_times.append(censor_time)
            lead_times.append(lead)
            root_targets.append(target_index)
            root_active.append(bool(end_time >= inject_time and math.isfinite(lead) and 0.0 < lead <= max(horizons)))
            root_labels.append(case.root_label)

    if not rows:
        raise ValueError("no v2 failure-onset windows were generated")
    return FailureWindowDataset(
        X=np.stack(rows),
        Y=np.stack(labels),
        label_mask=np.stack(label_masks),
        availability_mask=np.stack(modality_masks),
        feature_availability_mask=np.stack(feature_masks),
        horizons=horizons,
        case_ids=tuple(case_ids),
        scenario_ids=tuple(scenario_ids),
        systems=tuple(systems),
        end_times=np.asarray(end_times, dtype=float),
        inject_times=np.asarray(inject_times, dtype=float),
        failure_times=np.asarray(failure_times, dtype=float),
        censor_times=np.asarray(censor_times, dtype=float),
        lead_times=np.asarray(lead_times, dtype=float),
        root_targets=np.asarray(root_targets, dtype=np.int64),
        root_active=np.asarray(root_active, dtype=bool),
        root_labels=tuple(root_labels),
        schema=schema,
    )


def standardize_train_only(
    dataset: FailureWindowDataset,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit node/slot statistics using available train-fold telemetry only."""

    train_indices = np.asarray(train_indices, dtype=int)
    if len(train_indices) == 0:
        raise ValueError("train_indices cannot be empty")
    X = dataset.X.astype(np.float64, copy=True)
    feature_mask = dataset.feature_availability_mask
    n_nodes, n_features = X.shape[1], X.shape[3]
    means = np.zeros((n_nodes, n_features), dtype=np.float64)
    scales = np.ones((n_nodes, n_features), dtype=np.float64)
    for node in range(n_nodes):
        for feature in range(n_features):
            available_cases = train_indices[feature_mask[train_indices, node, feature]]
            if len(available_cases) == 0:
                continue
            values = X[available_cases, node, :, feature].reshape(-1)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                continue
            means[node, feature] = float(finite.mean())
            scale = float(finite.std())
            scales[node, feature] = scale if scale >= 1e-6 else 1.0
    X = (X - means[None, :, None, :]) / scales[None, :, None, :]
    X *= feature_mask[:, :, None, :]
    return X.astype(np.float32), means.astype(np.float32), scales.astype(np.float32)


def indices_for_cases(dataset: FailureWindowDataset, case_ids: Sequence[str]) -> np.ndarray:
    allowed = {str(item) for item in case_ids}
    return np.flatnonzero(np.asarray([case_id in allowed for case_id in dataset.case_ids], dtype=bool))


def service_signal(dataset: FailureWindowDataset, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """One train-only magnitude observation per window and topology service."""

    idx = np.asarray(indices, dtype=int)
    magnitude = np.abs(X[idx]).mean(axis=(2, 3))
    available = dataset.availability_mask[idx].any(axis=-1)
    return np.where(available, magnitude, 0.0)


def feature_exclusions_from_manifest(records: Sequence[Mapping]) -> tuple[str, ...]:
    """Collect target SLI and source columns for the leakage ablation."""

    excluded: set[str] = set()

    def collect(value) -> None:
        if isinstance(value, str):
            excluded.add(value)
        elif isinstance(value, Mapping):
            for key, nested in value.items():
                if key in {
                    "name",
                    "latency_column",
                    "error_column",
                    "request_column",
                    "columns",
                    "source_columns",
                }:
                    collect(nested)
                elif isinstance(nested, (Mapping, list, tuple)):
                    collect(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for nested in value:
                collect(nested)

    for record in records:
        collect(record.get("sli"))
        collect(record.get("sli_columns"))
    return tuple(sorted(excluded))
