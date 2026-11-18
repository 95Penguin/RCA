#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import (  # noqa: E402
    TIME_COLUMN,
    TelemetryCase,
    clean_metric_frame,
    load_multisource_sample_case,
    load_rcaeval_cases,
    load_sample_csv_case,
    split_stratified,
)
from ltfp.features import infer_service_name  # noqa: E402
from ltfp.metrics import evaluate_multi_horizon, tune_threshold  # noqa: E402
from ltfp.protocol import load_topology_manifest  # noqa: E402
from ltfp.synthetic import SyntheticConfig, make_synthetic_cases  # noqa: E402
from ltfp.torch_models import LTFPConfig, LTFPModel, lead_time_multitask_loss  # noqa: E402


@dataclass(frozen=True)
class TemporalGraphDataset:
    X: np.ndarray
    Y: np.ndarray
    horizons: tuple[float, ...]
    services: tuple[str, ...]
    metric_names: tuple[str, ...]
    modality_names: tuple[str, ...]
    modality_dims: tuple[int, ...]
    case_names: tuple[str, ...]
    end_times: np.ndarray
    lead_times: np.ndarray
    root_targets: np.ndarray
    root_candidate_mask: np.ndarray
    type_targets: np.ndarray
    fault_types: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "sample-csv", "multisource-sample", "rcaeval-cases"],
        default="sample-csv",
    )
    parser.add_argument("--case-root", default="data/RE1/RE1-OB")
    parser.add_argument("--sample-csv", default="data/raw/rcaeval/simple_data.csv")
    parser.add_argument("--multisource-dir", default="data/raw/rcaeval/multi-source-data")
    parser.add_argument("--incident-time", type=float, default=None)
    parser.add_argument("--root-label", default=None)
    parser.add_argument("--fault-type", default=None)
    parser.add_argument("--horizons", default="60,180,300,600,900")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dataset-cache", default="")
    parser.add_argument("--synthetic-cases", type=int, default=120)
    parser.add_argument("--max-raw-features", type=int, default=96)
    parser.add_argument("--feature-selection", choices=["first", "balanced"], default="balanced")
    parser.add_argument(
        "--allowed-modalities",
        default="metric,log,trace",
        help="Comma-separated modality allowlist for feature selection, e.g., metric or metric,log.",
    )
    parser.add_argument(
        "--feature-schema-modalities",
        default="",
        help=(
            "Optional modality set used to fit the shared feature budget before applying "
            "--allowed-modalities. This keeps modality ablations on the full model's feature schema."
        ),
    )
    parser.add_argument("--include-delta-features", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-tcn-layers", type=int, default=3)
    parser.add_argument("--n-graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--graph-top-k", type=int, default=2)
    parser.add_argument("--topology-manifest", default="")
    parser.add_argument(
        "--graph-mode",
        choices=[
            "hybrid-max",
            "dependency-only",
            "correlation-only",
            "hybrid-learned",
            "hybrid-lagged",
            "hybrid-random",
        ],
        default="hybrid-max",
    )
    parser.add_argument("--graph-lag-steps", type=int, default=3)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument(
        "--max-normal-ratio",
        type=float,
        default=0.0,
        help="If >0, cap normal training windows to this multiple of positive windows.",
    )
    parser.add_argument(
        "--hard-negative-seconds",
        type=float,
        default=0.0,
        help="Prefer normal training windows within this lead-time range before sampling far normals.",
    )
    parser.add_argument("--horizon-weight-mode", choices=["lead", "short", "flat"], default="short")
    parser.add_argument("--type-weight", type=float, default=0.7)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--root-prior-weight", type=float, default=0.0)
    parser.add_argument("--lead-weight", type=float, default=0.1)
    parser.add_argument("--lead-reg-weight", type=float, default=0.0)
    parser.add_argument("--focus-horizon", type=float, default=None)
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-fusion",
        choices=["none", "stat"],
        default="none",
        help="Fuse neural risk with a train-fitted statistical window-deviation score.",
    )
    return parser.parse_args()


def load_cases(args: argparse.Namespace) -> list[TelemetryCase]:
    if args.dataset == "synthetic":
        config = SyntheticConfig(n_cases=args.synthetic_cases, seed=args.seed + 6)
        return make_synthetic_cases(config)
    if args.dataset == "sample-csv":
        incident_time = args.incident_time if args.incident_time is not None else 1692569339.0
        return [
            load_sample_csv_case(
                args.sample_csv,
                incident_time=incident_time,
                root_label=args.root_label,
                fault_type=args.fault_type,
            )
        ]
    if args.dataset == "multisource-sample":
        return [
            load_multisource_sample_case(
                args.multisource_dir,
                incident_time=args.incident_time,
                root_label=args.root_label,
                fault_type=args.fault_type,
            )
        ]
    return load_rcaeval_cases(args.case_root, limit=args.case_limit)


def collect_metric_names(
    cases: Sequence[TelemetryCase],
    max_raw_features: int,
    feature_selection: str = "balanced",
    allowed_modalities: Sequence[str] | None = None,
    feature_schema_modalities: Sequence[str] | None = None,
) -> tuple[str, ...]:
    names = []
    seen = set()
    allowed = None
    if allowed_modalities is not None:
        allowed = {str(item).strip() for item in allowed_modalities if str(item).strip()}
    schema_allowed = None
    if feature_schema_modalities is not None:
        schema_allowed = {str(item).strip() for item in feature_schema_modalities if str(item).strip()}
    discovery_allowed = schema_allowed or allowed

    def retain_allowed(selected_names: Sequence[str]) -> tuple[str, ...]:
        if allowed is None:
            return tuple(selected_names)
        return tuple(name for name in selected_names if infer_modality_name(name) in allowed)

    for case in cases:
        frame = clean_metric_frame(case.frame)
        for col in frame.columns:
            if col == TIME_COLUMN or col in seen:
                continue
            if discovery_allowed is not None and infer_modality_name(col) not in discovery_allowed:
                continue
            seen.add(col)
            names.append(col)
    if feature_selection == "first":
        return retain_allowed(names[:max_raw_features])

    grouped: dict[tuple[str, str], list[str]] = {}
    service_order = []
    modality_order = []
    for name in names:
        service = infer_service_name(name)
        modality = infer_modality_name(name)
        key = (modality, service)
        if key not in grouped:
            grouped[key] = []
        if service not in service_order:
            service_order.append(service)
        if modality not in modality_order:
            modality_order.append(modality)
        grouped[key].append(name)

    selected = []
    preferred_modalities = [item for item in ("metric", "log", "trace") if item in modality_order]
    preferred_modalities.extend(item for item in modality_order if item not in preferred_modalities)
    max_group_size = max((len(values) for values in grouped.values()), default=0)
    for offset in range(max_group_size):
        for modality in preferred_modalities:
            for service in service_order:
                values = grouped.get((modality, service), [])
                if offset >= len(values):
                    continue
                selected.append(values[offset])
                if len(selected) >= max_raw_features:
                    return retain_allowed(selected)
    return retain_allowed(selected)


def infer_modality_name(metric_name: str) -> str:
    name = str(metric_name)
    if "_trace_err_" in name or "_trace_lat_" in name:
        return "trace"
    if "_log_" in name:
        return "log"
    return "metric"


def service_groups(
    metric_names: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, dict[str, list[int]]], tuple[str, ...], tuple[int, ...], int]:
    groups: dict[str, dict[str, list[int]]] = {}
    for idx, metric in enumerate(metric_names):
        service = infer_service_name(metric)
        modality = infer_modality_name(metric)
        groups.setdefault(service, {}).setdefault(modality, []).append(idx)
    services = tuple(sorted(groups))
    modality_names = tuple(item for item in ("metric", "log", "trace") if any(item in groups[svc] for svc in services))
    modality_dims = tuple(
        max((len(groups[service].get(modality, [])) for service in services), default=0)
        for modality in modality_names
    )
    node_feature_dim = max(sum(modality_dims), 1)
    return services, groups, modality_names, modality_dims, node_feature_dim


def build_temporal_graph_dataset(
    cases: Sequence[TelemetryCase],
    horizons: Sequence[float],
    window_size: int,
    stride: int,
    max_raw_features: int,
    feature_selection: str = "balanced",
    allowed_modalities: Sequence[str] | None = None,
    feature_schema_modalities: Sequence[str] | None = None,
    include_delta_features: bool = False,
) -> TemporalGraphDataset:
    metric_names = collect_metric_names(
        cases,
        max_raw_features,
        feature_selection=feature_selection,
        allowed_modalities=allowed_modalities,
        feature_schema_modalities=feature_schema_modalities,
    )
    if not metric_names:
        raise ValueError(f"no features remain after modality filtering: {allowed_modalities}")
    services, groups, modality_names, raw_modality_dims, raw_node_feature_dim = service_groups(metric_names)
    modality_dims = tuple(dim * 2 if include_delta_features else dim for dim in raw_modality_dims)
    node_feature_dim = max(sum(modality_dims), 1)
    type_names = tuple(sorted({case.fault_type for case in cases if case.fault_type}))
    type_to_idx = {name: idx for idx, name in enumerate(type_names)}
    root_candidate_mask = build_root_candidate_mask(cases, services)
    modality_offsets = {}
    offset = 0
    for modality, width in zip(modality_names, modality_dims):
        modality_offsets[modality] = offset
        offset += width

    rows = []
    labels = []
    case_names = []
    end_times = []
    lead_times = []
    root_targets = []
    type_targets = []

    for case in cases:
        frame = clean_metric_frame(case.frame)
        values = frame.reindex(columns=metric_names, fill_value=0.0).to_numpy(dtype=float)
        times = frame[TIME_COLUMN].to_numpy(dtype=float)
        if len(values) < window_size:
            continue

        for end_idx in range(window_size - 1, len(values), stride):
            end_time = float(times[end_idx])
            lead = math.inf
            if case.incident_time is not None:
                lead = float(case.incident_time - end_time)
                if lead < 0:
                    continue

            y = [int(0 <= lead <= horizon) for horizon in horizons]
            window = values[end_idx - window_size + 1 : end_idx + 1]
            tensor = np.zeros((len(services), window_size, node_feature_dim), dtype=np.float32)
            for service_idx, service in enumerate(services):
                for modality, metric_indices in groups[service].items():
                    start = modality_offsets[modality]
                    raw_width = raw_modality_dims[modality_names.index(modality)]
                    for local_idx, metric_idx in enumerate(metric_indices):
                        series = window[:, metric_idx]
                        tensor[service_idx, :, start + local_idx] = series
                        if include_delta_features:
                            delta = np.diff(series, prepend=series[0])
                            tensor[service_idx, :, start + raw_width + local_idx] = delta

            rows.append(tensor)
            labels.append(y)
            case_names.append(case.name)
            end_times.append(end_time)
            lead_times.append(lead)
            root_targets.append(resolve_root_target(case.root_label, services))
            type_targets.append(type_to_idx.get(case.fault_type, -1))

    if not rows:
        raise ValueError("no temporal graph windows were generated")

    return TemporalGraphDataset(
        X=np.stack(rows).astype(np.float32),
        Y=np.asarray(labels, dtype=np.float32),
        horizons=tuple(float(horizon) for horizon in horizons),
        services=services,
        metric_names=metric_names,
        modality_names=modality_names,
        modality_dims=modality_dims,
        case_names=tuple(case_names),
        end_times=np.asarray(end_times, dtype=float),
        lead_times=np.asarray(lead_times, dtype=float),
        root_targets=np.asarray(root_targets, dtype=np.int64),
        root_candidate_mask=root_candidate_mask,
        type_targets=np.asarray(type_targets, dtype=np.int64),
        fault_types=type_names,
    )


def save_temporal_graph_dataset(
    dataset: TemporalGraphDataset,
    path: str | Path,
    configuration: dict,
    n_cases: int,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        X=dataset.X,
        Y=dataset.Y,
        horizons=np.asarray(dataset.horizons),
        services=np.asarray(dataset.services),
        metric_names=np.asarray(dataset.metric_names),
        modality_names=np.asarray(dataset.modality_names),
        modality_dims=np.asarray(dataset.modality_dims),
        case_names=np.asarray(dataset.case_names),
        end_times=dataset.end_times,
        lead_times=dataset.lead_times,
        root_targets=dataset.root_targets,
        root_candidate_mask=dataset.root_candidate_mask,
        type_targets=dataset.type_targets,
        fault_types=np.asarray(dataset.fault_types),
        configuration=np.asarray(json.dumps(configuration, sort_keys=True)),
        n_cases=np.asarray(int(n_cases)),
    )


def load_temporal_graph_dataset(
    path: str | Path,
    configuration: dict,
) -> tuple[TemporalGraphDataset, int]:
    source = np.load(path)
    recorded = json.loads(str(source["configuration"].item()))
    if recorded != configuration:
        raise ValueError(f"dataset cache configuration mismatch for {path}")
    dataset = TemporalGraphDataset(
        X=source["X"],
        Y=source["Y"],
        horizons=tuple(source["horizons"].astype(float).tolist()),
        services=tuple(source["services"].astype(str).tolist()),
        metric_names=tuple(source["metric_names"].astype(str).tolist()),
        modality_names=tuple(source["modality_names"].astype(str).tolist()),
        modality_dims=tuple(source["modality_dims"].astype(int).tolist()),
        case_names=tuple(source["case_names"].astype(str).tolist()),
        end_times=source["end_times"],
        lead_times=source["lead_times"],
        root_targets=source["root_targets"],
        root_candidate_mask=source["root_candidate_mask"].astype(bool),
        type_targets=source["type_targets"],
        fault_types=tuple(source["fault_types"].astype(str).tolist()),
    )
    return dataset, int(source["n_cases"].item())


def build_root_candidate_mask(cases: Sequence[TelemetryCase], services: Sequence[str]) -> np.ndarray:
    labels = [normalize_name(case.root_label) for case in cases if case.root_label]
    if not labels:
        return np.ones(len(services), dtype=bool)
    mask = []
    for service in services:
        normalized = normalize_name(service)
        mask.append(any(label == normalized or label.startswith(normalized) or normalized.startswith(label) for label in labels))
    if not any(mask):
        return np.ones(len(services), dtype=bool)
    return np.asarray(mask, dtype=bool)


def resolve_root_target(label: str | None, services: Sequence[str]) -> int:
    if not label:
        return -1
    normalized = normalize_name(label)
    normalized_services = [normalize_name(service) for service in services]
    for idx, service in enumerate(normalized_services):
        if normalized == service:
            return idx
    for idx, service in enumerate(normalized_services):
        if normalized.startswith(service) or service.startswith(normalized):
            return idx
    return -1


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(".", "_").replace("/", "_").replace("-", "_")


def split_indices(dataset: TemporalGraphDataset, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_cases = np.asarray(sorted(set(dataset.case_names)))
    if len(unique_cases) <= 1:
        return split_stratified(dataset.Y[:, -1], test_ratio=test_ratio, seed=seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_cases)
    n_test = max(1, int(round(len(unique_cases) * test_ratio)))
    test_cases = set(unique_cases[:n_test])
    test_mask = np.asarray([name in test_cases for name in dataset.case_names])
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return split_stratified(dataset.Y[:, -1], test_ratio=test_ratio, seed=seed)
    return train_idx, test_idx


def select_training_indices(
    dataset: TemporalGraphDataset,
    train_idx: np.ndarray,
    max_normal_ratio: float,
    hard_negative_seconds: float,
    seed: int,
) -> np.ndarray:
    if max_normal_ratio <= 0:
        return train_idx

    positive_mask = dataset.Y[train_idx].max(axis=1) > 0
    positive_idx = train_idx[positive_mask]
    normal_idx = train_idx[~positive_mask]
    if len(positive_idx) == 0 or len(normal_idx) == 0:
        return train_idx

    hard_negative_idx = np.asarray([], dtype=train_idx.dtype)
    far_negative_idx = normal_idx
    if hard_negative_seconds > 0:
        lead = dataset.lead_times[normal_idx]
        hard_mask = np.isfinite(lead) & (lead <= hard_negative_seconds)
        hard_negative_idx = normal_idx[hard_mask]
        far_negative_idx = normal_idx[~hard_mask]

    rng = np.random.default_rng(seed + 23)
    max_normal = max(1, int(round(len(positive_idx) * max_normal_ratio)))
    if len(hard_negative_idx) >= max_normal:
        selected_normal_idx = rng.choice(hard_negative_idx, size=max_normal, replace=False)
    else:
        selected_parts = [hard_negative_idx]
        remaining = max_normal - len(hard_negative_idx)
        if remaining > 0 and len(far_negative_idx) > 0:
            sample_size = min(remaining, len(far_negative_idx))
            selected_parts.append(rng.choice(far_negative_idx, size=sample_size, replace=False))
        selected_normal_idx = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=train_idx.dtype)

    selected = np.concatenate([positive_idx, selected_normal_idx])
    selected.sort()
    return selected


def standardize_by_train(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = X[train_idx]
    mean = train.mean(axis=(0, 2), keepdims=True)
    scale = train.std(axis=(0, 2), keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((X - mean) / scale).astype(np.float32), mean, scale


def _service_index(name: object, service_to_idx: dict[str, int], aliases: dict[str, str]) -> int | None:
    raw = f"svc{name}" if isinstance(name, (int, np.integer)) else str(name)
    if raw in service_to_idx:
        return service_to_idx[raw]
    normalized = normalize_name(raw)
    normalized_services = {normalize_name(service): idx for service, idx in service_to_idx.items()}
    if normalized in normalized_services:
        return normalized_services[normalized]
    alias_target = aliases.get(normalized)
    if alias_target is not None:
        return normalized_services.get(normalize_name(alias_target))
    return None


def _sparsify_correlation(correlation: np.ndarray, top_k: int) -> np.ndarray:
    correlation = np.asarray(correlation, dtype=np.float32).copy()
    np.fill_diagonal(correlation, 0.0)
    sparse = np.zeros_like(correlation)
    if top_k <= 0 or len(correlation) <= 1:
        return sparse
    retained = min(int(top_k), len(correlation) - 1)
    for idx in range(len(correlation)):
        neighbors = np.argsort(correlation[idx])[-retained:]
        sparse[idx, neighbors] = correlation[idx, neighbors]
    return sparse


def _window_activity_correlation(X_train: np.ndarray, top_k: int) -> np.ndarray:
    n_nodes = X_train.shape[1]
    if len(X_train) <= 1 or n_nodes <= 1:
        return np.zeros((n_nodes, n_nodes), dtype=np.float32)
    service_signal = np.abs(X_train).mean(axis=(2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(service_signal, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    return _sparsify_correlation(correlation, top_k)


def _lagged_activity_correlation(X_train: np.ndarray, top_k: int, max_lag: int) -> np.ndarray:
    n_nodes = X_train.shape[1]
    if len(X_train) <= 1 or n_nodes <= 1:
        return np.zeros((n_nodes, n_nodes), dtype=np.float32)
    activity = np.abs(X_train).mean(axis=3)
    max_lag = max(0, min(int(max_lag), activity.shape[2] - 1))
    full = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for source in range(n_nodes):
        for target in range(source + 1, n_nodes):
            best = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    left = activity[:, source, -lag:]
                    right = activity[:, target, : activity.shape[2] + lag]
                elif lag > 0:
                    left = activity[:, source, : activity.shape[2] - lag]
                    right = activity[:, target, lag:]
                else:
                    left = activity[:, source]
                    right = activity[:, target]
                left = left.reshape(-1)
                right = right.reshape(-1)
                if left.std() < 1e-12 or right.std() < 1e-12:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    value = float(abs(np.corrcoef(left, right)[0, 1]))
                if np.isfinite(value):
                    best = max(best, value)
            full[source, target] = best
            full[target, source] = best
    return _sparsify_correlation(full, top_k)


def _randomize_correlation(correlation: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    randomized = np.zeros_like(correlation)
    nodes = np.arange(len(correlation))
    for idx in range(len(correlation)):
        weights = correlation[idx, correlation[idx] > 0].copy()
        if len(weights) == 0:
            continue
        candidates = nodes[nodes != idx]
        chosen = rng.choice(candidates, size=min(len(weights), len(candidates)), replace=False)
        rng.shuffle(weights)
        randomized[idx, chosen] = weights[: len(chosen)]
    return randomized.astype(np.float32)


def build_graph_components(
    cases: Sequence[TelemetryCase],
    dataset: TemporalGraphDataset,
    X_train: np.ndarray,
    top_k: int,
    topology_manifest: str = "",
    max_lag: int = 3,
    seed: int = 7,
    compute_lagged: bool = True,
    compute_random: bool = True,
) -> dict[str, np.ndarray]:
    n_nodes = len(dataset.services)
    dependency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    service_to_idx = {service: idx for idx, service in enumerate(dataset.services)}
    aliases: dict[str, str] = {}

    manifest_edges: list[tuple[object, object]] = []
    if topology_manifest:
        topology = load_topology_manifest(topology_manifest)
        aliases = {normalize_name(key): value for key, value in topology.aliases.items()}
        manifest_edges.extend(topology.edges)

    for case in cases:
        manifest_edges.extend((case.metadata or {}).get("edges", []))
    for source, target in manifest_edges:
        source_idx = _service_index(source, service_to_idx, aliases)
        target_idx = _service_index(target, service_to_idx, aliases)
        if source_idx is not None and target_idx is not None and source_idx != target_idx:
            dependency[source_idx, target_idx] = 1.0
            dependency[target_idx, source_idx] = 1.0

    correlation = _window_activity_correlation(X_train, top_k)
    lagged = (
        _lagged_activity_correlation(X_train, top_k, max_lag)
        if compute_lagged
        else np.zeros_like(correlation)
    )
    randomized = (
        _randomize_correlation(correlation, seed)
        if compute_random
        else np.zeros_like(correlation)
    )
    return {
        "dependency": dependency,
        "correlation": correlation,
        "lagged_correlation": lagged,
        "random_correlation": randomized,
    }


def adjacency_for_mode(components: dict[str, np.ndarray], graph_mode: str) -> np.ndarray:
    dependency = components["dependency"]
    correlation = components["correlation"]
    if graph_mode == "dependency-only":
        return dependency.copy()
    if graph_mode == "correlation-only":
        return correlation.copy()
    if graph_mode == "hybrid-lagged":
        return np.maximum(dependency, components["lagged_correlation"])
    if graph_mode == "hybrid-random":
        return np.maximum(dependency, components["random_correlation"])
    return np.maximum(dependency, correlation)


def build_adjacency(
    cases: Sequence[TelemetryCase],
    dataset: TemporalGraphDataset,
    X_train: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Backward-compatible default hybrid graph used by older callers."""

    components = build_graph_components(cases, dataset, X_train, top_k)
    return adjacency_for_mode(components, "hybrid-max")


def iter_minibatches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = indices.copy()
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def apply_root_candidate_mask(outputs: dict, root_candidate_mask):
    if root_candidate_mask is None:
        return outputs
    masked = dict(outputs)
    masked["root_logits"] = masked["root_logits"].masked_fill(~root_candidate_mask.unsqueeze(0), -1e9)
    return masked


def predict_fault_scores(
    model,
    X: np.ndarray,
    adjacency,
    indices: np.ndarray,
    batch_size: int,
    device,
    root_candidate_mask=None,
):
    import torch

    model.eval()
    scores = []
    type_logits = []
    root_logits = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
            out = apply_root_candidate_mask(model(xb, adjacency), root_candidate_mask)
            scores.append(torch.sigmoid(out["fault_logits"]).cpu().numpy())
            type_logits.append(out["type_logits"].cpu().numpy())
            root_logits.append(out["root_logits"].cpu().numpy())
    return np.vstack(scores), np.vstack(type_logits), np.vstack(root_logits)


def statistical_window_scores(X: np.ndarray) -> np.ndarray:
    """Cheap window-level deviation score from standardized telemetry.

    The score intentionally uses only generic temporal statistics so it can act
    as a calibration prior without dataset-specific labels or external models.
    """

    tail = X[:, :, -min(4, X.shape[2]) :, :]
    head = X[:, :, : min(4, X.shape[2]), :]
    tail_energy = np.mean(np.abs(tail), axis=(1, 2, 3))
    tail_peak = np.max(np.abs(tail), axis=(1, 2, 3))
    drift = np.mean(np.abs(tail.mean(axis=2) - head.mean(axis=2)), axis=(1, 2))
    volatility = np.std(X, axis=2).mean(axis=(1, 2))
    raw = 0.45 * tail_energy + 0.25 * tail_peak + 0.20 * drift + 0.10 * volatility
    return raw.astype(float)


def normalize_by_train(scores: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(train_scores, [0.01, 0.99])
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def fuse_scores_by_train(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    stat_train: np.ndarray,
    stat_test: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.zeros(train_scores.shape[1], dtype=float)
    fused_train = train_scores.copy()
    fused_test = test_scores.copy()
    candidates = np.linspace(0.0, 0.9, 10)
    for horizon_idx in range(train_scores.shape[1]):
        best_weight = 0.0
        best_f1 = -1.0
        for weight in candidates:
            candidate = (1.0 - weight) * train_scores[:, horizon_idx] + weight * stat_train
            threshold = tune_threshold(train_labels[:, horizon_idx], candidate)
            f1 = evaluate_multi_horizon(
                train_labels[:, [horizon_idx]],
                candidate[:, None],
                [float(horizon_idx)],
                [threshold],
            )[0]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_weight = float(weight)
        weights[horizon_idx] = best_weight
        fused_train[:, horizon_idx] = (1.0 - best_weight) * train_scores[:, horizon_idx] + best_weight * stat_train
        fused_test[:, horizon_idx] = (1.0 - best_weight) * test_scores[:, horizon_idx] + best_weight * stat_test
    return fused_train, fused_test, weights


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    mask = targets >= 0
    if not mask.any():
        return 0.0
    order = np.argsort(-logits[mask], axis=1, kind="stable")[:, :k]
    hits = [(target in row) for row, target in zip(order, targets[mask])]
    return float(np.mean(hits))


def mean_reciprocal_rank(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean reciprocal rank for one responsible-service label per window."""

    mask = targets >= 0
    if not mask.any():
        return 0.0
    order = np.argsort(-logits[mask], axis=1, kind="stable")
    valid_targets = targets[mask]
    ranks = np.asarray(
        [int(np.flatnonzero(row == target)[0]) + 1 for row, target in zip(order, valid_targets)],
        dtype=float,
    )
    return float(np.mean(1.0 / ranks))


def frequency_prior_logits(
    train_targets: np.ndarray,
    positive_train_mask: np.ndarray,
    candidate_mask: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Rank services by responsible-label frequency in positive training windows."""

    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    counts = np.zeros(len(candidate_mask), dtype=float)
    valid_targets = np.asarray(train_targets, dtype=int)[np.asarray(positive_train_mask, dtype=bool)]
    valid_targets = valid_targets[valid_targets >= 0]
    if len(valid_targets):
        np.add.at(counts, valid_targets, 1.0)
    counts[~candidate_mask] = -np.inf
    return np.broadcast_to(counts, (int(n_rows), len(counts))).copy()


def random_ranking_expectation(n_candidates: int, k: int) -> float:
    if n_candidates <= 0:
        return 0.0
    return float(min(k, n_candidates) / n_candidates)


def random_mrr_expectation(n_candidates: int) -> float:
    if n_candidates <= 0:
        return 0.0
    return float(sum(1.0 / rank for rank in range(1, n_candidates + 1)) / n_candidates)


def evaluate_event_level(
    dataset: TemporalGraphDataset,
    indices: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict]:
    """Case-level early-warning metrics.

    Point labels penalize every early alert outside a specific horizon window.
    Event metrics instead ask whether each incident case receives at least one
    alert inside the target pre-failure window, while separately reporting
    alerts that fire earlier than the window.
    """

    rows = []
    case_names = np.asarray(dataset.case_names)[indices]
    lead_times = dataset.lead_times[indices]
    unique_cases = sorted(set(case_names))
    for horizon_idx, horizon in enumerate(dataset.horizons):
        hits = 0
        missed = 0
        false_alarm_cases = 0
        first_alert_leads = []
        hit_alert_leads = []
        for case_name in unique_cases:
            mask = case_names == case_name
            case_leads = lead_times[mask]
            finite = np.isfinite(case_leads) & (case_leads >= 0)
            if not finite.any():
                continue
            case_scores = scores[mask, horizon_idx]
            pred = case_scores >= thresholds[horizon_idx]
            in_window = finite & (case_leads <= horizon)
            before_window = finite & (case_leads > horizon)

            if (pred & in_window).any():
                hits += 1
                hit_alert_leads.append(float(case_leads[pred & in_window].max()))
            else:
                missed += 1
            if (pred & before_window).any():
                false_alarm_cases += 1
            if (pred & finite).any():
                first_alert_leads.append(float(case_leads[pred & finite].max()))

        total_cases = hits + missed
        rows.append(
            {
                "horizon": float(horizon),
                "event_hit_rate": hits / max(total_cases, 1),
                "event_miss_rate": missed / max(total_cases, 1),
                "hit_cases": hits,
                "missed_cases": missed,
                "false_alarm_cases_before_window": false_alarm_cases,
                "false_alarm_case_rate_before_window": false_alarm_cases / max(total_cases, 1),
                "mean_first_alert_lead_seconds": float(np.mean(first_alert_leads)) if first_alert_leads else 0.0,
                "mean_hit_alert_lead_seconds": float(np.mean(hit_alert_leads)) if hit_alert_leads else 0.0,
            }
        )
    return rows


def run() -> dict:
    args = parse_args()
    horizons = [float(item) for item in args.horizons.split(",") if item.strip()]
    allowed_modalities = [item.strip() for item in args.allowed_modalities.split(",") if item.strip()]
    schema_modalities = [
        item.strip() for item in args.feature_schema_modalities.split(",") if item.strip()
    ] or None
    dataset_configuration = {
        "dataset": args.dataset,
        "case_root": args.case_root,
        "case_limit": args.case_limit,
        "horizons": horizons,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_raw_features": args.max_raw_features,
        "feature_selection": args.feature_selection,
        "allowed_modalities": allowed_modalities,
        "feature_schema_modalities": schema_modalities,
        "include_delta_features": args.include_delta_features,
    }
    cache_path = Path(args.dataset_cache) if args.dataset_cache else None
    if cache_path is not None and cache_path.exists():
        dataset, n_cases = load_temporal_graph_dataset(cache_path, dataset_configuration)
        cases: list[TelemetryCase] = []
    else:
        cases = load_cases(args)
        if not cases:
            raise ValueError("no cases loaded")
        n_cases = len(cases)
        dataset = build_temporal_graph_dataset(
            cases,
            horizons=horizons,
            window_size=args.window_size,
            stride=args.stride,
            max_raw_features=args.max_raw_features,
            feature_selection=args.feature_selection,
            allowed_modalities=allowed_modalities,
            feature_schema_modalities=schema_modalities,
            include_delta_features=args.include_delta_features,
        )
        if cache_path is not None:
            save_temporal_graph_dataset(dataset, cache_path, dataset_configuration, n_cases)
    train_idx, test_idx = split_indices(dataset, test_ratio=args.test_ratio, seed=args.seed)
    X, _, _ = standardize_by_train(dataset.X, train_idx)
    graph_components = build_graph_components(
        cases,
        dataset,
        X[train_idx],
        top_k=args.graph_top_k,
        topology_manifest=args.topology_manifest,
        max_lag=args.graph_lag_steps,
        seed=args.seed,
        compute_lagged=args.graph_mode == "hybrid-lagged",
        compute_random=args.graph_mode == "hybrid-random",
    )
    adjacency_np = adjacency_for_mode(graph_components, args.graph_mode)
    fit_idx = select_training_indices(
        dataset,
        train_idx=train_idx,
        max_normal_ratio=args.max_normal_ratio,
        hard_negative_seconds=args.hard_negative_seconds,
        seed=args.seed,
    )

    import torch

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    adjacency = torch.as_tensor(adjacency_np, dtype=torch.float32, device=device)
    dependency_adjacency = torch.as_tensor(
        graph_components["dependency"], dtype=torch.float32, device=device
    )
    correlation_adjacency = torch.as_tensor(
        graph_components["correlation"], dtype=torch.float32, device=device
    )
    source_logits = (
        torch.nn.Parameter(torch.zeros(2, dtype=torch.float32, device=device))
        if args.graph_mode == "hybrid-learned"
        else None
    )

    def current_adjacency():
        if source_logits is None:
            return adjacency
        weights = torch.softmax(source_logits, dim=0)
        return weights[0] * dependency_adjacency + weights[1] * correlation_adjacency

    n_fault_types = max(len(dataset.fault_types), 1)
    config = LTFPConfig(
        input_dim=X.shape[-1],
        hidden_dim=args.hidden_dim,
        n_fault_types=n_fault_types,
        n_horizons=len(dataset.horizons),
        n_tcn_layers=args.n_tcn_layers,
        n_graph_layers=args.n_graph_layers,
        dropout=args.dropout,
        modality_dims=dataset.modality_dims,
        root_prior_weight=args.root_prior_weight,
    )
    model = LTFPModel(config).to(device)
    optimized_parameters = list(model.parameters())
    if source_logits is not None:
        optimized_parameters.append(source_logits)
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    y_tensor = torch.as_tensor(dataset.Y, dtype=torch.float32, device=device)
    type_tensor = torch.as_tensor(dataset.type_targets, dtype=torch.long, device=device)
    root_tensor = torch.as_tensor(dataset.root_targets, dtype=torch.long, device=device)
    lead_time_tensor = torch.as_tensor(dataset.lead_times, dtype=torch.float32, device=device)
    root_candidate_mask = torch.as_tensor(dataset.root_candidate_mask, dtype=torch.bool, device=device)
    pos_weight = None
    if args.use_pos_weight:
        train_y = dataset.Y[fit_idx]
        pos = np.maximum(train_y.sum(axis=0), 1.0)
        neg = np.maximum(len(train_y) - train_y.sum(axis=0), 1.0)
        pos_weight = torch.as_tensor(np.clip(neg / pos, 0.25, 20.0), dtype=torch.float32, device=device)

    losses = []
    training_started = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch_idx in iter_minibatches(fit_idx, args.batch_size, rng):
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
            out = apply_root_candidate_mask(model(xb, current_adjacency()), root_candidate_mask)
            loss = lead_time_multitask_loss(
                out,
                y_tensor[batch_idx],
                type_labels=type_tensor[batch_idx],
                root_labels=root_tensor[batch_idx],
                horizons=dataset.horizons,
                type_weight=args.type_weight,
                root_weight=args.root_weight,
                lead_weight=args.lead_weight,
                lead_reg_weight=args.lead_reg_weight,
                horizon_weight_mode=args.horizon_weight_mode,
                focus_horizon=args.focus_horizon,
                focus_weight=args.focus_weight,
                lead_times=lead_time_tensor[batch_idx],
                pos_weight=pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    training_seconds = time.perf_counter() - training_started
    adjacency = current_adjacency().detach()

    train_scores, _, _ = predict_fault_scores(
        model,
        X,
        adjacency,
        train_idx,
        args.batch_size,
        device,
        root_candidate_mask=root_candidate_mask,
    )
    # The preceding train-set prediction also serves as a CPU warm-up, so the
    # timed test pass reflects steady-state batched inference.
    inference_started = time.perf_counter()
    test_scores, test_type_logits, test_root_logits = predict_fault_scores(
        model,
        X,
        adjacency,
        test_idx,
        args.batch_size,
        device,
        root_candidate_mask=root_candidate_mask,
    )
    inference_seconds = time.perf_counter() - inference_started
    fusion_weights = None
    if args.score_fusion == "stat":
        raw_stat = statistical_window_scores(X)
        stat_train = normalize_by_train(raw_stat[train_idx], raw_stat[train_idx])
        stat_test = normalize_by_train(raw_stat[test_idx], raw_stat[train_idx])
        train_scores, test_scores, fusion_weights = fuse_scores_by_train(
            train_scores,
            test_scores,
            stat_train,
            stat_test,
            dataset.Y[train_idx],
        )
    thresholds = np.asarray(
        [tune_threshold(dataset.Y[train_idx, idx], train_scores[:, idx]) for idx in range(dataset.Y.shape[1])]
    )
    rows = evaluate_multi_horizon(dataset.Y[test_idx], test_scores, dataset.horizons, thresholds)
    for horizon_idx, row in enumerate(rows):
        labels = dataset.Y[test_idx, horizon_idx].astype(int)
        predictions = test_scores[:, horizon_idx] >= thresholds[horizon_idx]
        negative_mask = labels == 0
        false_positives = int((predictions & negative_mask).sum())
        negative_windows = int(negative_mask.sum())
        row["negative_windows"] = negative_windows
        row["false_positives"] = false_positives
        row["false_positive_rate"] = (
            float(false_positives / negative_windows) if negative_windows else np.nan
        )
        row["model"] = "ltfp_tcn_gat"
    event_rows = evaluate_event_level(dataset, test_idx, test_scores, thresholds)
    for row in event_rows:
        row["model"] = "ltfp_tcn_gat"

    positive_test_mask = dataset.Y[test_idx].max(axis=1) > 0
    root_targets = dataset.root_targets[test_idx][positive_test_mask]
    root_logits = test_root_logits[positive_test_mask]
    positive_train_mask = dataset.Y[train_idx].max(axis=1) > 0
    prior_logits = frequency_prior_logits(
        dataset.root_targets[train_idx],
        positive_train_mask,
        dataset.root_candidate_mask,
        len(root_targets),
    )
    n_root_candidates = int(dataset.root_candidate_mask.sum())
    type_targets = dataset.type_targets[test_idx][positive_test_mask]
    type_valid = type_targets >= 0
    type_accuracy = 0.0
    if type_valid.any():
        type_accuracy = float((test_type_logits[positive_test_mask][type_valid].argmax(axis=1) == type_targets[type_valid]).mean())

    root_summary = {
        "root_top1": topk_accuracy(root_logits, root_targets, 1),
        "root_top3": topk_accuracy(root_logits, root_targets, 3),
        "root_top5": topk_accuracy(root_logits, root_targets, 5),
        "root_mrr": mean_reciprocal_rank(root_logits, root_targets),
        "frequency_prior_top1": topk_accuracy(prior_logits, root_targets, 1),
        "frequency_prior_top3": topk_accuracy(prior_logits, root_targets, 3),
        "frequency_prior_top5": topk_accuracy(prior_logits, root_targets, 5),
        "frequency_prior_mrr": mean_reciprocal_rank(prior_logits, root_targets),
        "random_expected_top1": random_ranking_expectation(n_root_candidates, 1),
        "random_expected_top3": random_ranking_expectation(n_root_candidates, 3),
        "random_expected_top5": random_ranking_expectation(n_root_candidates, 5),
        "random_expected_mrr": random_mrr_expectation(n_root_candidates),
        "fault_type_accuracy": type_accuracy,
        "evaluated_positive_windows": int(positive_test_mask.sum()),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows)
    event_frame = pd.DataFrame(event_rows)
    csv_path = output_dir / f"{args.dataset}_ltfp_tcn_gat_results.csv"
    event_csv_path = output_dir / f"{args.dataset}_ltfp_tcn_gat_event_results.csv"
    json_path = output_dir / f"{args.dataset}_ltfp_tcn_gat_summary.json"
    graph_path = output_dir / f"{args.dataset}_graph_components.npz"
    result_frame.to_csv(csv_path, index=False)
    event_frame.to_csv(event_csv_path, index=False)
    np.savez_compressed(
        graph_path,
        services=np.asarray(dataset.services),
        dependency=graph_components["dependency"],
        correlation=graph_components["correlation"],
        lagged_correlation=graph_components["lagged_correlation"],
        random_correlation=graph_components["random_correlation"],
        final=adjacency.detach().cpu().numpy(),
    )

    def edge_count(matrix: np.ndarray) -> int:
        matrix = np.asarray(matrix).copy()
        np.fill_diagonal(matrix, 0.0)
        return int((matrix > 0).sum())

    learned_source_weights = None
    if source_logits is not None:
        learned_source_weights = torch.softmax(source_logits.detach(), dim=0).cpu().numpy().tolist()

    summary = {
        "dataset": args.dataset,
        "model": "ltfp_tcn_gat",
        "n_cases": n_cases,
        "n_windows": int(len(dataset.X)),
        "n_train": int(len(train_idx)),
        "n_fit": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
        "n_services": int(len(dataset.services)),
        "n_root_candidates": int(dataset.root_candidate_mask.sum()),
        "input_dim_per_service": int(X.shape[-1]),
        "n_metrics": int(len(dataset.metric_names)),
        "modality_names": list(dataset.modality_names),
        "modality_dims": list(dataset.modality_dims),
        "horizons": list(dataset.horizons),
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "n_tcn_layers": args.n_tcn_layers,
        "n_graph_layers": args.n_graph_layers,
        "graph_mode": args.graph_mode,
        "graph_top_k": args.graph_top_k,
        "graph_lag_steps": args.graph_lag_steps,
        "topology_manifest": args.topology_manifest or None,
        "graph_edge_counts": {
            "dependency": edge_count(graph_components["dependency"]),
            "correlation": edge_count(graph_components["correlation"]),
            "lagged_correlation": edge_count(graph_components["lagged_correlation"]),
            "random_correlation": edge_count(graph_components["random_correlation"]),
            "final": edge_count(adjacency.detach().cpu().numpy()),
        },
        "learned_source_weights": learned_source_weights,
        "dropout": args.dropout,
        "feature_selection": args.feature_selection,
        "allowed_modalities": [item.strip() for item in args.allowed_modalities.split(",") if item.strip()],
        "feature_schema_modalities": [
            item.strip() for item in args.feature_schema_modalities.split(",") if item.strip()
        ],
        "include_delta_features": args.include_delta_features,
        "max_normal_ratio": args.max_normal_ratio,
        "hard_negative_seconds": args.hard_negative_seconds,
        "horizon_weight_mode": args.horizon_weight_mode,
        "loss_weights": {
            "type": args.type_weight,
            "root": args.root_weight,
            "lead": args.lead_weight,
            "lead_reg": args.lead_reg_weight,
        },
        "focus_horizon": args.focus_horizon,
        "focus_weight": args.focus_weight,
        "root_prior_weight": args.root_prior_weight,
        "score_fusion": args.score_fusion,
        "score_fusion_weights": None if fusion_weights is None else fusion_weights.tolist(),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "n_parameters": int(sum(parameter.numel() for parameter in optimized_parameters)),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_window": float(1000.0 * inference_seconds / max(len(test_idx), 1)),
        "inference_windows_per_second": float(len(test_idx) / max(inference_seconds, 1e-12)),
        "pos_weight": None if pos_weight is None else pos_weight.detach().cpu().numpy().tolist(),
        "macro_f1": float(result_frame["f1"].mean()),
        "macro_auprc": float(result_frame["auprc"].mean()),
        "macro_event_hit_rate": float(event_frame["event_hit_rate"].mean()),
        "macro_false_alarm_case_rate_before_window": float(event_frame["false_alarm_case_rate_before_window"].mean()),
        "root_summary": root_summary,
        "result_csv": str(csv_path),
        "event_result_csv": str(event_csv_path),
        "graph_components": str(graph_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(result_frame.pivot_table(index="model", columns="horizon", values="f1").round(3))
    print()
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Macro AUPRC: {summary['macro_auprc']:.4f}")
    print(f"Macro event hit rate: {summary['macro_event_hit_rate']:.4f}")
    print(f"Training time: {summary['training_seconds']:.3f} s")
    print(
        "Inference: "
        f"{summary['inference_ms_per_window']:.3f} ms/window "
        f"({summary['inference_windows_per_second']:.1f} windows/s)"
    )
    print("Root/type:")
    for key, value in root_summary.items():
        print(f"  {key}: {value}")
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return summary


if __name__ == "__main__":
    run()
