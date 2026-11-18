#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import load_rcaeval_cases  # noqa: E402
from ltfp.protocol import load_topology_manifest  # noqa: E402


SUBSETS = {
    "re3-ob": (
        "data/RE3/RE3-OB",
        "manifests/failure_onset_v2/topology/online-boutique.v1.json",
    ),
    "re3-ss": (
        "data/RE3/RE3-SS",
        "manifests/failure_onset_v2/topology/sock-shop.v1.json",
    ),
    "re3-tt": (
        "data/RE3/RE3-TT",
        "manifests/failure_onset_v2/topology/train-ticket.v1.json",
    ),
}


def key(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def service_resolver(services: list[str], topology) -> dict[str, str]:
    resolver = {key(service): service for service in services}
    for alias, target in topology.aliases.items():
        if target in services:
            resolver[key(alias)] = target
    return resolver


def trace_edges(path: Path, resolver: dict[str, str]) -> Counter:
    columns = ["traceID", "spanID", "parentSpanID", "serviceName"]
    frame = pd.read_csv(path, usecols=lambda value: value in columns, dtype=str)
    if not set(columns).issubset(frame.columns):
        return Counter()
    lookup = {
        (str(trace), str(span)): resolver.get(key(service))
        for trace, span, service in frame[["traceID", "spanID", "serviceName"]].itertuples(index=False)
    }
    counts = Counter()
    for trace, parent, child_service in frame[["traceID", "parentSpanID", "serviceName"]].itertuples(index=False):
        parent_service = lookup.get((str(trace), str(parent)))
        child = resolver.get(key(child_service))
        if parent_service and child and parent_service != child:
            counts[frozenset((parent_service, child))] += 1
    return counts


def matrix_pairs(matrix: np.ndarray, services: list[str]) -> set[frozenset[str]]:
    pairs = set()
    for source, target in zip(*np.where(matrix > 0)):
        if source != target:
            pairs.add(frozenset((services[source], services[target])))
    return pairs


def split_train_cases(cases, seed: int = 7, test_ratio: float = 0.3) -> set[str]:
    names = np.asarray(sorted(case.name for case in cases))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_test = max(1, int(round(len(names) * test_ratio)))
    return set(names[n_test:])


def random_precision(
    candidates: list[frozenset[str]],
    runtime: set[frozenset[str]],
    sample_size: int,
    seed: int,
    repetitions: int = 1000,
) -> float:
    if not candidates or sample_size <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    values = []
    size = min(sample_size, len(candidates))
    for _ in range(repetitions):
        indices = rng.choice(len(candidates), size=size, replace=False)
        sample = {candidates[index] for index in indices}
        values.append(len(sample & runtime) / size)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/r2c2_graph_study")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    detail = {}
    for subset_index, (subset, (case_root, topology_path)) in enumerate(SUBSETS.items()):
        graph_path = root / subset / "hybrid-max" / "rcaeval-cases_graph_components.npz"
        if not graph_path.exists():
            continue
        graph = np.load(graph_path)
        services = graph["services"].astype(str).tolist()
        correlation_pairs = matrix_pairs(graph["correlation"], services)
        dependency_pairs = matrix_pairs(graph["dependency"], services)
        added_pairs = correlation_pairs - dependency_pairs

        cases = load_rcaeval_cases(case_root, limit=30)
        train_names = split_train_cases(cases)
        topology = load_topology_manifest(topology_path)
        resolver = service_resolver(services, topology)
        observed = Counter()
        checked = []
        for case in cases:
            if case.name not in train_names:
                continue
            trace_path = Path((case.metadata or {}).get("case_dir", "")) / "traces.csv"
            if trace_path.exists():
                observed.update(trace_edges(trace_path, resolver))
                checked.append(case.name)
        runtime_pairs = set(observed)
        all_pairs = [
            frozenset((services[left], services[right]))
            for left in range(len(services))
            for right in range(left + 1, len(services))
        ]
        supported = correlation_pairs & runtime_pairs
        added_supported = added_pairs & runtime_pairs
        baseline = random_precision(
            all_pairs,
            runtime_pairs,
            len(correlation_pairs),
            seed=7 + subset_index,
        )
        precision = len(supported) / max(len(correlation_pairs), 1)
        rows.append(
            {
                "dataset": subset,
                "training_cases_with_traces": len(checked),
                "runtime_edges": len(runtime_pairs),
                "correlation_edges": len(correlation_pairs),
                "correlation_added_edges": len(added_pairs),
                "supported_edges": len(supported),
                "added_supported_edges": len(added_supported),
                "precision": precision,
                "random_precision": baseline,
                "enrichment": precision / baseline if baseline > 0 else None,
            }
        )
        detail[subset] = {
            "scope": "training-cases-only",
            "checked_cases": checked,
            "supported_correlation_edges": [sorted(pair) for pair in sorted(supported, key=lambda p: sorted(p))],
            "runtime_span_counts": {
                " -- ".join(sorted(pair)): int(observed[pair]) for pair in sorted(supported, key=lambda p: sorted(p))
            },
        }
    if not rows:
        raise SystemExit(f"no RE3 graph artifacts found under {root}")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "runtime_edge_validity.csv", index=False)
    aggregate_frame = frame.loc[frame.runtime_edges > 0].copy()
    total_correlation = int(aggregate_frame.correlation_edges.sum())
    total_supported = int(aggregate_frame.supported_edges.sum())
    weighted_random = float(
        np.average(
            aggregate_frame.random_precision,
            weights=np.maximum(aggregate_frame.correlation_edges, 1),
        )
    )
    aggregate = {
        "datasets": len(aggregate_frame),
        "training_cases_with_traces": int(aggregate_frame.training_cases_with_traces.sum()),
        "correlation_edges": total_correlation,
        "supported_edges": total_supported,
        "precision": total_supported / max(total_correlation, 1),
        "random_precision": weighted_random,
        "enrichment": (total_supported / max(total_correlation, 1)) / weighted_random if weighted_random > 0 else None,
    }
    payload = {"aggregate": aggregate, "by_dataset": rows, "detail": detail}
    (root / "runtime_edge_validity.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| Dataset | Train cases | Corr. edges | Trace-supported | Precision | Random | Enrichment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        enrichment_text = "--" if row["enrichment"] is None else f"{row['enrichment']:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['training_cases_with_traces']} | {row['correlation_edges']} | "
            f"{row['supported_edges']} | {row['precision']:.3f} | {row['random_precision']:.3f} | "
            f"{enrichment_text} |"
        )
    lines.append(
        f"| Aggregate | {aggregate['training_cases_with_traces']} | {total_correlation} | {total_supported} | "
        f"{aggregate['precision']:.3f} | {weighted_random:.3f} | {aggregate['enrichment']:.2f} |"
    )
    (root / "runtime_edge_validity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "runtime_edge_validity.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
