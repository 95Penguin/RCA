#!/usr/bin/env python3
"""Check a frozen static topology against training-fold traces without editing it."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import discover_rcaeval_cases, infer_case_identity  # noqa: E402
from ltfp.protocol import fold_roles, load_split_manifest, load_topology_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--topology-manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def normalize_service(value: str, topology) -> str | None:
    raw = str(value).strip()
    normal = raw.lower().replace("_", "-")
    alias = {str(key).lower().replace("_", "-"): target for key, target in topology.aliases.items()}
    services = {service.lower().replace("_", "-"): service for service in topology.services}
    return alias.get(normal, services.get(normal))


def trace_edges(path: Path, topology) -> Counter:
    columns = ["traceID", "spanID", "parentSpanID", "serviceName"]
    frame = pd.read_csv(path, usecols=lambda value: value in columns, dtype=str)
    if not set(columns).issubset(frame.columns):
        return Counter()
    lookup = {
        (str(trace), str(span)): normalize_service(service, topology)
        for trace, span, service in frame[["traceID", "spanID", "serviceName"]].itertuples(index=False)
    }
    counts = Counter()
    for trace, parent, child_service in frame[["traceID", "parentSpanID", "serviceName"]].itertuples(index=False):
        parent_service = lookup.get((str(trace), str(parent)))
        child = normalize_service(child_service, topology)
        if parent_service and child and parent_service != child:
            counts[(parent_service, child)] += 1
    return counts


def main() -> None:
    args = parse_args()
    split = load_split_manifest(args.split_manifest)
    train_ids = fold_roles(split, args.fold)["train"]
    topology = load_topology_manifest(args.topology_manifest)
    observed = Counter()
    checked_cases = []
    for case_dir in discover_rcaeval_cases(args.case_root):
        case_id, _, _ = infer_case_identity(case_dir, dataset_root=args.case_root)
        if case_id not in train_ids or not (case_dir / "traces.csv").exists():
            continue
        observed.update(trace_edges(case_dir / "traces.csv", topology))
        checked_cases.append(case_id)
    static_directed = set(topology.edges)
    static_undirected = {frozenset(edge) for edge in topology.edges}
    observed_edges = set(observed)
    supported = [edge for edge in observed_edges if frozenset(edge) in static_undirected]
    unexpected = [edge for edge in observed_edges if frozenset(edge) not in static_undirected]
    report = {
        "topology_version": topology.version,
        "fold": args.fold,
        "scope": "training-fold-only",
        "checked_cases": sorted(checked_cases),
        "n_checked_cases": len(checked_cases),
        "static_edge_count": len(static_directed),
        "observed_edge_count": len(observed_edges),
        "observed_edge_support_fraction": len(supported) / max(len(observed_edges), 1),
        "supported_observed_edges": [
            {"source": edge[0], "target": edge[1], "spans": observed[edge]}
            for edge in sorted(supported)
        ],
        "unexpected_observed_edges": [
            {"source": edge[0], "target": edge[1], "spans": observed[edge]}
            for edge in sorted(unexpected)
        ],
        "policy": "Audit only: unexpected trace edges do not mutate the frozen topology manifest.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_checked_cases", "observed_edge_count", "observed_edge_support_fraction")}, indent=2))


if __name__ == "__main__":
    main()
