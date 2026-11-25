#!/usr/bin/env python3
"""MicroRCA-adapted baseline: service anomaly scores + personalized PageRank.

MicroRCA (Wu et al., NOMS 2020) localizes root causes by combining per-service
anomaly scores with the service dependency graph.  This adaptation uses:

* robust per-service z-scores over the first 10 minutes of the case;
* the audited symmetric service adjacency at the decision minute;
* personalized PageRank with restart vector = normalized anomaly scores.

It is deliberately distinct from MicroRank (trace PageRank): MicroRCA uses
service-level metrics and the service graph, not request-level trace scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = {ds: ROOT / f"artifacts/re2_{ds}_panel_fair_v1" for ds in ("ob", "tt")}
SIGNAL_FEATURES = (
    "metric_cpu", "metric_mem", "metric_diskio", "metric_socket",
    "trace_duration_mean_ms", "log_error_count", "entry_p95_ms",
)


def epoch(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def load_panel(dataset: str):
    panel = PANELS[dataset]
    schema = json.loads((panel / "feature_schema.json").read_text())
    services = schema["services"]
    columns = schema["feature_columns"]
    with np.load(panel / "service_adjacency_1m.npz", allow_pickle=False) as graph:
        timestamps = graph["timestamps"].copy()
        adjacency = graph["adjacency"].copy()
    index = {int(t): i for i, t in enumerate(timestamps)}
    features = np.zeros((len(timestamps), len(services), len(columns)), dtype=np.float64)
    service_index = {name: i for i, name in enumerate(services)}
    with (panel / "service_features_1m.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            moment = index.get(epoch(row["timestamp_utc"]))
            service = service_index.get(row["service"])
            if moment is None or service is None:
                continue
            features[moment, service] = [float(row[column]) for column in columns]

    case_by_time = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((panel / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    labels = defaultdict(dict)
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = case_by_time.get(row["input_minute_utc"])
            if case is None:
                continue
            key = epoch(row["input_minute_utc"])
            entry = labels[case].setdefault(
                key, {"failure": 0, "root": None, "event": row.get("event_id") or ""}
            )
            if row["failure_next_minute"] == "1":
                entry["failure"] = 1
            if row["root_cause_next_minute"] == "1" and row["root_label_valid"] == "1":
                entry["root"] = row["service"]
    return services, columns, timestamps, adjacency, features, case_by_time, labels


def personalized_pagerank(adjacency: np.ndarray, seed: np.ndarray, alpha: float = 0.85):
    graph = adjacency + adjacency.T
    row_sum = graph.sum(axis=1, keepdims=True)
    graph = np.divide(graph, row_sum, out=np.zeros_like(graph), where=row_sum > 0)
    graph += np.eye(len(seed)) * (row_sum.ravel() == 0)
    probability = seed.copy()
    for _ in range(50):
        probability = alpha * graph.T.dot(probability) + (1.0 - alpha) * seed
    return probability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    services, columns, timestamps, adjacency, features, case_by_time, labels = load_panel(args.dataset)
    selected = [columns.index(name) for name in SIGNAL_FEATURES if name in columns]
    index = {int(t): i for i, t in enumerate(timestamps)}

    events: dict[str, list[tuple[np.ndarray, str]]] = defaultdict(list)
    for case, rows in labels.items():
        case_minutes = [epoch(t) for t, c in case_by_time.items() if c == case]
        if not case_minutes:
            continue
        start = min(case_minutes)
        normal = [index[t] for t in case_minutes if t < start + 600 and t in index]
        if len(normal) < 2:
            normal = [index[t] for t in case_minutes if t in index]
        median = np.median(features[normal], axis=0)
        mad = np.median(np.abs(features[normal] - median), axis=0) * 1.4826
        mad = np.maximum(mad, 1e-6)
        latency = [columns.index(name) for name in ("trace_duration_mean_ms", "entry_p95_ms")
                   if name in columns]

        for minute, row in rows.items():
            if row["failure"] != 1 or not row["root"] or minute not in index:
                continue
            moment = index[minute]
            z = np.abs((features[moment] - median) / mad)
            anomaly = z[:, latency].max(axis=1) if latency else z[:, selected].max(axis=1)
            if not np.any(anomaly > 0):
                anomaly = z[:, selected].max(axis=1)
            total = anomaly.sum()
            seed = anomaly / total if total > 0 else np.full(len(services), 1.0 / len(services))
            probability = personalized_pagerank(adjacency[moment], seed, args.alpha)
            events[row["event"]].append((probability, row["root"]))

    ranks = []
    for items in events.values():
        probability = np.mean([item[0] for item in items], axis=0)
        root = items[0][1]
        order = np.argsort(-probability)
        rank = int(np.flatnonzero(np.asarray(services)[order] == root)[0]) + 1
        ranks.append(rank)

    report = {
        "dataset": args.dataset,
        "method": "micro_rca_adapted",
        "events": len(ranks),
        "hit_at_1": float(np.mean([r == 1 for r in ranks])) if ranks else None,
        "hit_at_3": float(np.mean([r <= 3 for r in ranks])) if ranks else None,
        "hit_at_5": float(np.mean([r <= 5 for r in ranks])) if ranks else None,
        "mrr": float(np.mean([1.0 / r for r in ranks])) if ranks else None,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
