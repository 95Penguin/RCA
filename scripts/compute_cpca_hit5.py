#!/usr/bin/env python3
"""Compute CPCA event-level H@1/H@3/H@5/MRR from an audit JSON.

The audit stores per-window ``cpca_scores``; we average them per event, rank
once, and report both covered-only and coverage-aware (missing events last).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TOTAL_EVENTS = {"ob": 55, "tt": 64}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.audit.read_text())
    ranks = []
    for case in data["cases"].values():
        events: dict[str, list[dict]] = defaultdict(list)
        for window in case["windows"]:
            if window.get("cpca_scores"):
                events[window["event"]].append(window)
        for windows in events.values():
            names = sorted({name for window in windows for name in window["cpca_scores"]})
            scores = np.mean(
                [[window["cpca_scores"].get(name, 0.0) for name in names]
                 for window in windows],
                axis=0,
            )
            root = windows[0]["root"]
            if root not in names:
                continue
            order = np.argsort(-scores)
            ranks.append(int(np.flatnonzero(np.asarray(names)[order] == root)[0]) + 1)

    def metrics(denominator: int):
        return {
            "events": denominator,
            "hit_at_1": sum(rank == 1 for rank in ranks) / denominator,
            "hit_at_3": sum(rank <= 3 for rank in ranks) / denominator,
            "hit_at_5": sum(rank <= 5 for rank in ranks) / denominator,
            "mrr": sum(1.0 / rank for rank in ranks) / denominator,
        }

    report = {
        "dataset": args.dataset,
        "covered_events": len(ranks),
        "covered_only": metrics(len(ranks)),
        "coverage_aware": metrics(TOTAL_EVENTS[args.dataset]),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
