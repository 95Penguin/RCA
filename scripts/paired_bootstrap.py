#!/usr/bin/env python3
"""Event-level paired bootstrap: CPCA vs a baseline (default LTFP).

Ranks are computed per event (mean scores over the event's windows, rank once).
Events missing from the CPCA audit are assigned the worst rank, so the CPCA
side is not advantaged by coverage.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "artifacts/baselines/comparison_2026-11-19_seed42"
PANELS = {ds: ROOT / f"artifacts/re2_{ds}_panel_fair_v1" for ds in ("ob", "tt")}


def _rank(names: list[str], scores: np.ndarray, root: str) -> int:
    if not names or root not in names:
        return len(names) + 1
    order = np.argsort(-scores)
    return int(np.flatnonzero(np.asarray(names)[order] == root)[0]) + 1


def baseline_ranks(dataset: str, method: str) -> dict[str, int]:
    ranks = {}
    for fold in range(5):
        path = COMPARISON / "frameworks" / f"{dataset}_{method}_fold{fold}_seed42" / "predictions.jsonl"
        if not path.is_file():
            continue
        grouped = defaultdict(list)
        for line in path.open():
            row = json.loads(line)
            if row.get("failure_target") != 1 or not row.get("event_id"):
                continue
            names = row.get("ranked_services") or []
            scores = row.get("root_scores") or []
            targets = row.get("root_targets") or []
            if not names or not scores or not targets:
                continue
            root = names[int(np.argmax(targets))]
            grouped[row["event_id"]].append((names, np.asarray(scores, dtype=float), root))
        for event, items in grouped.items():
            names = items[0][0]
            scores = np.mean([item[1] for item in items], axis=0)
            ranks[event] = _rank(names, scores, items[0][2])
    return ranks


def cpca_ranks(audit: Path) -> dict[str, int]:
    data = json.loads(audit.read_text())
    ranks = {}
    for case in data["cases"].values():
        grouped = defaultdict(list)
        for window in case["windows"]:
            if window.get("cpca_scores"):
                grouped[window["event"]].append(window)
        for event, windows in grouped.items():
            names = sorted({name for window in windows for name in window["cpca_scores"]})
            scores = np.mean(
                [[window["cpca_scores"].get(name, 0.0) for name in names]
                 for window in windows],
                axis=0,
            )
            ranks[event] = _rank(names, scores, windows[0]["root"])
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--baseline", default="ltfp")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schema = json.loads((PANELS[args.dataset] / "feature_schema.json").read_text())
    worst = len(schema["services"]) + 1
    base = baseline_ranks(args.dataset, args.baseline)
    cpca = cpca_ranks(args.audit)
    events = sorted(base)
    cpca_ranks_aligned = np.asarray([cpca.get(event, worst) for event in events])
    base_ranks = np.asarray([base[event] for event in events])

    def stats(ranks):
        return float(np.mean(ranks == 1)), float(np.mean(1.0 / ranks))

    rng = np.random.default_rng(42)
    n = len(events)
    diffs_h1, diffs_mrr = [], []
    for _ in range(args.boot):
        idx = rng.integers(0, n, n)
        c = cpca_ranks_aligned[idx]
        b = base_ranks[idx]
        diffs_h1.append(float(np.mean(c == 1) - np.mean(b == 1)))
        diffs_mrr.append(float(np.mean(1.0 / c) - np.mean(1.0 / b)))
    diffs_h1 = np.asarray(diffs_h1)
    diffs_mrr = np.asarray(diffs_mrr)

    c_h1, c_mrr = stats(cpca_ranks_aligned)
    b_h1, b_mrr = stats(base_ranks)
    report = {
        "dataset": args.dataset,
        "baseline": args.baseline,
        "events": n,
        "cpca": {"hit_at_1": c_h1, "mrr": c_mrr},
        "baseline": {"hit_at_1": b_h1, "mrr": b_mrr},
        "delta": {"hit_at_1": c_h1 - b_h1, "mrr": c_mrr - b_mrr},
        "bootstrap": {
            "samples": args.boot,
            "hit_at_1_ci95": [float(np.percentile(diffs_h1, 2.5)),
                              float(np.percentile(diffs_h1, 97.5))],
            "mrr_ci95": [float(np.percentile(diffs_mrr, 2.5)),
                         float(np.percentile(diffs_mrr, 97.5))],
            "p_hit_at_1_le_0": float(np.mean(diffs_h1 <= 0)),
            "p_mrr_le_0": float(np.mean(diffs_mrr <= 0)),
        },
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
