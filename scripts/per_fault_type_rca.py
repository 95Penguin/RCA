#!/usr/bin/env python3
"""Per-fault-type event-level RCA: CPCA vs a baseline (default LTFP)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "artifacts/baselines/comparison_2026-11-19_seed42"
PANELS = {ds: ROOT / f"artifacts/re2_{ds}_panel_fair_v1" for ds in ("ob", "tt")}


def fault_type(case: str) -> str:
    return case.rsplit("_", 2)[-2]


def rank_of(names: list[str], scores: np.ndarray, root: str) -> int:
    if not names or root not in names:
        return len(names) + 1
    order = np.argsort(-scores)
    return int(np.flatnonzero(np.asarray(names)[order] == root)[0]) + 1


def baseline_events(dataset: str, method: str) -> dict[str, tuple[str, int]]:
    events = {}
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
            grouped[row["event_id"]].append(
                (row["case_id"], names, np.asarray(scores, dtype=float), root)
            )
        for event, items in grouped.items():
            names = items[0][1]
            scores = np.mean([item[2] for item in items], axis=0)
            events[event] = (items[0][0], rank_of(names, scores, items[0][3]))
    return events


def cpca_events(audit: Path) -> dict[str, tuple[str, int]]:
    data = json.loads(audit.read_text())
    events = {}
    for case_id, case in data["cases"].items():
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
            events[event] = (case_id, rank_of(names, scores, windows[0]["root"]))
    return events


def metrics(ranks: list[int]) -> dict:
    if not ranks:
        return {"events": 0, "hit_at_1": None, "hit_at_3": None, "mrr": None}
    return {
        "events": len(ranks),
        "hit_at_1": float(np.mean([r == 1 for r in ranks])),
        "hit_at_3": float(np.mean([r <= 3 for r in ranks])),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--baseline", default="ltfp")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schema = json.loads((PANELS[args.dataset] / "feature_schema.json").read_text())
    worst = len(schema["services"]) + 1
    base = baseline_events(args.dataset, args.baseline)
    cpca = cpca_events(args.audit)

    by_type = defaultdict(lambda: {"cpca": [], "base": []})
    for event, (case, base_rank) in base.items():
        kind = fault_type(case)
        by_type[kind]["base"].append(base_rank)
        by_type[kind]["cpca"].append(cpca.get(event, (case, worst))[1])

    report = {}
    for kind, values in sorted(by_type.items()):
        report[kind] = {
            "cpca": metrics(values["cpca"]),
            "baseline": metrics(values["base"]),
        }
    args.output.write_text(json.dumps(report, indent=2))
    print(f"{'fault':10s} {'n':>4} {'CPCA H1':>8} {'CPCA MRR':>9} "
          f"{'LTFP H1':>8} {'LTFP MRR':>9}")
    for kind, values in report.items():
        c, b = values["cpca"], values["baseline"]
        print(f"{kind:10s} {c['events']:>4} {c['hit_at_1']:>8.3f} {c['mrr']:>9.3f} "
              f"{b['hit_at_1']:>8.3f} {b['mrr']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
