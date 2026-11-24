#!/usr/bin/env python3
"""Recompute event-level RCA for every baseline from saved predictions.

The specialist/framework comparison reports window-level RCA.  Each run also
saves ``predictions.jsonl`` with per-window ``root_scores``, so the same
predictions can be aggregated per failure event (mean score over the event's
windows, then rank once).  This makes all methods comparable to OCSID/CPCA at
the event level.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/baseline_event_metrics.py \
  --dataset ob --output artifacts/baselines_event_metrics_ob_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "artifacts/baselines/comparison_2026-11-19_seed42"
PANELS = {name: ROOT / f"artifacts/re2_{name}_panel_fair_v1" for name in ("ob", "tt")}


def _event_map(dataset: str):
    panel = PANELS[dataset]
    mapping = {}
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_id"):
                mapping[(row["case_id"] if "case_id" in row else "", row["input_minute_utc"])] = row["event_id"]
    # next_minute_labels has no case_id; join through case_minutes.
    minute_to_case = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((panel / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    mapping = {}
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("event_id"):
                continue
            case = minute_to_case.get(row["input_minute_utc"])
            if case:
                mapping[case, row["input_minute_utc"]] = row["event_id"]
    return mapping


def _services(dataset: str):
    schema = json.loads((PANELS[dataset] / "feature_schema.json").read_text())
    return schema["services"]


def _event_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event"]].append(row)
    ranks = []
    for event, items in grouped.items():
        scores = np.mean([item["scores"] for item in items], axis=0)
        root = items[0]["root"]
        order = np.argsort(-scores)
        names = items[0]["services"]
        rank = int(np.flatnonzero(np.asarray(names)[order] == root)[0]) + 1
        ranks.append(rank)
    if not ranks:
        return {"events": 0, "hit_at_1": None, "hit_at_3": None, "mrr": None}
    return {
        "events": len(ranks),
        "hit_at_1": float(np.mean([r == 1 for r in ranks])),
        "hit_at_3": float(np.mean([r <= 3 for r in ranks])),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = _event_map(args.dataset)
    services = _services(args.dataset)
    report = {}
    for group in ("frameworks", "specialists"):
        for directory in sorted((COMPARISON / group).glob(f"{args.dataset}_*_fold*_seed42")):
            name = directory.name[len(args.dataset) + 1:].split("_fold")[0]
            rows = []
            for fold_dir in sorted((COMPARISON / group).glob(f"{args.dataset}_{name}_fold*_seed42")):
                path = fold_dir / "predictions.jsonl"
                if not path.is_file():
                    continue
                for line in path.open():
                    row = json.loads(line)
                    if row.get("failure_target") != 1:
                        continue
                    if not row.get("root_scores") or not row.get("root_label_valid"):
                        continue
                    targets = row.get("root_targets") or []
                    if not targets or sum(targets) <= 0:
                        continue
                    event = events.get((row["case_id"], row["input_minute"]))
                    if event is None:
                        continue
                    rows.append({
                        "event": event,
                        "root": services[int(np.argmax(targets))],
                        "scores": np.asarray(row["root_scores"], dtype=np.float64),
                        "services": services,
                    })
            if rows:
                report[name] = _event_metrics(rows)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
