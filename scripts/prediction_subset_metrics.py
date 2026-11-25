#!/usr/bin/env python3
"""Prediction metrics on onset / persistence / healthy subsets.

For h=1, each window is labelled by (current state y_t, target state y_{t+1}):
onset = (0,1), persistence = (1,1), healthy = (0,0), recovery = (1,0).
Reports detection rate (recall at the validation threshold) and onset-vs-healthy AP.
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
COMPARISON = ROOT / "artifacts/baselines/comparison_2026-11-19_seed42"


def epoch(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def failure_at(dataset: str):
    panel = PANELS[dataset]
    case_by_time = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((panel / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    values = {}
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = case_by_time.get(row["input_minute_utc"])
            if case:
                values[(case, epoch(row["target_minute_utc"]))] = int(row["failure_next_minute"])
    return values


def ap(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall = tp / labels.sum()
    return float(np.sum(precision * np.diff(np.r_[0, recall])))


def summarize(rows):
    onset = [(y, p, thr) for y, p, thr in rows if y == (0, 1)]
    persist = [(y, p, thr) for y, p, thr in rows if y == (1, 1)]
    healthy = [(y, p, thr) for y, p, thr in rows if y == (0, 0)]
    recovery = [(y, p, thr) for y, p, thr in rows if y == (1, 0)]
    def rate(items):
        return float(np.mean([p >= thr for _, p, thr in items])) if items else None
    onset_ap = ap([1] * len(onset) + [0] * len(healthy),
                  [p for _, p, _ in onset] + [p for _, p, _ in healthy])
    return {
        "onset_events": len(onset),
        "onset_recall": rate(onset),
        "onset_ap_vs_healthy": onset_ap,
        "persistence_events": len(persist),
        "persistence_recall": rate(persist),
        "healthy_events": len(healthy),
        "healthy_fpr": rate(healthy),
        "recovery_events": len(recovery),
        "recovery_alert_rate": rate(recovery),
    }


def ocsid_rows(dataset, labels):
    rows = []
    path = ROOT / f"artifacts/ocsid_joint_{dataset}_seed42_auxroot_v3.json"
    data = json.loads(path.read_text())
    for fold in data["folds"]:
        threshold = fold["calibration"]["threshold"]
        for prediction in fold["test_predictions"]:
            key = (prediction["case"], int(prediction["minute"]))
            current = labels.get(key)
            target = labels.get((key[0], key[1] + 60))
            if current is None or target is None:
                continue
            rows.append(((current, target), float(prediction["probability"]), threshold))
    return summarize(rows)


def baseline_rows(dataset, method, labels):
    rows = []
    for fold in range(5):
        folder = COMPARISON / "frameworks" / f"{dataset}_{method}_fold{fold}_seed42"
        metrics_path = folder / "metrics.json"
        path = folder / "predictions.jsonl"
        if not metrics_path.is_file() or not path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        threshold = metrics["at_validation_threshold"]["threshold"]
        for line in path.open():
            row = json.loads(line)
            key = (row["case_id"], epoch(row["input_minute"]))
            current = labels.get(key)
            target = labels.get((key[0], key[1] + 60))
            if current is None or target is None:
                continue
            rows.append(((current, target), float(row["risk_score"]), threshold))
    return summarize(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = failure_at(args.dataset)
    report = {
        "dataset": args.dataset,
        "ocsid": ocsid_rows(args.dataset, labels),
        "ltfp": baseline_rows(args.dataset, "ltfp", labels),
        "instantops": baseline_rows(args.dataset, "instantops", labels),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
