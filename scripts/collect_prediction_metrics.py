#!/usr/bin/env python3
"""Collect prediction Precision/Recall/F1 for the paper table.

Baselines: per-fold metrics.json (frameworks use top-level fields, specialists
use ``failure_prediction``).  OCSID: per-window test probabilities are matched
to the next-minute labels, then P/R/F1 are computed at each fold's calibrated
threshold.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/baselines/comparison_2026-11-19_seed42"
PANELS = {ds: ROOT / f"artifacts/re2_{ds}_panel_fair_v1" for ds in ("ob", "tt")}


def _epoch(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def _labels(dataset: str) -> dict[tuple[str, int], int]:
    panel = PANELS[dataset]
    minute_to_case = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((panel / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    labels = {}
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = minute_to_case.get(row["input_minute_utc"])
            if case is None:
                continue
            labels[case, _epoch(row["input_minute_utc"])] = int(row["failure_next_minute"])
    return labels


def _prf(labels: list[int], probabilities: list[float], threshold: float):
    predicted = [1 if p >= threshold else 0 for p in probabilities]
    tp = sum(1 for y, p in zip(labels, predicted) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, predicted) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predicted) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def baseline_rows():
    framework = ("ltfp", "instantops", "eadro", "art")
    specialist = ("anofusion", "hades", "maat")
    rows = []
    for dataset in ("ob", "tt"):
        for method in framework + specialist:
            folder = "frameworks" if method in framework else "specialists"
            paths = sorted((BASE / folder).glob(f"{dataset}_{method}_fold*_seed42/metrics.json"))
            if not paths:
                continue
            metrics = []
            for path in paths:
                data = json.loads(path.read_text())
                block = data.get("failure_prediction") or data
                at = block["at_validation_threshold"]
                metrics.append((
                    block["average_precision"],
                    at["precision"], at["recall"], at["f1"],
                ))
            values = np.asarray(metrics, dtype=float)
            rows.append({
                "dataset": dataset, "method": method, "folds": len(metrics),
                "ap": float(values[:, 0].mean()),
                "precision": float(values[:, 1].mean()),
                "recall": float(values[:, 2].mean()),
                "f1": float(values[:, 3].mean()),
            })
    return rows


def ocsid_rows():
    rows = []
    for dataset in ("ob", "tt"):
        labels = _labels(dataset)
        per_seed = defaultdict(list)
        for seed in (42, 123, 2026):
            path = ROOT / f"artifacts/ocsid_joint_{dataset}_seed{seed}_v3.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            for fold in data["folds"]:
                threshold = fold["calibration"]["threshold"]
                y, p = [], []
                for prediction in fold["test_predictions"]:
                    label = labels.get((prediction["case"], int(prediction["minute"])))
                    if label is None:
                        continue
                    y.append(label)
                    p.append(float(prediction["probability"]))
                if not y:
                    continue
                precision, recall, f1 = _prf(y, p, threshold)
                per_seed[seed].append((precision, recall, f1))
        if not per_seed:
            continue
        seed_means = np.asarray([np.mean(values, axis=0) for values in per_seed.values()])
        mean = seed_means.mean(axis=0)
        sd = seed_means.std(axis=0, ddof=1) if len(seed_means) > 1 else np.zeros(3)
        rows.append({
            "dataset": dataset, "method": "ocsid", "seeds": len(seed_means),
            "precision": float(mean[0]), "precision_sd": float(sd[0]),
            "recall": float(mean[1]), "recall_sd": float(sd[1]),
            "f1": float(mean[2]), "f1_sd": float(sd[2]),
        })
    return rows


def main() -> int:
    rows = baseline_rows() + ocsid_rows()
    out = ROOT / "artifacts/paper_prediction_metrics_v1.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"{'ds':3s} {'method':12s} {'P':>6} {'R':>6} {'F1':>6} {'AP':>6}")
    for row in rows:
        ap = row.get("ap")
        print(f"{row['dataset']:3s} {row['method']:12s} "
              f"{row['precision']:6.3f} {row['recall']:6.3f} "
              f"{row['f1']:6.3f} {ap if ap is not None else float('nan'):6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
