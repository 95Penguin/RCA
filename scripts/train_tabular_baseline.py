#!/usr/bin/env python3
"""Tabular GBDT baseline (scikit-learn HistGradientBoosting).

XGBoost/LightGBM were not loadable in this environment (missing libomp), so this
uses the same GBDT family from scikit-learn.  Features: 5-minute panel tensor
flattened + last-minute operation features + operation mask.  Same deployment
folds and validation-threshold F1 protocol as the neural baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset  # noqa: E402
from train_ocsid_joint import PANELS, _build_arrays  # noqa: E402


def prf(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_threshold(labels, probabilities):
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        f1 = prf(labels, probabilities, threshold)[2]
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset, operations, ends, X, A, O, M, y, node, events = _build_arrays(
        args.dataset, PANELS[args.dataset], 1
    )
    features = np.concatenate([
        X.reshape(len(ends), -1),
        O[:, -1].reshape(len(ends), -1),
        M[:, -1],
    ], axis=1).astype(np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    folds = []
    for fold in range(5):
        split = Re2WindowDataset(
            PANELS[args.dataset], history=5, split_policy="deployment",
            split_fold=fold, horizon_minutes=1,
        )
        positions = {end: index for index, end in enumerate(ends)}
        train = [positions[end] for end in split.predefined_splits[0]]
        validation = [positions[end] for end in split.predefined_splits[1]]
        test = [positions[end] for end in split.predefined_splits[2]]
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.15,
            random_state=args.seed,
        )
        model.fit(features[train], y[train])
        val_prob = model.predict_proba(features[validation])[:, 1]
        threshold = best_threshold(y[validation].astype(int), val_prob)
        test_prob = model.predict_proba(features[test])[:, 1]
        precision, recall, f1 = prf(y[test].astype(int), test_prob, threshold)
        folds.append({
            "fold": fold,
            "average_precision": float(average_precision_score(y[test], test_prob)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
        })

    def mean(key):
        values = [fold[key] for fold in folds]
        return float(np.mean(values)), float(np.std(values, ddof=1))

    report = {
        "dataset": args.dataset,
        "method": "tabular-gbdt",
        "seed": args.seed,
        "folds": folds,
        "summary": {
            key: {"mean": mean(key)[0], "sd": mean(key)[1]}
            for key in ("average_precision", "precision", "recall", "f1")
        },
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
