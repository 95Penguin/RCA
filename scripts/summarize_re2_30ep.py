#!/usr/bin/env python3
"""Summarize paired RE2 30-epoch deployment-holdout runs and TT onset errors."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (42, 123, 2026)
FOLDS = range(5)
MODELS = ("gru", "gcn_gru")
VARIANTS = ("legacy", "fair")


def run_dir(root: Path, dataset: str, model: str, variant: str,
            fold: int, seed: int) -> Path:
    return (root / f"re2_{dataset}_{model}_deployment_fold{fold}_{variant}"
            f"_30ep_seed{seed}_v3")


def first_event_predictions(root: Path, model: str, variant: str,
                            seed: int) -> dict[str, dict]:
    by_event = {}
    for fold in FOLDS:
        path = run_dir(root, "tt", model, variant, fold, seed) / "predictions.jsonl"
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                if row["failure_target"] and row["event_id"]:
                    by_event.setdefault(row["event_id"], row)
    return by_event


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "seed_sd": statistics.stdev(values)}


def summarize(root: Path, tt_events: Path) -> dict:
    result: dict = {
        "protocol": {
            "datasets": ["RE2-OB", "RE2-TT"],
            "models": list(MODELS), "features": list(VARIANTS),
            "seeds": list(SEEDS), "folds": list(FOLDS),
            "epochs": 30, "device": "cpu", "checkpoint": "minimum validation loss",
            "split": "deployment holdout balanced by raw case count",
            "f1_ap_aggregation": "mean of five test folds per seed, then mean/SD of three seeds",
            "event_aggregation": "each event occurs in one test fold per seed",
        },
        "results": {}, "tt_error_analysis": {},
    }
    for dataset in ("ob", "tt"):
        dataset_result = {}
        for model in MODELS:
            for variant in VARIANTS:
                seed_results = []
                for seed in SEEDS:
                    folds = []
                    for fold in FOLDS:
                        folder = run_dir(root, dataset, model, variant, fold, seed)
                        audit = json.loads((folder / "audit.json").read_text())["test"]
                        metrics = json.loads((folder / "metrics.json").read_text())
                        best_epoch = min(metrics["training_history"],
                                         key=lambda row: row["val_loss"])["epoch"]
                        folds.append({
                            "fold": fold,
                            "test_windows": audit["model_at_0_5"]["samples"],
                            "positive_windows": audit["model_at_0_5"]["positive_samples"],
                            "f1_0_5": audit["model_at_0_5"]["f1_at_0_5"],
                            "ap": audit["model_at_0_5"]["average_precision"],
                            "validation_selected_f1": audit["model_at_selected_threshold"]["f1"],
                            "validation_selected_threshold": audit["model_at_selected_threshold"]["threshold"],
                            "onset_events": audit["onset_events"],
                            "onset_alarm_hits": round(audit["onset_model_recall_at_0_5"]
                                                      * audit["onset_events"]),
                            "onset_root_top1_hits": round(audit["onset_root_hit_at_1"]
                                                          * audit["onset_events"]),
                            "joint_onset_top1_hits": round(audit["onset_joint_hit_at_1_at_0_5"]
                                                           * audit["onset_events"]),
                            "joint_validation_threshold_hits": round(
                                audit["onset_joint_hit_at_1_selected"] * audit["onset_events"]),
                            "best_epoch": best_epoch,
                        })
                    seed_results.append({
                        "seed": seed, "folds": folds,
                        "mean_f1_0_5": statistics.mean(f["f1_0_5"] for f in folds),
                        "mean_ap": statistics.mean(f["ap"] for f in folds),
                        "mean_validation_selected_f1": statistics.mean(
                            f["validation_selected_f1"] for f in folds),
                        "onset_events": sum(f["onset_events"] for f in folds),
                        "onset_alarm_hits": sum(f["onset_alarm_hits"] for f in folds),
                        "onset_root_top1_hits": sum(f["onset_root_top1_hits"] for f in folds),
                        "joint_onset_top1_hits": sum(f["joint_onset_top1_hits"] for f in folds),
                        "joint_validation_threshold_hits": sum(
                            f["joint_validation_threshold_hits"] for f in folds),
                    })
                dataset_result[f"{model}_{variant}"] = {
                    "seeds": seed_results,
                    "f1_0_5": mean_sd([s["mean_f1_0_5"] for s in seed_results]),
                    "ap": mean_sd([s["mean_ap"] for s in seed_results]),
                    "validation_selected_f1": mean_sd([
                        s["mean_validation_selected_f1"] for s in seed_results]),
                    "joint_onset_hits_per_64_or_55_events": [
                        s["joint_onset_top1_hits"] for s in seed_results],
                }
        result["results"]["RE2-" + dataset.upper()] = dataset_result

    with tt_events.open(newline="") as source:
        metadata = {row["event_id"]: row for row in csv.DictReader(source)}
    for model in MODELS:
        by_fault = defaultdict(Counter)
        by_root = defaultdict(Counter)
        paired = Counter()
        for seed in SEEDS:
            old = first_event_predictions(root, model, "legacy", seed)
            new = first_event_predictions(root, model, "fair", seed)
            if set(old) != set(new) or set(old) != set(metadata):
                raise ValueError("TT event sets differ across variants or source labels")
            for event_id in old:
                a, b = old[event_id], new[event_id]
                old_alarm = a["failure_probability"] >= 0.5
                new_alarm = b["failure_probability"] >= 0.5
                old_root = a["ranking"][0]["service"] in a["true_services"]
                new_root = b["ranking"][0]["service"] in b["true_services"]
                paired["alarm_lost"] += old_alarm and not new_alarm
                paired["alarm_gained"] += new_alarm and not old_alarm
                paired["root_top1_lost"] += old_root and not new_root
                paired["root_top1_gained"] += new_root and not old_root
                paired["event_seed_pairs"] += 1
                for dimension, table in (("fault_type", by_fault),
                                         ("root_service", by_root)):
                    item = table[metadata[event_id][dimension]]
                    item["event_seed_pairs"] += 1
                    item["legacy_alarm"] += old_alarm
                    item["legacy_root_top1"] += old_root
                    item["legacy_joint"] += old_alarm and old_root
                    item["fair_alarm"] += new_alarm
                    item["fair_root_top1"] += new_root
                    item["fair_joint"] += new_alarm and new_root
        result["tt_error_analysis"][model] = {
            "paired_changes": dict(paired),
            "by_fault_type": {key: dict(value) for key, value in sorted(by_fault.items())},
            "by_root_service": {key: dict(value) for key, value in sorted(by_root.items())},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--tt-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.artifacts, args.tt_events), indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
