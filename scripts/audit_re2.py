#!/usr/bin/env python3
"""Audit RE2 next-minute predictions, onset timing, and simple baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset  # noqa: E402
from models.instantops import InstantOpsGNNGRU  # noqa: E402
from training.engine import fit_feature_scaler  # noqa: E402
from training.metrics import classification_metrics  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer(model: InstantOpsGNNGRU, dataset: Re2WindowDataset, ends: list[int]) -> np.ndarray:
    index = {end: i for i, end in enumerate(dataset.sample_indices)}
    loader = DataLoader(Subset(dataset, [index[end] for end in ends]), batch_size=64)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs.append(model(batch["node_features"], batch["adjacency"])["failure_logits"]
                           .softmax(dim=-1)[:, 1].numpy())
    return np.concatenate(outputs)


def at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = classification_metrics(labels, (scores >= threshold).astype(float))
    result.pop("average_precision")
    result["precision"] = result.pop("precision_at_0_5")
    result["recall"] = result.pop("recall_at_0_5")
    result["f1"] = result.pop("f1_at_0_5")
    result["threshold"] = threshold
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_metrics = json.loads((args.checkpoint.parent / "metrics.json").read_text())
    split_policy = run_metrics["run_config"].get("split_policy", "repetition")
    split_fold = run_metrics["run_config"].get("split_fold", 0)
    dataset = Re2WindowDataset(args.data_dir, split_policy=split_policy,
                               split_fold=split_fold)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scaled = copy.copy(dataset)
    mean, scale = fit_feature_scaler(scaled, dataset.predefined_splits[0])
    if not (np.allclose(mean, checkpoint["feature_mean"])
            and np.allclose(scale, checkpoint["feature_scale"])):
        raise ValueError("checkpoint scaler does not match the training split")
    model = InstantOpsGNNGRU(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    train, validation, test = dataset.predefined_splits
    val_scores = infer(model, scaled, validation)
    test_scores = infer(model, scaled, test)
    val_labels = dataset.failure_targets[validation]
    test_labels = dataset.failure_targets[test]
    candidates = sorted(set(float(score) for score in val_scores))
    selected = max(candidates, key=lambda threshold: (
        at_threshold(val_labels, val_scores, threshold)["f1"], threshold
    ))

    source = Path(json.loads((args.data_dir / "feature_schema.json").read_text())["source_labels"])
    source_rows = read_csv(source)
    by_target = {row["target_minute_utc"]: row for row in source_rows}
    case_rows = {row["case_id"]: row for row in read_csv(source.parent / "cases.csv")}
    event_rows = {row["event_id"]: row for row in read_csv(source.parent / "events.csv")}

    def current_flag(end: int) -> bool:
        row = by_target.get(dataset.timestamps[end])
        if not row or row.get("entry_request_observed", row["entry_observed"]) != "1":
            return False
        if "entry_high" in row:
            return row["entry_high"] == "1"
        return float(row["entry_p95_latency_seconds"]) >= float(row["latency_threshold_seconds"])

    val_baseline = np.asarray([current_flag(end) for end in validation], dtype=float)
    test_baseline = np.asarray([current_flag(end) for end in test], dtype=float)
    val_rolling = test_rolling = None
    if "entry_degradation_score_rolling" in dataset.feature_columns:
        score_column = dataset.feature_columns.index("entry_degradation_score_rolling")
        def rolling_flag(end: int) -> bool:
            return bool(dataset.features[end, :, score_column].max() >= 1.0)

        val_rolling = np.asarray([rolling_flag(end) for end in validation], dtype=float)
        test_rolling = np.asarray([rolling_flag(end) for end in test], dtype=float)
    with (args.checkpoint.parent / "predictions.jsonl").open() as handle:
        predictions = [json.loads(line) for line in handle]
    if not np.allclose(test_scores, [row["failure_probability"] for row in predictions], atol=1e-6):
        raise ValueError("checkpoint test predictions differ from saved predictions")
    by_event: dict[str, list[int]] = defaultdict(list)
    for row, end in enumerate(test):
        event_id = dataset.event_ids[end]
        if event_id and test_labels[row]:
            by_event[event_id].append(row)
    timing = Counter()
    onset_rank = Counter()
    first_rows = []
    by_fault: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event_id, indices in by_event.items():
        row = indices[0]
        end = test[row]
        event = event_rows[event_id]
        injection = datetime.fromisoformat(event["injection_utc"])
        start = datetime.fromisoformat(dataset.timestamps[end])
        target = start + timedelta(minutes=1)
        if target.isoformat() != event["impact_start_utc"]:
            raise ValueError(f"first usable prediction is not true onset: {event_id}")
        phase = "pre_injection" if target <= injection else (
            "mixed_injection_minute" if start < injection else "post_injection")
        timing[phase] += 1
        fault = case_rows[event["case_id"]]["fault_type"]
        by_fault[fault]["events"] += 1
        by_fault[fault]["model_hits_0_5"] += int(test_scores[row] >= 0.5)
        by_fault[fault]["baseline_hits"] += int(test_baseline[row] >= 0.5)
        if test_rolling is not None:
            by_fault[fault]["rolling_hits"] += int(test_rolling[row] >= 0.5)
        truth = set(predictions[row]["true_services"])
        ranks = [index + 1 for index, candidate in enumerate(predictions[row]["ranking"])
                 if candidate["service"] in truth]
        if not ranks:
            raise ValueError(f"missing onset root label: {event_id}")
        rank = min(ranks)
        onset_rank["hit_at_1"] += int(rank == 1)
        onset_rank["hit_at_3"] += int(rank <= 3)
        onset_rank["joint_at_0_5"] += int(rank == 1 and test_scores[row] >= 0.5)
        onset_rank["joint_at_selected"] += int(rank == 1 and test_scores[row] >= selected)
        first_rows.append({
            "event_id": event_id, "fault_type": fault, "input_minute_utc": start.isoformat(),
            "impact_start_utc": target.isoformat(), "injection_utc": injection.isoformat(),
            "input_phase": phase, "model_score": float(test_scores[row]),
            "current_latency_baseline": bool(test_baseline[row]),
            "rolling_latency_baseline": (bool(test_rolling[row]) if test_rolling is not None else None),
            "onset_root_rank": rank,
            "event_duration_minutes": int(event["duration_minutes"]),
        })
    first_indices = [indices[0] for indices in by_event.values()]
    ongoing = [i for i, end in enumerate(test) if test_labels[i]
               and i not in first_indices]
    report = {
        "protocol": {
            "prediction": "five observed minutes through input minute predict next complete minute",
            "split": dataset.split_report,
            "threshold_selection": "maximize validation-window F1; test labels never used",
            "case_calibrated_baseline": "uses injection-aligned pre-case p95 from source labels; diagnostic upper reference, not fully online",
            "rolling_baseline": "entry-operation p95 exceeds both dataset ratio and 0.1-second increase relative to median of 3+ observed minutes in preceding 5; only completed current/prior spans",
            "onset_timing": "pre if entire input minute ends before injection; mixed if injection occurs inside input minute",
        },
        "validation": {
            "model_at_0_5": classification_metrics(val_labels, val_scores),
            "model_at_selected_threshold": at_threshold(val_labels, val_scores, selected),
            "current_latency_baseline": classification_metrics(val_labels, val_baseline),
            "rolling_latency_baseline": (
                classification_metrics(val_labels, val_rolling) if val_rolling is not None else None
            ),
        },
        "test": {
            "model_at_0_5": classification_metrics(test_labels, test_scores),
            "model_at_selected_threshold": at_threshold(test_labels, test_scores, selected),
            "current_latency_baseline": classification_metrics(test_labels, test_baseline),
            "rolling_latency_baseline": (
                classification_metrics(test_labels, test_rolling) if test_rolling is not None else None
            ),
            "ongoing_positive_minutes": len(ongoing),
            "ongoing_model_recall_at_0_5": float((test_scores[ongoing] >= 0.5).mean()),
            "ongoing_baseline_recall": float(test_baseline[ongoing].mean()),
            "ongoing_rolling_recall": (
                float(test_rolling[ongoing].mean()) if test_rolling is not None else None
            ),
            "onset_events": len(first_rows),
            "onset_timing": dict(timing),
            "onset_model_recall_at_0_5": float((test_scores[first_indices] >= 0.5).mean()),
            "onset_model_recall_selected": float((test_scores[first_indices] >= selected).mean()),
            "onset_baseline_recall": float(test_baseline[first_indices].mean()),
            "onset_rolling_recall": (
                float(test_rolling[first_indices].mean()) if test_rolling is not None else None
            ),
            "onset_root_hit_at_1": onset_rank["hit_at_1"] / len(first_rows),
            "onset_root_hit_at_3": onset_rank["hit_at_3"] / len(first_rows),
            "onset_joint_hit_at_1_at_0_5": onset_rank["joint_at_0_5"] / len(first_rows),
            "onset_joint_hit_at_1_selected": onset_rank["joint_at_selected"] / len(first_rows),
            "onset_by_fault_type": {key: dict(value) for key, value in sorted(by_fault.items())},
        },
        "events": first_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"validation": report["validation"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
