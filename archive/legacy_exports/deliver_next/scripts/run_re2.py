#!/usr/bin/env python3
"""Label, prepare and train the RCAEval RE2-OB five-to-one-minute pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2_labels import build_re2_ob_labels  # noqa: E402


def label_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-entry-spans", type=int, default=100)
    parser.add_argument("--latency-ratio", type=float, default=1.5)
    parser.add_argument("--min-latency-increase-seconds", type=float, default=0.1)
    parser.add_argument("--consecutive-minutes", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(build_re2_ob_labels(
        args.archive, args.output_dir,
        min_entry_spans=args.min_entry_spans,
        latency_ratio=args.latency_ratio,
        min_latency_increase_seconds=args.min_latency_increase_seconds,
        consecutive_minutes=args.consecutive_minutes,
    ), indent=2))
    return 0


def prepare_main(argv: list[str]) -> int:
    from data.re2 import prepare_re2_ob

    parser = argparse.ArgumentParser(description="Build RE2-OB minute features and graph")
    parser.add_argument("archive", type=Path)
    parser.add_argument("labels", type=Path, help="Output of the RE2 label command")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_re2_ob(args.archive, args.labels, args.output_dir), indent=2))
    return 0


def train_main(argv: list[str]) -> int:
    import csv
    import numpy as np

    from data.re2 import Re2WindowDataset
    from training.experiment import run_training
    from training.metrics import classification_metrics

    parser = argparse.ArgumentParser(description="Train on case-separated RE2-OB windows")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--model", choices=("gcn-gru", "gru"), default="gcn-gru")
    parser.add_argument("--rca-method", choices=("node-head", "masking"), default="node-head")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.history < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("history, epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.node_loss_weight < 0:
        parser.error("learning-rate must be positive and node-loss-weight nonnegative")
    dataset = Re2WindowDataset(args.data_dir, history=args.history)
    definition = json.loads((args.data_dir / "label_schema.json").read_text())
    metrics = run_training(dataset, args, str(args.data_dir / "service_adjacency_1m.npz"), definition)
    source = json.loads((args.data_dir / "feature_schema.json").read_text())["source_labels"]
    with Path(source).open(newline="", encoding="utf-8") as handle:
        by_target = {row["target_minute_utc"]: row for row in csv.DictReader(handle)}
    test_ends = dataset.predefined_splits[2]
    flags = []
    missing = 0
    for end in test_ends:
        row = by_target.get(dataset.timestamps[end])
        observed = bool(row and row["entry_observed"] == "1")
        missing += not observed
        flags.append(float(
            observed and float(row["entry_p95_latency_seconds"])
            >= float(row["latency_threshold_seconds"])
        ))
    with (args.output_dir / "predictions.jsonl").open() as handle:
        predictions = [json.loads(line) for line in handle]
    first_by_event = {}
    for index, row in enumerate(predictions):
        if row["event_id"] and row["failure_target"]:
            first_by_event.setdefault(row["event_id"], index)
    first_indices = list(first_by_event.values())
    metrics["current_latency_baseline"] = {
        "definition": "observed current-minute frontend p95 above its case threshold predicts next-minute degradation; missing current evidence predicts normal",
        "missing_current_minutes": missing,
        "metrics": classification_metrics(
            dataset.failure_targets[test_ends], np.asarray(flags)
        ),
    }
    metrics["first_failure_minute"] = {
        "test_events": len(first_indices),
        "model_recall_at_0_5": (
            sum(predictions[index]["failure_probability"] >= 0.5 for index in first_indices)
            / len(first_indices) if first_indices else None
        ),
        "current_latency_baseline_recall": (
            sum(flags[index] >= 0.5 for index in first_indices) / len(first_indices)
            if first_indices else None
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({
        "current_latency_baseline": metrics["current_latency_baseline"],
        "first_failure_minute": metrics["first_failure_minute"],
    }, indent=2))
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "prepare":
        return prepare_main(argv[1:])
    if argv and argv[0] == "train":
        return train_main(argv[1:])
    return label_main(argv[1:] if argv and argv[0] == "label" else argv)


if __name__ == "__main__":
    raise SystemExit(main())
