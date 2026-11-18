#!/usr/bin/env python3
"""Label, prepare and train RCAEval RE2-OB/TT future failure forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2_labels import build_re2_ob_labels, build_re2_tt_labels  # noqa: E402
from models.factory import (  # noqa: E402
    FEATURE_MODALITIES,
    MODEL_NAMES,
    named_feature_groups,
)
from training.tasks import ForecastTask, TASK_NAMES  # noqa: E402


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


def label_tt_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Label RE2-TT from its audited entry operations")
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-operation-spans", type=int, default=20)
    parser.add_argument("--latency-ratio", type=float, default=2.0)
    parser.add_argument("--min-latency-increase-seconds", type=float, default=0.1)
    args = parser.parse_args(argv)
    print(json.dumps(build_re2_tt_labels(
        args.source_audit, args.output_dir,
        min_operation_spans=args.min_operation_spans,
        latency_ratio=args.latency_ratio,
        min_latency_increase_seconds=args.min_latency_increase_seconds,
    ), indent=2))
    return 0


def prepare_main(argv: list[str]) -> int:
    from data.re2 import prepare_re2_ob, prepare_re2_tt

    parser = argparse.ArgumentParser(description="Build RE2 minute features and graph")
    parser.add_argument("archive", type=Path)
    parser.add_argument("labels", type=Path, help="Output of the RE2 label command")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    builder = prepare_re2_tt if sys.argv[1] == "prepare-tt" else prepare_re2_ob
    print(json.dumps(builder(args.archive, args.labels, args.output_dir), indent=2))
    return 0


def train_main(argv: list[str]) -> int:
    import csv
    import numpy as np

    from data.re2 import Re2FirstVisibleWindowDataset, Re2WindowDataset
    from training.experiment import run_training
    from training.metrics import classification_metrics

    parser = argparse.ArgumentParser(description="Train on case-separated RE2-OB windows")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument(
        "--task",
        choices=TASK_NAMES,
        default="state",
        help=(
            "state predicts the exact t+h degradation state; first-visible predicts "
            "whether the first audited user-visible failure occurs within (t, t+h]"
        ),
    )
    parser.add_argument("--horizon-minutes", type=int, default=1)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=FEATURE_MODALITIES,
        default=FEATURE_MODALITIES,
        help=(
            "telemetry groups retained for this run; entry-operation features are "
            "trace-derived and belong to trace"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="gcn-gru",
        help=(
            "last-mlp uses only the latest minute; tcn and gru model time without a graph; "
            "gcn-gru is supervised MTG; instantops-adapted uses GCN+GRU with "
            "service-masking RCA; ltfp is task-aligned; ltfp-type-aux adds "
            "positive-window fault-type supervision; ltfp-rc2 derives failure "
            "prediction and RCA from shared service risks; ltfp-dual-router uses "
            "separate incident-adaptive routes for prediction and RCA; ltfp-forecast predicts "
            "next-minute service telemetry before classification; ltfp-task-graph uses "
            "independent graph experts for prediction and RCA; ltfp-shared-graph-experts "
            "is its capacity-matched shared control; ltfp-no-graph removes message passing; "
            "ltfp-null-joint and ltfp-null-marginal compare the same null-anchored "
            "service-evidence model with and without root supervision; "
            "ltfp-residual-joint and ltfp-residual-marginal retain the global LTFP "
            "head and add a zero-initialized service-evidence residual; "
            "ltfp-transition adds a "
            "latest-state prior and temporal residual; eadro-adapted ports Eadro's "
            "reactive architecture to the shared future label"
        ),
    )
    parser.add_argument("--rca-method", choices=("node-head", "masking"), default="node-head")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--joint-service-pooling", action="store_true")
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--event-balanced-root-loss",
        action="store_true",
        help="give every labeled failure event equal total root-ranking weight",
    )
    parser.add_argument(
        "--telemetry-loss-weight",
        type=float,
        default=0.0,
        help="weight for ltfp-forecast next-minute telemetry reconstruction",
    )
    parser.add_argument(
        "--fault-type-loss-weight",
        type=float,
        default=0.0,
        help=(
            "weight for ltfp-type-aux fault-type classification on positive "
            "future-failure windows"
        ),
    )
    parser.add_argument(
        "--evidence-loss-weight",
        type=float,
        default=0.0,
        help="auxiliary normal/failure loss for LTFP residual service evidence",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation-loss", "validation-ap"),
        default="validation-loss",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-policy", choices=("repetition", "configuration", "deployment", "session"),
                        default="repetition")
    parser.add_argument("--split-fold", type=int, choices=range(5), default=0)
    args = parser.parse_args(argv)
    if min(args.history, args.horizon_minutes, args.epochs, args.batch_size) < 1:
        parser.error("history, horizon-minutes, epochs and batch-size must be positive")
    if args.learning_rate <= 0 or min(
        args.node_loss_weight,
        args.rank_loss_weight,
        args.telemetry_loss_weight,
        args.fault_type_loss_weight,
        args.evidence_loss_weight,
    ) < 0:
        parser.error("learning-rate must be positive and loss weights nonnegative")
    if args.model == "ltfp-type-aux" and args.fault_type_loss_weight <= 0:
        parser.error("ltfp-type-aux requires --fault-type-loss-weight > 0")
    if args.model != "ltfp-type-aux" and args.fault_type_loss_weight:
        parser.error("--fault-type-loss-weight is only valid for ltfp-type-aux")
    task = ForecastTask(args.task, args.horizon_minutes)
    if not task.is_point_forecast:
        dataset = Re2FirstVisibleWindowDataset(
            args.data_dir, history=args.history, split_policy=args.split_policy,
            split_fold=args.split_fold, horizon_minutes=args.horizon_minutes,
        )
    else:
        dataset = Re2WindowDataset(
            args.data_dir, history=args.history,
            split_policy=args.split_policy, split_fold=args.split_fold,
            horizon_minutes=args.horizon_minutes,
        )
    requested_modalities = tuple(dict.fromkeys(args.modalities))
    grouped_columns = named_feature_groups(dataset.feature_columns)
    missing_modalities = [
        name for name in requested_modalities if name not in grouped_columns
    ]
    if missing_modalities:
        parser.error(f"selected modalities have no feature columns: {missing_modalities}")
    selected_indices = sorted(
        index
        for name in requested_modalities
        for index in grouped_columns[name]
    )
    dataset.features = dataset.features[..., selected_indices].copy()
    dataset.feature_columns = tuple(
        dataset.feature_columns[index] for index in selected_indices
    )
    args.modalities = requested_modalities
    definition = json.loads((args.data_dir / "label_schema.json").read_text())
    definition = task.apply_to_label_definition(definition)
    metrics = run_training(
        dataset,
        args,
        str(args.data_dir / "service_adjacency_1m.npz"),
        definition,
    )
    source = Path(
        json.loads((args.data_dir / "feature_schema.json").read_text())["source_labels"]
    )
    if not source.is_file():
        portable_source = (
            ROOT / "data" / "processed" / source.parent.name / source.name
        )
        if not portable_source.is_file():
            raise FileNotFoundError(
                f"source labels are missing at {source} and {portable_source}"
            )
        source = portable_source
    with source.open(newline="", encoding="utf-8") as handle:
        by_target = {row["target_minute_utc"]: row for row in csv.DictReader(handle)}
    test_ends = dataset.predefined_splits[2]
    flags = []
    missing = 0
    for end in test_ends:
        row = by_target.get(dataset.timestamps[end])
        observed = bool(row and row.get("entry_request_observed", row["entry_observed"]) == "1")
        missing += not observed
        flags.append(float(observed and (
            row["entry_high"] == "1" if "entry_high" in row else
            float(row["entry_p95_latency_seconds"]) >= float(row["latency_threshold_seconds"])
        )))
    with (args.output_dir / "predictions.jsonl").open() as handle:
        predictions = [json.loads(line) for line in handle]
    first_by_event = {}
    for index, row in enumerate(predictions):
        if row["event_id"] and row["failure_target"]:
            first_by_event.setdefault(row["event_id"], index)
    first_indices = list(first_by_event.values())
    metrics["current_latency_baseline"] = {
        "definition": (
            "case-calibrated current-minute entry latency baseline for the configured "
            "forecast target; source baseline is anchored to injection-aligned pre-case minutes"
        ),
        "missing_current_minutes": missing,
        "metrics": classification_metrics(
            dataset.failure_targets[test_ends], np.asarray(flags)
        ),
    }
    rolling_flags = None
    if "entry_degradation_score_rolling" in dataset.feature_columns:
        rolling_column = dataset.feature_columns.index("entry_degradation_score_rolling")
        rolling_flags = np.asarray([
            float(dataset.features[end, :, rolling_column].max() >= 1.0)
            for end in test_ends
        ])
        metrics["rolling_latency_baseline"] = {
            "definition": (
                "current-minute entry p95 relative to median of 3+ observed preceding "
                "minutes among 5; no injection time or future telemetry"
            ),
            "metrics": classification_metrics(dataset.failure_targets[test_ends], rolling_flags),
        }
    metrics["first_failure_minute"] = {
        "forecast_semantics": task.description,
        "test_events": len(first_indices),
        "model_recall_at_0_5": (
            sum(predictions[index]["failure_probability"] >= 0.5 for index in first_indices)
            / len(first_indices) if first_indices else None
        ),
        "current_latency_baseline_recall": (
            sum(flags[index] >= 0.5 for index in first_indices) / len(first_indices)
            if first_indices else None
        ),
        "rolling_latency_baseline_recall": (
            sum(rolling_flags[index] >= 0.5 for index in first_indices) / len(first_indices)
            if first_indices and rolling_flags is not None else None
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({
        "current_latency_baseline": metrics["current_latency_baseline"],
        "rolling_latency_baseline": metrics.get("rolling_latency_baseline"),
        "first_failure_minute": metrics["first_failure_minute"],
    }, indent=2))
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in {"prepare", "prepare-tt"}:
        return prepare_main(argv[1:])
    if argv and argv[0] == "label-tt":
        return label_tt_main(argv[1:])
    if argv and argv[0] == "train":
        return train_main(argv[1:])
    return label_main(argv[1:] if argv and argv[0] == "label" else argv)


if __name__ == "__main__":
    raise SystemExit(main())
