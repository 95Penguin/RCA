#!/usr/bin/env python3
"""Screen modality complementarity under the fixed LTFP state-1m protocol."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
COMBINATIONS = {
    "metric": ("metric",),
    "log": ("log",),
    "trace": ("trace",),
    "metric_log": ("metric", "log"),
    "metric_trace": ("metric", "trace"),
    "log_trace": ("log", "trace"),
    "all": ("metric", "log", "trace"),
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _output(dataset: str, combination: str, fold: int, args: argparse.Namespace) -> Path:
    return ROOT / "artifacts" / (
        f"re2_{dataset}_ltfp_modality_{combination}_deployment_fold{fold}_"
        f"state_1m_fair_{args.epochs}ep_seed{args.seed}_v1"
    )


def _run_one(
    dataset: str, combination: str, fold: int, args: argparse.Namespace
) -> Path:
    output = _output(dataset, combination, fold, args)
    metrics = output / "metrics.json"
    if metrics.is_file() and not args.force:
        print(
            f"skip dataset={dataset} modalities={combination} fold={fold}",
            flush=True,
        )
        return metrics
    if output.exists() and not args.force:
        raise FileExistsError(f"incomplete output exists: {output}")
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/run_re2.py"),
        "train",
        "--data-dir",
        str(ROOT / "artifacts" / f"re2_{dataset}_panel_fair_v1"),
        "--output-dir",
        str(output),
        "--history",
        "5",
        "--task",
        "state",
        "--horizon-minutes",
        "1",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--model",
        "ltfp",
        "--rca-method",
        "node-head",
        "--learning-rate",
        "0.001",
        "--node-loss-weight",
        "0",
        "--rank-loss-weight",
        "0.5",
        "--seed",
        str(args.seed),
        "--split-policy",
        "deployment",
        "--split-fold",
        str(fold),
        "--modalities",
        *COMBINATIONS[combination],
    ]
    print(
        f"start dataset={dataset} modalities={combination} fold={fold}",
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        print(completed.stdout[-4000:])
        raise RuntimeError(f"training failed; see {output / 'run.log'}")
    return metrics


def _summarize(
    paths: dict[tuple[str, str], list[Path]], args: argparse.Namespace
) -> dict[str, object]:
    summary: dict[str, object] = {
        "protocol": {
            "task": "exact next-minute failure state and event-level RCA",
            "model": "ltfp",
            "split": "deployment holdout",
            "folds": args.folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "entry_feature_group": "trace",
            "purpose": (
                "screen whether modality-specific evidence is complementary before "
                "implementing incident-adaptive routing"
            ),
        },
        "results": {},
    }
    for (dataset, combination), metric_paths in paths.items():
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
        values = {
            "ap": [r["failure_prediction"]["average_precision"] for r in reports],
            "f1": [r["failure_prediction"]["f1_at_0_5"] for r in reports],
            "hit_at_1": [r["root_cause_event_proxy"]["hit_at_1"] for r in reports],
            "mrr": [r["root_cause_event_proxy"]["mrr"] for r in reports],
        }
        summary["results"].setdefault(f"RE2-{dataset.upper()}", {})[
            combination
        ] = {
            name: {"mean": _mean(scores), "folds": scores}
            for name, scores in values.items()
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--combinations",
        nargs="+",
        choices=tuple(COMBINATIONS),
        default=tuple(COMBINATIONS),
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(5), default=tuple(range(5))
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch-size must be positive")

    paths: dict[tuple[str, str], list[Path]] = {}
    for dataset in args.datasets:
        for combination in args.combinations:
            paths[dataset, combination] = [
                _run_one(dataset, combination, fold, args) for fold in args.folds
            ]
            reports = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in paths[dataset, combination]
            ]
            print(
                f"done dataset={dataset} modalities={combination} "
                f"AP={_mean([r['failure_prediction']['average_precision'] for r in reports]):.3f} "
                f"Hit@1={_mean([r['root_cause_event_proxy']['hit_at_1'] for r in reports]):.3f}",
                flush=True,
            )

    summary = _summarize(paths, args)
    dataset_tag = "_".join(args.datasets)
    combination_tag = "_".join(args.combinations)
    fold_tag = "_".join(str(fold) for fold in args.folds)
    destination = ROOT / "artifacts" / (
        f"re2_modality_feasibility_{dataset_tag}_{combination_tag}_"
        f"folds_{fold_tag}_{args.epochs}ep_seed{args.seed}_v1.json"
    )
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
