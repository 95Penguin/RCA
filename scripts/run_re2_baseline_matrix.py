#!/usr/bin/env python3
"""Run and summarize next-minute RE2 baselines on deployment-held-out folds."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import MODEL_NAMES  # noqa: E402
from training.tasks import ForecastTask, TASK_NAMES  # noqa: E402


DATASETS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _run_one(dataset: str, model: str, fold: int, args: argparse.Namespace) -> Path:
    task_tag = ForecastTask(args.task, args.horizon_minutes).slug
    output = ROOT / (
        f"artifacts/re2_{dataset}_{model.replace('-', '_')}_deployment_"
        f"fold{fold}_{task_tag}_fair_{args.epochs}ep_seed{args.seed}_baseline_v1"
    )
    metrics = output / "metrics.json"
    if metrics.exists() and not args.force:
        print(f"skip existing dataset={dataset} model={model} fold={fold}")
        return metrics
    command = [
        sys.executable,
        str(ROOT / "scripts/run_re2.py"),
        "train",
        "--data-dir",
        str(DATASETS[dataset]),
        "--output-dir",
        str(output),
        "--model",
        model,
        "--task",
        args.task,
        "--horizon-minutes",
        str(args.horizon_minutes),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--split-policy",
        "deployment",
        "--split-fold",
        str(fold),
        "--seed",
        str(args.seed),
    ]
    if model == "ltfp":
        command.extend(("--node-loss-weight", "0", "--rank-loss-weight", "0.5"))
    elif model == "eadro-adapted":
        command.extend(("--node-loss-weight", "0", "--rank-loss-weight", "1.0"))
    elif model == "instantops-adapted":
        command.extend(("--rca-method", "masking", "--node-loss-weight", "0"))
    print(f"run dataset={dataset} model={model} fold={fold}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"baseline failed:\n{tail}")
    report = json.loads(metrics.read_text())
    prediction = report["failure_prediction"]
    rca = report["root_cause_event_proxy"]
    print(
        f"done F1={prediction['f1_at_0_5']:.3f} "
        f"AP={prediction['average_precision']:.3f} "
        f"Hit@1={rca['hit_at_1']:.3f} MRR={rca['mrr']:.3f}"
    )
    return metrics


def _summarize(paths: dict[tuple[str, str], list[Path]], args: argparse.Namespace) -> dict:
    summary: dict[str, object] = {
        "protocol": {
            "task": ForecastTask(args.task, args.horizon_minutes).description,
            "task_id": args.task,
            "horizon_minutes": args.horizon_minutes,
            "datasets": ["RE2-OB", "RE2-TT"],
            "models": args.models,
            "folds": args.folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "split": "deployment holdout; test fold=N, validation fold=(N+1)%5",
            "aggregation": "unweighted mean and sample SD over five held-out folds",
        },
        "results": {},
    }
    for (dataset, model), metric_paths in paths.items():
        reports = [json.loads(path.read_text()) for path in metric_paths]
        metrics = {
            "precision": [r["failure_prediction"]["precision_at_0_5"] for r in reports],
            "recall": [r["failure_prediction"]["recall_at_0_5"] for r in reports],
            "f1": [r["failure_prediction"]["f1_at_0_5"] for r in reports],
            "ap": [r["failure_prediction"]["average_precision"] for r in reports],
            "event_hit_at_1": [r["root_cause_event_proxy"]["hit_at_1"] for r in reports],
            "event_hit_at_3": [r["root_cause_event_proxy"]["hit_at_3"] for r in reports],
            "event_hit_at_5": [r["root_cause_event_proxy"]["hit_at_5"] for r in reports],
            "event_mrr": [r["root_cause_event_proxy"]["mrr"] for r in reports],
        }
        summary["results"].setdefault(f"RE2-{dataset.upper()}", {})[model] = {
            key: {**_mean_sd(values), "folds": values} for key, values in metrics.items()
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=("last-mlp", "tcn"),
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=tuple(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=tuple(range(5)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--task", choices=TASK_NAMES, default="state")
    parser.add_argument("--horizon-minutes", type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.horizon_minutes) < 1:
        parser.error("epochs, batch-size and horizon-minutes must be positive")
    paths: dict[tuple[str, str], list[Path]] = {}
    for dataset in args.datasets:
        for model in args.models:
            paths[dataset, model] = [
                _run_one(dataset, model, fold, args) for fold in args.folds
            ]
    summary = _summarize(paths, args)
    model_tag = "_".join(model.replace("-", "_") for model in args.models)
    dataset_tag = "_".join(args.datasets)
    task_tag = ForecastTask(args.task, args.horizon_minutes).slug
    destination = ROOT / (
        f"artifacts/re2_{task_tag}_{dataset_tag}_{model_tag}_"
        f"{args.epochs}ep_seed{args.seed}_v1.json"
    )
    destination.write_text(json.dumps(summary, indent=2))
    print(f"saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
