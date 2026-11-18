#!/usr/bin/env python3
"""Run the minimal task-adaptive graph-interaction feasibility matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
MODELS = (
    "ltfp",
    "ltfp-task-graph",
    "ltfp-shared-graph-experts",
    "ltfp-no-graph",
)


def _baseline_path(dataset: str, fold: int, epochs: int, seed: int) -> Path:
    return ROOT / "artifacts" / (
        f"re2_{dataset}_ltfp_deployment_fold{fold}_fair_"
        f"{epochs}ep_seed{seed}_baseline_v1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=(0,))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for fold in args.folds:
            for seed in args.seeds:
                for model in args.models:
                    if model == "ltfp":
                        output = _baseline_path(dataset, fold, args.epochs, seed)
                    else:
                        output = args.output_root / (
                            f"re2_{dataset}_{model}_deployment_fold{fold}_"
                            f"{args.epochs}ep_seed{seed}"
                        )
                    metrics_path = output / "metrics.json"
                    if not metrics_path.is_file():
                        if model == "ltfp" and output == _baseline_path(
                            dataset, fold, args.epochs, seed
                        ):
                            output = args.output_root / (
                                f"re2_{dataset}_{model}_deployment_fold{fold}_"
                                f"{args.epochs}ep_seed{seed}"
                            )
                            metrics_path = output / "metrics.json"
                        if not metrics_path.is_file():
                            if output.exists():
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
                                "64",
                                "--device",
                                args.device,
                                "--model",
                                model,
                                "--modalities",
                                "metric",
                                "log",
                                "trace",
                                "--rca-method",
                                "node-head",
                                "--learning-rate",
                                "0.001",
                                "--node-loss-weight",
                                "0",
                                "--rank-loss-weight",
                                "0.5",
                                "--seed",
                                str(seed),
                                "--split-policy",
                                "deployment",
                                "--split-fold",
                                str(fold),
                            ]
                            print(
                                f"start dataset={dataset} fold={fold} seed={seed} model={model}",
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
                            (output / "run.log").write_text(
                                completed.stdout, encoding="utf-8"
                            )
                            if completed.returncode:
                                print(completed.stdout[-5000:])
                                raise RuntimeError(
                                    f"training failed; see {output / 'run.log'}"
                                )

                    report = json.loads(metrics_path.read_text(encoding="utf-8"))
                    failure = report["failure_prediction"]
                    rca = report["root_cause_event_proxy"]
                    row = {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "average_precision": failure["average_precision"],
                        "f1": failure["f1_at_0_5"],
                        "hit_at_1": rca["hit_at_1"],
                        "mrr": rca["mrr"],
                        "metrics_path": str(metrics_path),
                    }
                    rows.append(row)
                    print(
                        f"done dataset={dataset} fold={fold} seed={seed} model={model} "
                        f"AP={row['average_precision']:.3f} MRR={row['mrr']:.3f}",
                        flush=True,
                    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / "task_graph_summary.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
