#!/usr/bin/env python3
"""Run the controlled five-fold future-telemetry auxiliary experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")


def _output(dataset: str, fold: int, epochs: int, seed: int) -> Path:
    return ROOT / "artifacts" / (
        f"re2_{dataset}_ltfp_forecast_deployment_fold{fold}_"
        f"first_visible_1m_fair_{epochs}ep_seed{seed}_v1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--telemetry-loss-weight", type=float, default=0.2)
    args = parser.parse_args()
    if args.epochs < 1 or args.telemetry_loss_weight < 0:
        parser.error("epochs must be positive and telemetry loss weight nonnegative")

    summaries = []
    for dataset in args.datasets:
        for fold in range(5):
            output = _output(dataset, fold, args.epochs, args.seed)
            metrics_path = output / "metrics.json"
            if metrics_path.is_file():
                print(f"skip complete dataset={dataset} fold={fold}", flush=True)
            else:
                if output.exists():
                    raise FileExistsError(f"incomplete output exists: {output}")
                command = [
                    sys.executable, "-B", str(ROOT / "scripts/run_re2.py"), "train",
                    "--data-dir", str(ROOT / "artifacts" / f"re2_{dataset}_panel_fair_v1"),
                    "--output-dir", str(output), "--history", "5",
                    "--task", "first-visible", "--horizon-minutes", "1",
                    "--epochs", str(args.epochs), "--batch-size", "64",
                    "--device", "auto", "--model", "ltfp-forecast",
                    "--rca-method", "node-head", "--learning-rate", "0.001",
                    "--node-loss-weight", "0", "--rank-loss-weight", "0.5",
                    "--telemetry-loss-weight", str(args.telemetry_loss_weight),
                    "--seed", str(args.seed), "--split-policy", "deployment",
                    "--split-fold", str(fold),
                ]
                print(f"start dataset={dataset} fold={fold}", flush=True)
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
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            failure = metrics["failure_prediction"]
            rca = metrics["root_cause_event_proxy"]
            telemetry = metrics["telemetry_forecast"]
            summary = {
                "dataset": dataset,
                "fold": fold,
                "average_precision": failure["average_precision"],
                "hit_at_1": rca["hit_at_1"],
                "forecast_rmse": telemetry["model_rmse"],
                "persistence_rmse": telemetry["persistence_rmse"],
            }
            summaries.append(summary)
            print(
                f"done dataset={dataset} fold={fold} "
                f"AP={summary['average_precision']:.3f} "
                f"Hit@1={summary['hit_at_1']:.3f} "
                f"forecast_RMSE={summary['forecast_rmse']:.3f} "
                f"persistence_RMSE={summary['persistence_rmse']:.3f}",
                flush=True,
            )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
