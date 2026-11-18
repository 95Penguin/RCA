#!/usr/bin/env python3
"""Train complete coarse/fine models and matched LTFP controls from scratch."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ob","tt"), default=("ob","tt"))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=range(5))
    parser.add_argument("--models", nargs="+", choices=("ltfp","joint-granularity","joint-granularity-encoded-local","joint-granularity-null-selection"), default=("ltfp","joint-granularity"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.output_dir.exists():
        parser.error("positive epochs and new output directory required")
    args.output_dir.mkdir(parents=True)
    rows = []
    for name in args.datasets:
        for fold in args.folds:
            for seed in args.seeds:
                for model in args.models:
                    output = args.output_dir / f"{name}_fold{fold}_seed{seed}_{model}"
                    command = [sys.executable, "-u", "-B", str(ROOT/"scripts/run_re2.py"), "train",
                               "--data-dir", str(ROOT/f"artifacts/re2_{name}_panel_fair_v1"),
                               "--output-dir", str(output), "--model", model,
                               "--history", "5", "--task", "state", "--horizon-minutes", "1",
                               "--epochs", str(args.epochs), "--batch-size", "64", "--device", "cpu",
                               "--learning-rate", "0.001", "--node-loss-weight", "0",
                               "--rank-loss-weight", "0.5", "--event-balanced-root-loss",
                               "--checkpoint-selection", "validation-ap", "--rca-method", "node-head",
                               "--split-policy", "deployment", "--split-fold", str(fold), "--seed", str(seed)]
                    log = args.output_dir / f"{output.name}.log"
                    print("start", name, fold, seed, model, flush=True)
                    with log.open("w") as handle:
                        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f"training failed; see {log}")
                    metrics = json.loads((output/"metrics.json").read_text())
                    row = {"dataset": name, "fold": fold, "seed": seed, "model": model,
                           "failure": metrics["failure_prediction"],
                           "rca": metrics["root_cause_event_proxy"],
                           "checkpoint_selection": metrics["checkpoint_selection"],
                           "effective_losses": metrics["effective_loss_weights"],
                           "output_directory": str(output)}
                    rows.append(row)
                    (args.output_dir/"results.json").write_text(json.dumps(rows, indent=2))
                    print("done", name, fold, model, "AP", row["failure"]["average_precision"],
                          "MRR", row["rca"]["mrr"], flush=True)


if __name__ == "__main__":
    main()
