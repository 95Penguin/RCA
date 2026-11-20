#!/usr/bin/env python3
"""Run author specialist adapters on common splits, preserving task boundaries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.specialists import METHODS


def summarize(records):
    rows = []
    for dataset, method in sorted({(r["dataset"], r["method"]) for r in records}):
        runs = [r for r in records if (r["dataset"], r["method"]) == (dataset, method)]
        row = {"dataset": dataset, "method": method, "runs": len(runs),
               "smoke_only": any(r["smoke_only"] for r in runs)}
        paths = {"AP": ("failure_prediction", "average_precision"),
                 "MRR_window": ("root_cause_window", "mrr"),
                 "Hit1_window": ("root_cause_window", "hit_at_1"),
                 "root_coverage": ("root_cause_window", "coverage")}
        for label, (task, metric) in paths.items():
            values = [r[task][metric] for r in runs if r[task] is not None and r[task][metric] is not None]
            row[label] = {"mean": statistics.fmean(values) if values else None,
                          "sd": statistics.stdev(values) if len(values) > 1 else None,
                          "valid_runs": len(values)}
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=["ob", "tt"])
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--reference-rows", type=int, default=2048)
    p.add_argument("--diff-steps", type=int, default=100)
    p.add_argument("--forecast-samples", type=int, default=16)
    p.add_argument("--detector-features", type=int, default=64)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--max-windows", type=int, help="Prefix smoke test only, never a formal performance result")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("jobs must be positive")
    if args.output_root.exists():
        p.error("output root already exists; choose a fresh run directory")
    for key in ("methods", "datasets", "folds", "seeds"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            p.error(f"duplicate {key}")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True)
    tasks = [(dataset, method, fold, seed) for dataset in args.datasets for method in args.methods
             for fold in args.folds for seed in args.seeds]

    def execute(task):
        dataset, method, fold, seed = task
        name = f"{dataset}_{method}_fold{fold}_seed{seed}"
        out = args.output_root / name
        command = [sys.executable, str(ROOT / "scripts/run_baselines.py"), "specialist",
                   "--method", method, "--dataset", dataset, "--fold", str(fold),
                   "--seed", str(seed), "--output-dir", str(out)]
        for key in ("epochs", "batch_size", "hidden", "reference_rows", "diff_steps",
                    "forecast_samples", "detector_features", "max_windows"):
            value = getattr(args, key)
            if value is not None:
                command.extend(["--"+key.replace("_", "-"), str(value)])
        with (args.output_root / (name+".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            return None, {"run": name, "exit_code": result.returncode, "command": command}
        return json.loads((out / "metrics.json").read_text()), None

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            record, failure = future.result()
            if failure:
                failures.append(failure)
            else:
                records.append(record)
            for filename, payload in (("runs.json", records), ("failures.json", failures),
                                      ("summary.json", summarize(records))):
                (args.output_root / filename).write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
            print(json.dumps({"finished": len(records)+len(failures), "total": len(tasks),
                              "failed": len(failures), "task": futures[future]}), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
