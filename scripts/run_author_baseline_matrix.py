#!/usr/bin/env python3
"""Run pinned author adapters; never reuse legacy/stale directories silently."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=("ltfp", "instantops", "eadro", "art"),
                   default=["ltfp", "instantops", "eadro", "art"])
    p.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=["ob", "tt"])
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--max-windows", type=int, help="Smoke runs only")
    args = p.parse_args()
    if args.output_root.exists():
        p.error("output root already exists; choose a new run directory")
    for name in ("methods", "datasets", "folds"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            p.error(f"duplicate {name}")
    args.output_root.mkdir(parents=True)
    records = []
    for dataset in args.datasets:
        for method in args.methods:
            for fold in args.folds:
                out = args.output_root / f"{dataset}_{method}_fold{fold}_seed{args.seed}"
                command = [sys.executable, str(ROOT / "scripts/run_baselines.py"), "run",
                           "--method", method, "--dataset", dataset, "--fold", str(fold),
                           "--epochs", str(args.epochs), "--seed", str(args.seed),
                           "--batch-size", str(args.batch_size), "--hidden", str(args.hidden),
                           "--repeats", str(args.repeats), "--output-dir", str(out)]
                if args.max_windows:
                    command += ["--max-windows", str(args.max_windows)]
                subprocess.run(command, cwd=ROOT, check=True)
                records.append(json.loads((out / "metrics.json").read_text()))
                (args.output_root / "runs.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")
    summary = []
    for dataset in args.datasets:
        for method in args.methods:
            subset = [r for r in records if r["dataset"] == dataset and r["method"] == method]
            row = {"dataset": dataset, "method": method, "folds": args.folds,
                   "seed": args.seed, "smoke_only": bool(args.max_windows)}
            for key, values in {
                "AP": [r["average_precision"] for r in subset],
                "F1_val_threshold": [r["at_validation_threshold"]["f1"] for r in subset],
                "MRR_window": [r["root_cause_window"]["mrr"] for r in subset],
                "Hit1_window": [r["root_cause_window"]["hit_at_1"] for r in subset],
            }.items():
                valid = [v for v in values if v is not None]
                row[key] = {"mean": statistics.fmean(valid) if valid else None,
                            "sd": statistics.stdev(valid) if len(valid)>1 else None,
                            "valid_folds": len(valid)}
            summary.append(row)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
