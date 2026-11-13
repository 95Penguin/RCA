#!/usr/bin/env python3
"""Run paired full-July MicroSS experiments on a fixed event-purged split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    panel = Path("data/processed/micross_mps_smoke_01")
    rows = []
    for model in ("gcn-gru", "gru"):
        for seed in args.seeds:
            run_dir = root / f"{model}_seed{seed}"
            command = [
                str(repo / ".venv/bin/python"), "-u", str(repo / "scripts/run_micross.py"), "train",
                "--data-dir", str(panel), "--graph", str(panel / "service_adjacency_1m.npz"),
                "--labels", str(panel / "entry_failure_v2.csv"),
                "--output-dir", str(run_dir), "--model", model,
                "--rca-method", "node-head", "--epochs", str(args.epochs),
                "--batch-size", "128", "--device", "cpu", "--seed", str(seed),
            ]
            print(f"START {model} seed={seed}", flush=True)
            started = time.monotonic()
            run_dir.mkdir(exist_ok=True)
            with (run_dir / "run.log").open("w") as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            result = {
                "model": model, "seed": seed, "seconds": round(time.monotonic() - started, 1),
                "best_epoch": min(metrics["training_history"], key=lambda x: x["val_loss"])["epoch"],
                "split_sizes": metrics["split_sizes"],
                "split_positive_samples": metrics["split_positive_samples"],
                "failure_prediction": metrics["failure_prediction"],
                "current_error_baseline": metrics["current_error_baseline"]["all_test_windows"]["metrics"],
                "root_label_coverage": metrics["root_label_coverage"],
                "root_cause_event_proxy": metrics["root_cause_event_proxy"],
                "root_cause_proxy": metrics["root_cause_proxy"],
            }
            rows.append(result)
            (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"DONE {model} seed={seed} {result['seconds']}s "
                  f"AP={result['failure_prediction']['average_precision']:.3f} "
                  f"F1={result['failure_prediction']['f1_at_0_5']:.3f} "
                  f"root_event_hit1={result['root_cause_event_proxy']['hit_at_1']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
