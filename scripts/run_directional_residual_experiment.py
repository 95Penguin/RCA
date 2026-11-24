#!/usr/bin/env python3
"""Evaluate the StageRoute symmetric backbone plus directed RCA residual."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
MODEL = "ltfp-stage-route-directional"


def _learned_gate(checkpoint_path: Path, name: str, bound: float) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint["model_state_dict"][name]
    return float(bound * math.tanh(float(raw)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=range(5))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=Path("/private/tmp/nextminuterca_directed_panels"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/private/tmp/nextminuterca_directional_residual"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        panel = args.panel_root / f"re2_{dataset}_panel_directed_v1"
        if not (panel / "service_adjacency_1m.npz").is_file():
            raise FileNotFoundError(f"missing directed panel: {panel}")
        for fold in args.folds:
            output = args.output_root / (
                f"re2_{dataset}_{MODEL}_fold{fold}_state_1m_"
                f"{args.epochs}ep_seed{args.seed}"
            )
            metrics_path = output / "metrics.json"
            if not metrics_path.is_file():
                if output.exists():
                    raise FileExistsError(f"incomplete output exists: {output}")
                command = [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts" / "run_re2.py"),
                    "train",
                    "--data-dir",
                    str(panel),
                    "--output-dir",
                    str(output),
                    "--history",
                    "5",
                    "--task",
                    "state",
                    "--horizon-minutes",
                    "1",
                    "--graph-direction",
                    "callee-to-caller",
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    "64",
                    "--device",
                    args.device,
                    "--model",
                    MODEL,
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
                    str(args.seed),
                    "--split-policy",
                    "deployment",
                    "--split-fold",
                    str(fold),
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
                    print(completed.stdout[-5000:])
                    raise RuntimeError(f"training failed; see {output / 'run.log'}")

            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint_path = output / "best_model.pt"
            row = {
                "dataset": dataset,
                "fold": fold,
                "seed": args.seed,
                "average_precision": report["failure_prediction"]["average_precision"],
                "f1": report["failure_prediction"]["f1_at_0_5"],
                "window_mrr": report["root_cause_proxy"]["mrr"],
                "event_mrr": report["root_cause_event_proxy"]["mrr"],
                "directional_gate": _learned_gate(
                    checkpoint_path, "directional_residual_scale", 0.5
                ),
                "spatial_gate": _learned_gate(
                    checkpoint_path, "spatial_residual_scale", 0.5
                ),
                "metrics_path": str(metrics_path),
            }
            rows.append(row)
            print(
                f"done dataset={dataset} fold={fold} "
                f"AP={row['average_precision']:.3f} "
                f"window_MRR={row['window_mrr']:.3f} "
                f"direction_gate={row['directional_gate']:+.4f}",
                flush=True,
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / "directional_residual_summary.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
