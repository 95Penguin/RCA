#!/usr/bin/env python3
"""Run the safe residual service-evidence feasibility comparison."""

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
    "ltfp-residual-marginal",
    "ltfp-residual-joint",
    "ltfp-residual-protected-joint",
    "ltfp-local-adapter-joint",
    "ltfp-local-adapter-isolated-joint",
    "ltfp-protected-root-sidecar",
    "ltfp-context-evidence-collab",
)
ROOT_SUPERVISED_MODELS = {
    "ltfp-residual-joint",
    "ltfp-residual-protected-joint",
    "ltfp-local-adapter-joint",
    "ltfp-local-adapter-isolated-joint",
    "ltfp-protected-root-sidecar",
    "ltfp-context-evidence-collab",
}
EVIDENCE_MODELS = {
    "ltfp-residual-marginal",
    "ltfp-residual-joint",
    "ltfp-residual-protected-joint",
    "ltfp-local-adapter-joint",
    "ltfp-local-adapter-isolated-joint",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=(0,))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--root-loss-weight", type=float, default=0.1)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.1)
    parser.add_argument("--event-balanced-root-loss", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if min(args.root_loss_weight, args.evidence_loss_weight) < 0:
        parser.error("loss weights must be nonnegative")

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for fold in args.folds:
            for seed in args.seeds:
                for model in args.models:
                    rank_weight = (
                        args.root_loss_weight
                        if model in ROOT_SUPERVISED_MODELS
                        else (0.5 if model == "ltfp" else 0.0)
                    )
                    evidence_weight = (
                        args.evidence_loss_weight if model in EVIDENCE_MODELS else 0.0
                    )
                    suffix = (
                        f"root{rank_weight:g}_evidence{evidence_weight:g}"
                        .replace(".", "p")
                    )
                    if args.event_balanced_root_loss:
                        suffix += "_eventbalanced"
                    output = args.output_root / (
                        f"re2_{dataset}_{model}_deployment_fold{fold}_"
                        f"{args.epochs}ep_seed{seed}_{suffix}"
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
                            str(rank_weight),
                            "--evidence-loss-weight",
                            str(evidence_weight),
                            "--checkpoint-selection",
                            "validation-ap",
                            "--seed",
                            str(seed),
                            "--split-policy",
                            "deployment",
                            "--split-fold",
                            str(fold),
                        ]
                        if args.event_balanced_root_loss:
                            command.append("--event-balanced-root-loss")
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
                    evidence = report.get("service_evidence", {})
                    row = {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "average_precision": failure["average_precision"],
                        "f1": failure["f1_at_0_5"],
                        "hit_at_1": rca["hit_at_1"],
                        "mrr": rca["mrr"],
                        "root_loss_weight": rank_weight,
                        "evidence_loss_weight": evidence_weight,
                        "event_balanced_root_loss": args.event_balanced_root_loss,
                        "residual_scale": evidence.get("residual_scale"),
                        "global_ap": evidence.get("global_branch", {}).get(
                            "average_precision"
                        ),
                        "evidence_ap": evidence.get("evidence_branch", {}).get(
                            "average_precision"
                        ),
                        "metrics_path": str(metrics_path),
                    }
                    rows.append(row)
                    print(
                        f"done dataset={dataset} fold={fold} seed={seed} model={model} "
                        f"AP={row['average_precision']:.3f} MRR={row['mrr']:.3f} "
                        f"alpha={row['residual_scale']}",
                        flush=True,
                    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    weight_tag = (
        f"root{args.root_loss_weight:g}_evidence{args.evidence_loss_weight:g}"
        .replace(".", "p")
    )
    if args.event_balanced_root_loss:
        weight_tag += "_eventbalanced"
    destination = args.output_root / f"residual_evidence_summary_{weight_tag}.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
