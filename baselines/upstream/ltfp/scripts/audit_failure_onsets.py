#!/usr/bin/env python3
"""Create or evaluate the blinded 20% two-reviewer onset audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.audit import (  # noqa: E402
    evaluate_two_reviewer_audit,
    stratified_audit_sample,
)
from ltfp.failure_onset import read_failure_onset_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--candidate-manifest", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--fraction", type=float, default=0.20)
    sample.add_argument("--seed", type=int, default=7)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--candidate-manifest", required=True)
    evaluate.add_argument("--reviewer-a", required=True)
    evaluate.add_argument("--reviewer-b", required=True)
    evaluate.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = read_failure_onset_manifest(args.candidate_manifest)
    if args.command == "sample":
        sample = stratified_audit_sample(
            pd.DataFrame(candidate["cases"]), fraction=args.fraction, seed=args.seed
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output, index=False)
        print(f"Wrote {len(sample)} blinded review rows to {output}")
        return
    report = evaluate_two_reviewer_audit(
        pd.read_csv(args.reviewer_a),
        pd.read_csv(args.reviewer_b),
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
