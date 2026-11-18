#!/usr/bin/env python3
"""Build versioned failure-onset/right-censor labels for RCAEval cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import load_rcaeval_cases  # noqa: E402
from ltfp.failure_onset import (  # noqa: E402
    DEFAULT_LABEL_VERSION,
    FailureOnsetConfig,
    attach_failure_onset_quality_control,
    build_failure_onset_manifest,
    finalize_failure_onset_manifest,
    write_failure_onset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer degradation onsets from pre-injection-fitted service SLIs. "
            "Buckets are 15-second right-closed intervals and the default "
            "event rule is the first sustained 3-of-4 breach."
        )
    )
    parser.add_argument(
        "--case-root",
        action="append",
        required=True,
        help="RCAEval subset root; repeat this option to combine subsets",
    )
    parser.add_argument("--output", required=True, help="Output JSON manifest")
    parser.add_argument("--limit-per-root", type=int, default=None)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--baseline-quantile", type=float, default=0.995)
    parser.add_argument("--mad-multiplier", type=float, default=6.0)
    parser.add_argument("--required-breaches", type=int, default=3)
    parser.add_argument("--window-bins", type=int, default=4)
    parser.add_argument("--min-baseline-bins", type=int, default=8)
    parser.add_argument("--min-baseline-seconds", type=float, default=120.0)
    parser.add_argument("--normal-baseline-fraction", type=float, default=0.5)
    parser.add_argument("--max-followup-seconds", type=float, default=None)
    parser.add_argument("--label-version", default=DEFAULT_LABEL_VERSION)
    parser.add_argument(
        "--review-decisions",
        default=None,
        help=(
            "Optional JSON decisions from two blinded reviewers per case; "
            "reviewer disagreement requires one adjudicator. Without this, "
            "the output is a candidate-only manifest and cannot train models."
        ),
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help=(
            "Passing JSON report from audit_failure_onsets.py evaluate. Required "
            "with --review-decisions before labels become confirmatory-ready."
        ),
    )
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Also load logs/traces; default onset SLIs use compact metric files only",
    )
    parser.add_argument(
        "--use-full-metrics",
        action="store_true",
        help="Prefer full metric exports instead of simple service-level metrics",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest without --overwrite: {output}")

    config = FailureOnsetConfig(
        interval_seconds=args.interval_seconds,
        baseline_quantile=args.baseline_quantile,
        mad_multiplier=args.mad_multiplier,
        required_breaches=args.required_breaches,
        window_bins=args.window_bins,
        min_baseline_bins=args.min_baseline_bins,
        min_baseline_seconds=args.min_baseline_seconds,
        normal_baseline_fraction=args.normal_baseline_fraction,
        max_followup_seconds=args.max_followup_seconds,
        label_version=args.label_version,
    )

    cases = []
    roots = [Path(value) for value in args.case_root]
    for case_root in roots:
        cases.extend(
            load_rcaeval_cases(
                case_root,
                limit=args.limit_per_root,
                interval_seconds=config.interval_seconds,
                include_auxiliary=args.include_auxiliary,
                prefer_simple_metrics=not args.use_full_metrics,
            )
        )

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        raise ValueError(f"duplicate case IDs across roots: {duplicates[:5]}")

    manifest = build_failure_onset_manifest(cases, config=config)
    if bool(args.review_decisions) != bool(args.quality_report):
        raise ValueError("--review-decisions and --quality-report must be supplied together")
    if args.review_decisions:
        manifest = finalize_failure_onset_manifest(manifest, args.review_decisions)
        quality_report = json.loads(Path(args.quality_report).read_text(encoding="utf-8"))
        manifest = attach_failure_onset_quality_control(manifest, quality_report)
    manifest["source_roots"] = [str(root) for root in roots]
    manifest.pop("source_root", None)
    write_failure_onset_manifest(manifest, output)

    summary = manifest["summary"]
    print(f"Wrote {output}")
    if manifest["stage"] == "candidate_review":
        print(
            f"cases={summary['case_count']} candidates={summary['candidate_onset_count']} "
            f"pending_review={summary['pending_review_count']} label_version={manifest['label_version']}"
        )
    else:
        print(
            f"cases={summary['case_count']} accepted={summary['accepted_onset_count']} "
            f"censored={summary['right_censored_count']} adjudicated={summary['adjudicated_count']} "
            f"label_version={manifest['label_version']}"
        )


if __name__ == "__main__":
    main()
