#!/usr/bin/env python3
"""Create the one-time seed-7 scenario-disjoint fold manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.protocol import (  # noqa: E402
    load_label_manifest,
    make_scenario_folds,
    summarize_fold_composition,
    write_manifest_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--output", default="manifests/failure_onset_v2/splits_v1.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_label_manifest(args.label_manifest, require_confirmatory=True)
    split = make_scenario_folds(labels, n_folds=args.folds, seed=args.seed)
    output = write_manifest_csv(split, args.output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "version": "failure-onset-splits-v1",
                "label_manifest": str(Path(args.label_manifest)),
                "seed": args.seed,
                "n_folds": args.folds,
                "composition": summarize_fold_composition(split),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(split)} cases to {output}")
    print(f"Wrote fold summary to {summary_path}")


if __name__ == "__main__":
    main()
