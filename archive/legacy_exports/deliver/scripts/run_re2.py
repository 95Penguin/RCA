#!/usr/bin/env python3
"""Build auditable next-minute labels for RCAEval RE2-OB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2_labels import build_re2_ob_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-entry-spans", type=int, default=100)
    parser.add_argument("--latency-ratio", type=float, default=1.5)
    parser.add_argument("--min-latency-increase-seconds", type=float, default=0.1)
    parser.add_argument("--consecutive-minutes", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build_re2_ob_labels(
        args.archive, args.output_dir,
        min_entry_spans=args.min_entry_spans,
        latency_ratio=args.latency_ratio,
        min_latency_increase_seconds=args.min_latency_increase_seconds,
        consecutive_minutes=args.consecutive_minutes,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
