"""Post-hoc fusion of the learned OCSID RCA head with CPCA responsibility.

For each positive test window the fused score is

    score_i = log(a_i) + lambda * r_i

where ``a_i`` is the unified model's node score and ``r_i`` is CPCA's
counterfactual removable-latency fraction.  ``lambda=0`` recovers the learned
head; ``r-only`` recovers CPCA.  The script reports the lambda curve and the
best value, with event-level Hit@1 / MRR.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/fuse_ocsid_cpca.py \
  --dataset ob \
  --ocsid artifacts/ocsid_joint_ob_seed42_v1.json \
  --cpca artifacts/cpca_gate2_ob_first10_z2_scores_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paths import ROOT
PREFIX = {"ob": "RE2-OB", "tt": "RE2-TT"}


def _case_id(member: str, dataset: str) -> str:
    return f"{PREFIX[dataset]}_{member.replace('/', '_')}"


def _load_cpca(path: Path, dataset: str):
    data = json.loads(path.read_text())
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    for member, case in data["cases"].items():
        case_id = _case_id(member, dataset)
        for window in case.get("windows", []):
            scores = window.get("cpca_scores") or {}
            if scores:
                lookup[case_id, int(window["minute"])] = scores
    return lookup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--ocsid", type=Path, required=True)
    parser.add_argument("--cpca", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--mode", choices=("logit", "rrf"), default="logit")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    ocsid = json.loads(args.ocsid.read_text())
    cpca = _load_cpca(args.cpca, args.dataset)
    services = ocsid["folds"][0]["services"]
    index = {name: i for i, name in enumerate(services)}

    rows = []
    covered = 0
    total = 0
    for fold in ocsid["folds"]:
        for prediction in fold["test_predictions"]:
            if not prediction["event"] or not prediction["root"]:
                continue
            if "node_scores" not in prediction:
                raise SystemExit(
                    "This OCSID JSON predates the removal of the learned head; "
                    "fuse_ocsid_cpca.py only supports legacy outputs. "
                    "New runs are CPCA-only by design."
                )
            total += 1
            scores = cpca.get((prediction["case"], prediction["minute"]))
            if scores is not None:
                covered += 1
            rows.append({
                "event": prediction["event"],
                "root": prediction["root"],
                "a": np.asarray(prediction["node_scores"], dtype=np.float64),
                "r": np.asarray([
                    scores.get(name, 0.0) if scores else 0.0 for name in services
                ], dtype=np.float64),
            })

    report = {
        "dataset": args.dataset,
        "positive_windows": total,
        "cpca_covered_windows": covered,
        "coverage": covered / total if total else None,
        "lambdas": {},
    }
    best = None
    if args.mode == "rrf":
        # Reciprocal rank fusion: scale-free, so a poorly calibrated learned
        # head cannot dominate the structural score.
        for row in rows:
            order_a = np.argsort(-row["a"])
            order_r = np.argsort(-row["r"])
            rank_a = np.empty_like(order_a)
            rank_r = np.empty_like(order_r)
            rank_a[order_a] = np.arange(1, len(order_a) + 1)
            rank_r[order_r] = np.arange(1, len(order_r) + 1)
            row["fused"] = 1.0 / (60 + rank_a) + 1.0 / (60 + rank_r)
        args.lambdas = [0.0]
    for lam in args.lambdas:
        if args.mode == "logit":
            for row in rows:
                row["fused"] = np.log(row["a"] + 1e-9) + lam * row["r"]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["event"]].append(row)
        ranks = []
        for event, items in grouped.items():
            means = np.mean([item["fused"] for item in items], axis=0)
            order = np.argsort(-means)
            root = items[0]["root"]
            rank = int(np.flatnonzero(np.array(services)[order] == root)[0]) + 1
            ranks.append(rank)
        metrics = {
            "events": len(ranks),
            "hit_at_1": float(np.mean([r == 1 for r in ranks])),
            "hit_at_3": float(np.mean([r <= 3 for r in ranks])),
            "mrr": float(np.mean([1.0 / r for r in ranks])),
        }
        report["lambdas"][str(lam)] = metrics
        if best is None or metrics["mrr"] > best[1]["mrr"]:
            best = (lam, metrics)

    # CPCA-only reference.
    for row in rows:
        row["fused"] = row["r"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event"]].append(row)
    ranks = []
    for event, items in grouped.items():
        means = np.mean([item["fused"] for item in items], axis=0)
        order = np.argsort(-means)
        root = items[0]["root"]
        rank = int(np.flatnonzero(np.array(services)[order] == root)[0]) + 1
        ranks.append(rank)
    report["cpca_only"] = {
        "events": len(ranks),
        "hit_at_1": float(np.mean([r == 1 for r in ranks])),
        "hit_at_3": float(np.mean([r <= 3 for r in ranks])),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }
    report["best_lambda"] = best[0] if best else None
    report["best"] = best[1] if best else None
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
