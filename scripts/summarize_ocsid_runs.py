#!/usr/bin/env python3
"""Aggregate OCSID runs: three seeds, ablations, and CPCA fusion.

Produces a compact markdown table plus a JSON summary.  The prediction numbers
are seed means +- SD of the five-fold means.  The RCA headline uses the
CPCA-only column from the fusion artifact, because the learned head was shown to
be redundant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SEEDS = (42, 123, 2026)


def _load(path: Path):
    return json.loads(path.read_text())


def _seed_summary(dataset: str, suffix: str):
    rows = []
    for seed in SEEDS:
        path = ART / f"ocsid_joint_{dataset}_seed{seed}{suffix}.json"
        if not path.is_file():
            continue
        data = _load(path)
        calibration = [
            fold["calibration"]["test_f1_at_validation_threshold"]
            for fold in data["folds"] if fold.get("calibration")
        ]
        rows.append({
            "seed": seed,
            "ap": data["summary"]["ap"]["mean"],
            "f1": data["summary"]["f1"]["mean"],
            "calibrated_f1": float(np.mean(calibration)) if calibration else None,
            "event_hit": (data.get("event_summary") or {}).get("hit_at_1", {}).get("mean"),
            "event_mrr": (data.get("event_summary") or {}).get("mrr", {}).get("mean"),
        })
    if not rows:
        return None
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows if row[key] is not None])),
            "sd": float(np.std([row[key] for row in rows if row[key] is not None], ddof=1))
            if sum(row[key] is not None for row in rows) > 1 else 0.0,
        }
        for key in ("ap", "f1", "calibrated_f1", "event_hit", "event_mrr")
    }


def main() -> int:
    report = {}
    lines = []
    for dataset in ("ob", "tt"):
        main = _seed_summary(dataset, "_v3")
        fusion = _load(ART / f"ocsid_cpca_fusion_{dataset}_v1.json")
        nograph = _load(ART / f"ocsid_joint_{dataset}_nograph_v3.json")
        nooperation = _load(ART / f"ocsid_joint_{dataset}_nooperation_v3.json")
        report[dataset] = {
            "main_three_seeds": main,
            "cpca_only": fusion["cpca_only"],
            "fusion_learned_head": fusion["lambdas"]["0.0"],
            "nograph": {
                "ap": nograph["summary"]["ap"]["mean"],
                "f1": nograph["summary"]["f1"]["mean"],
            },
            "nooperation": {
                "ap": nooperation["summary"]["ap"]["mean"],
                "f1": nooperation["summary"]["f1"]["mean"],
            },
        }
        lines.append(f"### {dataset.upper()}")
        lines.append("")
        lines.append("| 指标 | 均值±标准差 |")
        lines.append("|---|---:|")
        if main:
            for key, label in (
                ("ap", "预测 AP"), ("f1", "预测 F1@0.5"),
                ("calibrated_f1", "校准后 F1"), ("event_hit", "学习头事件 H@1"),
                ("event_mrr", "学习头事件 MRR"),
            ):
                value = main[key]
                lines.append(f"| {label} | {value['mean']:.3f} ± {value['sd']:.3f} |")
        lines.append(f"| CPCA-only 事件 H@1 | {fusion['cpca_only']['hit_at_1']:.3f} |")
        lines.append(f"| CPCA-only 事件 MRR | {fusion['cpca_only']['mrr']:.3f} |")
        lines.append(f"| no-graph AP | {nograph['summary']['ap']['mean']:.3f} |")
        lines.append(f"| no-operation AP | {nooperation['summary']['ap']['mean']:.3f} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (ART / "ocsid_summary_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {ART / 'ocsid_summary_v3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
