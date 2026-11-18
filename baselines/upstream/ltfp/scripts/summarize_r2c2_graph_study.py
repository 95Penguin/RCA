#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = ["re1-ob", "re1-ss", "re1-tt", "re2-ss", "re3-ob", "re3-ss", "re3-tt"]
MODES = [
    "hybrid-max",
    "dependency-only",
    "correlation-only",
    "hybrid-learned",
    "hybrid-lagged",
    "hybrid-random",
]
LABELS = {
    "hybrid-max": "Hybrid-max (current)",
    "dependency-only": "Dependency-only",
    "correlation-only": "Correlation-only",
    "hybrid-learned": "Hybrid-learned",
    "hybrid-lagged": "Hybrid-lagged",
    "hybrid-random": "Hybrid-random",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/r2c2_graph_study")
    parser.add_argument("--horizon", type=float, default=600.0)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for dataset in DATASETS:
        for mode in MODES:
            directory = root / dataset / mode
            result_path = directory / "rcaeval-cases_ltfp_tcn_gat_results.csv"
            summary_path = directory / "rcaeval-cases_ltfp_tcn_gat_summary.json"
            if not result_path.exists() or not summary_path.exists():
                continue
            result = pd.read_csv(result_path)
            row = result.loc[(result.horizon.astype(float) - args.horizon).abs().idxmin()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            root_summary = summary["root_summary"]
            learned = summary.get("learned_source_weights")
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "label": LABELS[mode],
                    "precision": float(row.precision),
                    "recall": float(row.recall),
                    "f1": float(row.f1),
                    "auprc": float(row.auprc),
                    "top1": float(root_summary["root_top1"]),
                    "top3": float(root_summary["root_top3"]),
                    "dependency_weight": learned[0] if learned else None,
                    "correlation_weight": learned[1] if learned else None,
                }
            )
    if not rows:
        raise SystemExit(f"no graph-study results found under {root}")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "graph_study_by_dataset.csv", index=False)
    summary = (
        frame.groupby(["mode", "label"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    order = {mode: index for index, mode in enumerate(MODES)}
    summary["order"] = summary["mode"].map(order)
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(root / "graph_study_summary.csv", index=False)

    lines = [
        "| Graph | F1 | AUPRC | Top-1 | Top-3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.label} | {row.f1:.3f} | {row.auprc:.3f} | {row.top1:.3f} | {row.top3:.3f} |"
        )
    (root / "graph_study_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "graph_study_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
