#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASET_ORDER = ["re1-ob", "re1-ss", "re1-tt", "re2-ss", "re3-ob", "re3-ss", "re3-tt"]
VARIANT_ORDER = [
    "full",
    "no_logs_traces",
    "no_graph",
    "no_lead_time",
    "no_aux_tasks",
    "no_delta",
]
VARIANT_LABELS = {
    "full": "Full LTFP",
    "no_logs_traces": "w/o logs and traces",
    "no_graph": "w/o graph propagation",
    "no_lead_time": "w/o lead-time awareness",
    "no_aux_tasks": "w/o root-cause supervision",
    "no_delta": "w/o temporal-delta features",
}


def load_variant(root: Path, dataset: str, variant: str, horizon: float) -> dict | None:
    out_dir = root / dataset / variant
    result_path = out_dir / "rcaeval-cases_ltfp_tcn_gat_results.csv"
    summary_path = out_dir / "rcaeval-cases_ltfp_tcn_gat_summary.json"
    if not result_path.exists() or not summary_path.exists():
        return None

    results = pd.read_csv(result_path)
    if results.empty:
        return None
    idx = (results["horizon"].astype(float) - horizon).abs().idxmin()
    row = results.loc[idx].to_dict()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    root_summary = summary.get("root_summary") or {}
    return {
        "dataset": dataset,
        "variant": variant,
        "method": VARIANT_LABELS.get(variant, variant),
        "horizon": float(row["horizon"]),
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "f1": float(row.get("f1", 0.0)),
        "auprc": float(row.get("auprc", 0.0)),
        "root_r3": root_summary.get("root_top3"),
        "modalities": "+".join(summary.get("modality_names") or []),
        "n_graph_layers": summary.get("n_graph_layers"),
        "include_delta_features": summary.get("include_delta_features"),
        "horizon_weight_mode": summary.get("horizon_weight_mode"),
        "lead_weight": (summary.get("loss_weights") or {}).get("lead"),
        "root_weight": (summary.get("loss_weights") or {}).get("root"),
    }


def write_markdown(avg: pd.DataFrame, path: Path) -> None:
    display = avg[["method", "precision", "recall", "f1", "auprc"]].copy()
    display.columns = ["Variant", "P", "R", "F1", "AUPRC"]
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(avg: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Ablation study averaged over all RCAEval subsets at the default 600 s horizon.}",
        "\\label{tab:ablation_results}",
        "\\footnotesize",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Variant} & \\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{AUPRC} \\\\",
        "\\hline",
    ]
    for _, row in avg.iterrows():
        method = str(row["method"]).replace("_", "\\_")
        lines.append(
            f"{method} & {row['precision']:.3f} & {row['recall']:.3f} & {row['f1']:.3f} & "
            f"{row['auprc']:.3f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/ltfp_ablation")
    parser.add_argument("--horizon", type=float, default=600.0)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for dataset in DATASET_ORDER:
        for variant in VARIANT_ORDER:
            row = load_variant(root, dataset, variant, args.horizon)
            if row is not None:
                rows.append(row)

    if not rows:
        raise SystemExit(f"no ablation results found under {root}")

    frame = pd.DataFrame(rows)
    variant_rank = {name: idx for idx, name in enumerate(VARIANT_ORDER)}
    dataset_rank = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    frame["variant_rank"] = frame["variant"].map(variant_rank)
    frame["dataset_rank"] = frame["dataset"].map(dataset_rank)
    frame = frame.sort_values(["variant_rank", "dataset_rank"]).drop(columns=["variant_rank", "dataset_rank"])

    metric_cols = ["precision", "recall", "f1", "auprc", "root_r3"]
    avg = (
        frame.groupby(["variant", "method"], sort=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    avg["variant_rank"] = avg["variant"].map(variant_rank)
    avg = avg.sort_values("variant_rank").drop(columns=["variant_rank"])

    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "ablation_by_dataset.csv", index=False)
    avg.to_csv(root / "ablation_summary.csv", index=False)
    write_markdown(avg, root / "ablation_summary.md")
    write_latex(avg, root / "ablation_table.tex")

    print((root / "ablation_summary.md").read_text(encoding="utf-8"))
    print(f"Saved: {root / 'ablation_by_dataset.csv'}")
    print(f"Saved: {root / 'ablation_summary.csv'}")
    print(f"Saved: {root / 'ablation_table.tex'}")


if __name__ == "__main__":
    main()
