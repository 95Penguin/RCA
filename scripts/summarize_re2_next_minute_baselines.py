#!/usr/bin/env python3
"""Combine all generic and paper-derived next-minute RE2 baselines."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDS = range(5)
MODELS = (
    "last-mlp", "tcn", "gru", "gcn-gru", "instantops-adapted",
    "ltfp", "eadro-adapted",
)


def _directory(dataset: str, model: str, fold: int) -> Path:
    token = model.replace("-", "_")
    suffix = "baseline_v1" if model in {
        "last-mlp", "tcn", "instantops-adapted", "ltfp", "eadro-adapted"
    } else "v3"
    return ROOT / (
        f"artifacts/re2_{dataset}_{token}_deployment_fold{fold}_fair_30ep_seed42_{suffix}"
    )


def _event_ranks(prediction_path: Path) -> dict[str, float | int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in prediction_path.read_text().splitlines():
        row = json.loads(line)
        if row["event_id"] and row["failure_target"] and row["true_services"]:
            grouped[row["event_id"]].append(row)
    first_ranks = []
    for rows in grouped.values():
        truth = set(rows[0]["true_services"])
        if any(set(row["true_services"]) != truth for row in rows):
            raise ValueError(f"root labels vary inside event {rows[0]['event_id']}")
        scores: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for candidate in row["ranking"]:
                scores[candidate["service"]].append(float(candidate["score"]))
        order = sorted(scores, key=lambda service: (-statistics.fmean(scores[service]), service))
        first_ranks.append(min(order.index(service) + 1 for service in truth))
    return {
        "events": len(first_ranks),
        "hit_at_1": statistics.fmean(rank <= 1 for rank in first_ranks),
        "hit_at_3": statistics.fmean(rank <= 3 for rank in first_ranks),
        "hit_at_5": statistics.fmean(rank <= 5 for rank in first_ranks),
        "mrr": statistics.fmean(1 / rank for rank in first_ranks),
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    result: dict[str, dict] = {}
    for dataset in ("ob", "tt"):
        dataset_name = f"RE2-{dataset.upper()}"
        result[dataset_name] = {}
        for model in MODELS:
            folds = []
            for fold in FOLDS:
                directory = _directory(dataset, model, fold)
                metrics = json.loads((directory / "metrics.json").read_text())
                detection = metrics["failure_prediction"]
                folds.append({
                    "fold": fold,
                    "precision": detection["precision_at_0_5"],
                    "recall": detection["recall_at_0_5"],
                    "f1": detection["f1_at_0_5"],
                    "ap": detection["average_precision"],
                    **_event_ranks(directory / "predictions.jsonl"),
                })
            result[dataset_name][model] = {
                key: _mean_sd([float(row[key]) for row in folds])
                for key in (
                    "precision", "recall", "f1", "ap",
                    "hit_at_1", "hit_at_3", "hit_at_5", "mrr",
                )
            }
            result[dataset_name][model]["folds"] = folds

    report = {
        "protocol": {
            "task": "past five complete minutes predict next-minute business-degradation state",
            "root_task": "rank the injected root service for attributed next-minute failure events",
            "split": "deployment-group five-fold holdout",
            "epochs": 30,
            "seed": 42,
            "aggregation": "unweighted mean and sample SD across five test folds",
            "threshold": 0.5,
            "rca_unit": "event; mean service score over attributed failure windows",
            "paper_baseline_scope": {
                "ltfp": (
                    "task-aligned architecture baseline: modality-specific temporal encoders, "
                    "gated fusion, weighted graph attention, graph/node heads; uses the common "
                    "causal latest-minute dependency graph rather than LTFP's separately fitted "
                    "hybrid correlation graph"
                ),
                "eadro-adapted": (
                    "architecture-transfer baseline: Eadro multimodal encoders, GLU fusion, "
                    "GATv2-style propagation, attention pooling, detector/localizer; trained on "
                    "future next-minute labels instead of Eadro's original current-state task"
                ),
                "instantops-adapted": (
                    "task-aligned InstantOps architecture: two-layer GCN plus GRU trained "
                    "only for next-minute failure prediction; service masking approximates "
                    "the paper's permutation-based root-cause importance"
                ),
            },
        },
        "results": result,
    }
    json_path = ROOT / "artifacts/re2_next_minute_all_baselines_seed42_v1.json"
    json_path.write_text(json.dumps(report, indent=2))

    rows = [
        "# RE2 next-minute baselines (seed 42)",
        "",
        "All values are unweighted mean±sample SD over five deployment-held-out folds.",
        "Detection uses threshold 0.5; RCA is retrospective event-level ranking.",
        "LTFP is task-aligned. Eadro-adapted ports Eadro's reactive architecture to the future label and is not an exact reproduction of Eadro's original task.",
        "",
        "| Dataset | Model | Precision | Recall | F1 | AP | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "last-mlp": "Last-MLP",
        "tcn": "TCN",
        "gru": "GRU",
        "gcn-gru": "MTG (GCN+GRU)",
        "instantops-adapted": "InstantOps-adapted",
        "ltfp": "LTFP",
        "eadro-adapted": "Eadro-adapted",
    }
    for dataset, models in result.items():
        for model in MODELS:
            values = models[model]
            formatted = [
                f"{values[key]['mean']:.3f}±{values[key]['sd']:.3f}"
                for key in (
                    "precision", "recall", "f1", "ap",
                    "hit_at_1", "hit_at_3", "hit_at_5", "mrr",
                )
            ]
            rows.append(f"| {dataset} | {labels[model]} | " + " | ".join(formatted) + " |")
    rows.extend([
        "",
        "## Baseline scope",
        "",
        "- **LTFP** is a task-aligned architecture baseline: modality-specific temporal encoders, gated fusion, weighted graph attention, and graph/node multitask heads. It uses the common causal latest-minute dependency graph; it does not reproduce LTFP's separately fitted hybrid correlation graph.",
        "- **InstantOps-adapted** uses the paper's two-layer GCN plus GRU next-step predictor and trains only on failure labels. Service masking is the deterministic service-level approximation of its permutation-importance RCA; the paper does not provide enough released preprocessing/code detail for a bit-exact reproduction.",
        "- **Eadro-adapted** transfers Eadro's multimodal encoders, GLU fusion, GATv2-style propagation, attention pooling, and detector/localizer. Eadro originally performs current-state troubleshooting, so these numbers are not an exact Eadro task reproduction.",
        "- BARO is excluded from this prediction table because its Bayesian change-point detector and RobustScorer are reactive detection/RCA components. Relabeling them as a next-minute predictor would no longer be a faithful BARO baseline.",
        "",
        "## Immediate reading",
        "",
        "- InstantOps-adapted has the best RE2-OB detection F1/AP, but its perturbation-based localization is much weaker than LTFP's supervised service ranker.",
        "- LTFP is the strongest RE2-TT model for both detection and localization and gives the best localization balance on RE2-OB.",
        "- Eadro-adapted improves over MTG detection on both datasets, but its localization is inconsistent: strong Hit@3/5, weaker Hit@1.",
        "- The results use one seed. They establish runnable baselines, not final statistical significance; the paper table still needs multiple seeds or scenario-level confidence intervals.",
    ])
    markdown_path = ROOT / "docs/re2-next-minute-baselines.md"
    markdown_path.write_text("\n".join(rows) + "\n")
    print(markdown_path.read_text())
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
