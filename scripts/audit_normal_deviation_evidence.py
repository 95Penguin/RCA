#!/usr/bin/env python3
"""Audit root separability of service-conditioned normal-state deviations."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset
from models import LTFPBaseline
from training.engine import choose_device, fit_feature_scaler, seed_everything
from training.metrics import rca_metrics


DATASETS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}


def _checkpoint(dataset: str, fold: int) -> Path:
    return ROOT / (
        f"artifacts/re2_{dataset}_ltfp_deployment_fold{fold}_"
        "fair_30ep_seed42_baseline_v1/best_model.pt"
    )


def _healthy_train_ends(
    dataset: Re2WindowDataset, train_ends: list[int]
) -> list[int]:
    """Require the complete input history and next target to be normal."""
    healthy = []
    for end in train_ends:
        start = end - dataset.history + 1
        if start == 0 or dataset.case_ids[start - 1] != dataset.case_ids[end]:
            continue
        if not dataset.failure_targets[start - 1 : end + 1].any():
            healthy.append(end)
    return healthy


def _robust_reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return training-only median and a stable robust dispersion estimate."""
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    standard = values.std(axis=0)
    scale = np.where(mad > 1e-3, mad, np.where(standard > 1e-3, standard, 1.0))
    return median.astype(np.float32), scale.astype(np.float32)


def _deviation(values: np.ndarray, reference: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    median, scale = reference
    return np.clip(np.abs((values - median) / scale), 0.0, 10.0)


def _event_metrics(
    dataset: Re2WindowDataset,
    ends: list[int],
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    rows_by_event: dict[str, list[int]] = {}
    for row, end in enumerate(ends):
        event_id = dataset.event_ids[end]
        if event_id and targets[row].sum() > 0:
            rows_by_event.setdefault(event_id, []).append(row)
    event_targets = []
    event_scores = []
    for event_id, rows in rows_by_event.items():
        group_targets = targets[rows]
        if not np.all(group_targets == group_targets[0]):
            raise ValueError(f"root labels change within event {event_id}")
        event_targets.append(group_targets[0])
        event_scores.append(scores[rows].mean(axis=0))
    result = rca_metrics(np.stack(event_targets), np.stack(event_scores))
    result["positive_events"] = result.pop("positive_windows")
    return result


def _score_diagnostics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    truth = targets.astype(bool)
    true_score = np.where(truth, scores, -np.inf).max(axis=1)
    other_score = np.where(~truth, scores, -np.inf).max(axis=1)
    margin = true_score - other_score
    return {
        "mean_root_margin": float(margin.mean()),
        "median_root_margin": float(np.median(margin)),
        "positive_root_margin_fraction": float((margin > 0).mean()),
    }


def _feature_groups(columns: tuple[str, ...]) -> dict[str, list[int]]:
    groups = {
        "metric": [i for i, name in enumerate(columns) if name.startswith("metric_")],
        "log": [
            i
            for i, name in enumerate(columns)
            if name.startswith("log_") and name != "log_observed"
        ],
        "trace": [
            i
            for i, name in enumerate(columns)
            if name.startswith(("trace_", "entry_"))
            and not name.startswith("observed_")
        ],
    }
    groups["all"] = sorted({index for values in groups.values() for index in values})
    return groups


def _embedding_arrays(
    model: LTFPBaseline,
    dataset: Re2WindowDataset,
    ends: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {end: row for row, end in enumerate(dataset.sample_indices)}
    loader = DataLoader(
        Subset(dataset, [index[end] for end in ends]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    local_states = []
    graph_states = []
    root_scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            local, graph, application = model._encode_paths(
                batch["node_features"].to(device), batch["adjacency"].to(device)
            )
            local_states.append(local.cpu().numpy())
            graph_states.append(graph.cpu().numpy())
            output = model._output(
                graph, application, model.failure_head(application)
            )
            root_scores.append(output["node_scores"].cpu().numpy())
    return (
        np.concatenate(local_states),
        np.concatenate(graph_states),
        np.concatenate(root_scores),
    )


def _evaluate_scores(
    dataset: Re2WindowDataset,
    ends: list[int],
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    return {
        "window": rca_metrics(targets, scores),
        "event": _event_metrics(dataset, ends, targets, scores),
        "diagnostics": _score_diagnostics(targets, scores),
    }


def _run_fold(
    dataset_name: str, fold: int, device: torch.device, batch_size: int
) -> dict[str, object]:
    seed_everything(42)
    dataset = Re2WindowDataset(
        DATASETS[dataset_name], history=5, split_policy="deployment", split_fold=fold
    )
    train_ends, _, test_ends = dataset.predefined_splits
    checkpoint_path = _checkpoint(dataset_name, fold)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean, scale = fit_feature_scaler(dataset, train_ends)
    if not (
        np.allclose(mean, checkpoint["feature_mean"], rtol=1e-5, atol=1e-6)
        and np.allclose(scale, checkpoint["feature_scale"], rtol=1e-5, atol=1e-6)
    ):
        raise ValueError("recomputed train scaler differs from checkpoint")

    healthy_ends = _healthy_train_ends(dataset, train_ends)
    attributed_ends = [end for end in test_ends if dataset.node_targets[end].sum() > 0]
    targets = dataset.node_targets[attributed_ends]
    healthy_windows = np.stack(
        [dataset.features[end - dataset.history + 1 : end + 1] for end in healthy_ends]
    )
    test_windows = np.stack(
        [dataset.features[end - dataset.history + 1 : end + 1] for end in attributed_ends]
    )
    raw_deviation = _deviation(test_windows, _robust_reference(healthy_windows))

    methods: dict[str, dict[str, object]] = {}
    for group_name, indices in _feature_groups(dataset.feature_columns).items():
        group_deviation = np.take(raw_deviation, indices, axis=-1)
        latest = group_deviation[:, -1].mean(axis=-1)
        history_max = group_deviation.max(axis=1).mean(axis=-1)
        history_mean = group_deviation.mean(axis=(1, 3))
        methods[f"raw_{group_name}_latest"] = _evaluate_scores(
            dataset, attributed_ends, targets, latest
        )
        methods[f"raw_{group_name}_history_max"] = _evaluate_scores(
            dataset, attributed_ends, targets, history_max
        )
        methods[f"raw_{group_name}_history_mean"] = _evaluate_scores(
            dataset, attributed_ends, targets, history_mean
        )

    config = checkpoint["model_config"]
    model = LTFPBaseline(
        feature_dim=config["feature_dim"], feature_groups=config["feature_groups"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    healthy_local, healthy_graph, _ = _embedding_arrays(
        model, dataset, healthy_ends, device, batch_size
    )
    test_local, test_graph, baseline_root_scores = _embedding_arrays(
        model, dataset, attributed_ends, device, batch_size
    )
    methods["ltfp_supervised_root_head"] = _evaluate_scores(
        dataset, attributed_ends, targets, baseline_root_scores
    )
    for name, healthy_values, test_values in (
        ("ltfp_local", healthy_local, test_local),
        ("ltfp_graph", healthy_graph, test_graph),
    ):
        scores = _deviation(
            test_values, _robust_reference(healthy_values)
        ).mean(axis=-1)
        methods[f"{name}_deviation"] = _evaluate_scores(
            dataset, attributed_ends, targets, scores
        )

    return {
        "dataset": dataset_name,
        "fold": fold,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "healthy_train_windows": len(healthy_ends),
        "test_root_windows": len(attributed_ends),
        "test_root_events": len(
            {dataset.event_ids[end] for end in attributed_ends if dataset.event_ids[end]}
        ),
        "methods": methods,
    }


def _summary(folds: list[dict[str, object]]) -> dict[str, object]:
    methods = folds[0]["methods"]
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in ("hit_at_1", "hit_at_3", "mrr"):
            values = [fold["methods"][method]["event"][metric] for fold in folds]
            summary[method][metric] = {
                "mean": statistics.fmean(values),
                "values": values,
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = choose_device(args.device)
    folds = [
        _run_fold(args.dataset, fold, device, args.batch_size) for fold in args.folds
    ]
    report = {
        "protocol": {
            "normal_reference": "training-only fully healthy windows",
            "reference_statistic": "service/position/feature median with robust scale",
            "deviation_clip": 10.0,
            "test_selection": "attributed failure windows only",
        },
        "folds": folds,
        "summary": _summary(folds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for fold in folds:
        print(f"{fold['dataset']} fold={fold['fold']}")
        ordered = sorted(
            fold["methods"].items(),
            key=lambda item: item[1]["event"]["mrr"],
            reverse=True,
        )
        for name, result in ordered[:8]:
            event = result["event"]
            print(
                f"  {name}: Hit@1={event['hit_at_1']:.3f} "
                f"Hit@3={event['hit_at_3']:.3f} MRR={event['mrr']:.3f}"
            )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
