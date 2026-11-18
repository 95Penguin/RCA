#!/usr/bin/env python3
"""Audit task-gradient conflict on a trained LTFP shared graph encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset  # noqa: E402
from models.factory import build_model  # noqa: E402
from training.engine import (  # noqa: E402
    choose_device,
    class_pos_weight,
    fit_feature_scaler,
    seed_everything,
)


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[nn.Parameter],
) -> torch.Tensor:
    return torch.cat([
        gradient.reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.norm() * second.norm()
    if denominator <= 0:
        return float("nan")
    return float(torch.dot(first, second).div(denominator).detach().cpu())


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not finite.size:
        return {"batches": 0, "mean": None, "median": None, "negative_fraction": None}
    return {
        "batches": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "negative_fraction": float((finite < 0).mean()),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset = Re2WindowDataset(
        args.data_dir,
        history=5,
        split_policy="deployment",
        split_fold=args.split_fold,
        horizon_minutes=1,
    )
    train_ends, _, _ = dataset.predefined_splits
    fit_feature_scaler(dataset, train_ends)
    index = {end: row for row, end in enumerate(dataset.sample_indices)}
    loader = DataLoader(
        Subset(dataset, [index[end] for end in train_ends]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if tuple(checkpoint["feature_columns"]) != dataset.feature_columns:
        raise ValueError("checkpoint feature columns do not match the dataset")
    built = build_model(
        "ltfp",
        feature_columns=dataset.feature_columns,
        num_services=len(dataset.services),
    )
    model = built.model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graph_parameters = [
        *model.graph_layers.parameters(),
        *model.graph_norms.parameters(),
    ]
    train_labels = torch.tensor(
        dataset.failure_targets[train_ends], dtype=torch.float32
    )
    positive_weight = class_pos_weight(train_labels, torch.device("cpu")).item()
    failure_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, positive_weight], device=device)
    )

    mixed_cosines: list[float] = []
    positive_cosines: list[float] = []
    mixed_aggregate = torch.zeros(
        sum(parameter.numel() for parameter in graph_parameters), device=device
    )
    positive_aggregate = torch.zeros_like(mixed_aggregate)
    root_aggregate = torch.zeros_like(mixed_aggregate)
    attributed_examples = 0

    for batch in loader:
        features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["failure_target"].to(device)
        node_targets = batch["node_targets"].to(device)
        node_valid = batch.get(
            "node_label_valid", torch.ones_like(labels, dtype=torch.bool)
        ).to(device)
        rank_valid = node_valid & (labels == 1) & (node_targets.sum(dim=1) > 0)
        if not rank_valid.any():
            continue

        output = model(features, adjacency)
        failure_mixed = failure_loss_fn(output["failure_logits"], labels)
        failure_positive = failure_loss_fn(
            output["failure_logits"][rank_valid], labels[rank_valid]
        )
        roots = node_targets[rank_valid]
        roots = roots / roots.sum(dim=1, keepdim=True)
        root_loss = -(
            roots * torch.log_softmax(output["node_logits"][rank_valid], dim=1)
        ).sum(dim=1).mean()

        mixed_gradient = _flatten_gradients(
            torch.autograd.grad(
                failure_mixed, graph_parameters, retain_graph=True, allow_unused=True
            ),
            graph_parameters,
        )
        positive_gradient = _flatten_gradients(
            torch.autograd.grad(
                failure_positive, graph_parameters, retain_graph=True, allow_unused=True
            ),
            graph_parameters,
        )
        root_gradient = _flatten_gradients(
            torch.autograd.grad(root_loss, graph_parameters, allow_unused=True),
            graph_parameters,
        )
        mixed_cosines.append(_cosine(mixed_gradient, root_gradient))
        positive_cosines.append(_cosine(positive_gradient, root_gradient))
        examples = int(rank_valid.sum().item())
        mixed_aggregate += mixed_gradient * examples
        positive_aggregate += positive_gradient * examples
        root_aggregate += root_gradient * examples
        attributed_examples += examples

    result = {
        "data_dir": str(args.data_dir),
        "checkpoint": str(args.checkpoint),
        "split_policy": "deployment",
        "split_fold": args.split_fold,
        "audited_split": "training",
        "graph_parameter_count": sum(
            parameter.numel() for parameter in graph_parameters
        ),
        "attributed_positive_examples": attributed_examples,
        "mixed_batch_failure_vs_root": {
            **_summary(mixed_cosines),
            "aggregate_cosine": _cosine(mixed_aggregate, root_aggregate),
        },
        "positive_failure_vs_root": {
            **_summary(positive_cosines),
            "aggregate_cosine": _cosine(positive_aggregate, root_aggregate),
        },
        "interpretation_boundary": (
            "Negative cosine is evidence of optimization conflict on the shared graph "
            "parameters, not proof that the tasks require different causal graphs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
