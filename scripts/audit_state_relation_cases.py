#!/usr/bin/env python3
"""Event-level RCA replay with sidecar-only topology perturbations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from run_state_relation_pilot import ROOT, evaluate
from data.re2 import Re2WindowDataset
from models.factory import build_model
from training.engine import fit_feature_scaler
from training.experiment import _event_rca_metrics


def event_rows(dataset, ends, outputs):
    events = {}
    for row, end in enumerate(ends):
        event = dataset.event_ids[end]
        if event and dataset.node_targets[end].sum() > 0:
            events.setdefault(event, []).append(row)
    result = []
    for event, rows in events.items():
        truth = dataset.node_targets[ends[rows[0]]].astype(bool)
        record = {"event": event, "root_services": list(np.array(dataset.services)[truth]),
                  "windows": len(rows), "methods": {}}
        for name, output in outputs.items():
            scores = output["node_scores"][rows].mean(axis=0)
            order = np.argsort(-scores, kind="stable")
            rank = next(i + 1 for i, index in enumerate(order) if truth[index])
            first_order = np.argsort(-output["node_scores"][rows[0]], kind="stable")
            first_rank = next(i + 1 for i, index in enumerate(first_order) if truth[index])
            record["methods"][name] = {
                "root_rank": rank, "first_available_window_root_rank": first_rank,
                "top3": [dataset.services[index] for index in order[:3]],
                "root_score": float(scores[truth].max()),
                "top_nonroot_score": float(scores[~truth].max()),
            }
        result.append(record)
    return result


def run(fold, checkpoint_dir, control_dirs):
    dataset = Re2WindowDataset(ROOT / "artifacts/re2_ob_panel_fair_v1", history=5,
                              split_policy="deployment", split_fold=fold)
    train, val, test = dataset.predefined_splits
    baseline_path = ROOT / f"artifacts/re2_ob_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1/best_model.pt"
    baseline = build_model("ltfp", feature_columns=dataset.feature_columns, num_services=len(dataset.services)).model
    state = torch.load(baseline_path, map_location="cpu", weights_only=False)
    mean, scale = fit_feature_scaler(dataset, train)
    np.testing.assert_allclose(mean, state["feature_mean"], atol=1e-6)
    np.testing.assert_allclose(scale, state["feature_scale"], atol=1e-6)
    baseline.load_state_dict(state["model_state_dict"])
    indices = {end: i for i, end in enumerate(dataset.sample_indices)}
    subset = Subset(dataset, [indices[end] for end in test])
    original_loader = DataLoader(subset, batch_size=64)
    base = evaluate(baseline, original_loader, torch.device("cpu"))
    samples = []
    for i in range(len(subset)):
        sample = dict(subset[i])
        sample["baseline_node_logits"] = torch.from_numpy(base["node_logits"][i])
        sample["baseline_failure_logits"] = torch.from_numpy(base["failure_logits"][i])
        samples.append(sample)
    loader = DataLoader(samples, batch_size=64)
    outputs = {"baseline": base}
    replay = []
    graph_changes = []
    for variant in ("local-normal", "state-relation"):
        model = build_model(f"ltfp-{variant}-evidence", feature_columns=dataset.feature_columns,
                            num_services=len(dataset.services), history=5).model
        checkpoint = torch.load(checkpoint_dir / f"ob_fold{fold}_{variant}.pt", map_location="cpu", weights_only=False)
        missing = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing.unexpected_keys or set(missing.missing_keys) - {"relation_permutation"}:
            raise ValueError(missing)
        outputs[variant] = evaluate(model, loader, torch.device("cpu"))
        if variant == "state-relation":
            model.use_relation_graph = False
            outputs["relation_identity_replay"] = evaluate(model, loader, torch.device("cpu"))
            model.use_relation_graph = True
            for seed in range(2026, 2036):
                permutation = np.random.default_rng(seed).permutation(len(dataset.services))
                model.relation_permutation.copy_(torch.from_numpy(permutation))
                output = evaluate(model, loader, torch.device("cpu"))
                name = f"relation_permuted_{seed}"
                outputs[name] = output
                replay.append({"seed": seed, "metrics": _event_rca_metrics(dataset, test, output)})
                graph = dataset.adjacency[test].numpy()
                changed = graph[:, permutation][:, :, permutation]
                offdiag = ~np.eye(len(dataset.services), dtype=bool)
                true_edges = graph[:, offdiag] > 0
                shuffled_edges = changed[:, offdiag] > 0
                union = np.logical_or(true_edges, shuffled_edges).sum()
                graph_changes.append({"seed": seed, "offdiag_edge_jaccard": float(np.logical_and(true_edges, shuffled_edges).sum() / max(union, 1))})
    for directory in control_dirs:
        checkpoint = torch.load(directory / f"ob_fold{fold}_state-relation.pt", map_location="cpu", weights_only=False)
        model = build_model("ltfp-state-relation-evidence", feature_columns=dataset.feature_columns,
                            num_services=len(dataset.services), history=5).model
        model.load_state_dict(checkpoint["model_state_dict"])
        outputs[f"trained_{directory.name}"] = evaluate(model, loader, torch.device("cpu"))
    split_roots = {}
    for name, ends in (("train", train), ("validation", val), ("test", test)):
        groups = {}
        for end in ends:
            for i in np.flatnonzero(dataset.node_targets[end]):
                groups.setdefault(dataset.services[i], set()).add(dataset.event_ids[end])
        split_roots[name] = {service: len(events) for service, events in groups.items()}
    events = event_rows(dataset, test, outputs)
    return {"dataset": "ob", "fold": fold, "root_event_counts": split_roots,
            "metrics": {name: _event_rca_metrics(dataset, test, output) for name, output in outputs.items()},
            "permutation_replay": replay, "graph_changes": graph_changes, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-dirs", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    rows = [run(fold, args.checkpoint_dir, args.control_dirs) for fold in (3, 4)]
    args.output.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print("fold", row["fold"], {key: value["mrr"] for key, value in row["metrics"].items()})


if __name__ == "__main__":
    main()
