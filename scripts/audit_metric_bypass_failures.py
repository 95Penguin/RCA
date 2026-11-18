#!/usr/bin/env python3
"""Replay existing calibrators to diagnose harmed events; no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from audit_state_relation_cases import event_rows
from run_state_relation_pilot import ROOT, evaluate, validation_rank_loss
from data.re2 import Re2WindowDataset
from models.factory import build_model
from training.engine import fit_feature_scaler
from training.experiment import _configure_root_event_weights, _event_rca_metrics


def root_coverage(dataset, ends):
    events = {}
    for end in ends:
        for i in np.flatnonzero(dataset.node_targets[end]):
            events.setdefault(dataset.services[i], set()).add(dataset.event_ids[end])
    return {service: len(ids) for service, ids in events.items()}


def audit(name, fold, seed):
    dataset = Re2WindowDataset(ROOT / f"artifacts/re2_{name}_panel_fair_v1", history=5,
                              split_policy="deployment", split_fold=fold)
    splits = dict(zip(("train", "validation", "test"), dataset.predefined_splits))
    _configure_root_event_weights(dataset, dataset.predefined_splits, True)
    baseline_path = ROOT / f"artifacts/re2_{name}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1/best_model.pt"
    checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    mean, scale = fit_feature_scaler(dataset, splits["train"])
    np.testing.assert_allclose(mean, checkpoint["feature_mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scale, checkpoint["feature_scale"], rtol=1e-5, atol=1e-6)
    baseline = build_model("ltfp", feature_columns=dataset.feature_columns, num_services=len(dataset.services)).model
    baseline.load_state_dict(checkpoint["model_state_dict"])
    index = {end: i for i, end in enumerate(dataset.sample_indices)}
    loaders, outputs = {}, {}
    for split in ("validation", "test"):
        subset = Subset(dataset, [index[end] for end in splits[split]])
        output = evaluate(baseline, DataLoader(subset, batch_size=64), torch.device("cpu"))
        outputs[split] = {"baseline": output}
        samples = []
        for row in range(len(subset)):
            sample = dict(subset[row])
            sample["baseline_node_logits"] = torch.from_numpy(output["node_logits"][row])
            sample["baseline_failure_logits"] = torch.from_numpy(output["failure_logits"][row])
            samples.append(sample)
        loaders[split] = DataLoader(samples, batch_size=64)
    directory = Path(f"/private/tmp/metric_bypass_seed{seed}_20261118")
    window_changes = []
    for variant in ("raw-metric-bypass", "metric-deviation-bypass"):
        checkpoint = torch.load(directory / f"{name}_fold{fold}_{variant}.pt", map_location="cpu", weights_only=False)
        model = build_model(f"ltfp-{variant}-evidence", feature_columns=dataset.feature_columns,
                            num_services=len(dataset.services), history=5).model
        model.load_state_dict(checkpoint["model_state_dict"])
        for split in loaders:
            outputs[split][variant] = evaluate(model, loaders[split], torch.device("cpu"))
        ends = splits["test"]
        base = outputs["test"]["baseline"]
        updated = outputs["test"][variant]
        counts = {"baseline_correct_windows": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}
        for row, end in enumerate(ends):
            truth = base["node_targets"][row].astype(bool)
            if not truth.any():
                continue
            old_top = int(base["node_scores"][row].argmax())
            new_top = int(updated["node_scores"][row].argmax())
            old_correct, new_correct = bool(truth[old_top]), bool(truth[new_top])
            counts["baseline_correct_windows"] += int(old_correct)
            counts["correct_to_wrong"] += int(old_correct and not new_correct)
            counts["wrong_to_correct"] += int(not old_correct and new_correct)
            if old_correct and not new_correct:
                root = int(np.flatnonzero(truth)[0])
                old = base["node_logits"][row]
                new = updated["node_logits"][row]
                x = dataset.features[end-dataset.history+1:end+1]
                values = torch.from_numpy(x[..., list(model.metric_value_indices)])
                masks = model._metric_masks(torch.from_numpy(x).unsqueeze(0)).squeeze(0)
                dev = ((values-model.normal_median).abs()/model.normal_scale).clamp(max=10)*masks
                window_changes.append({
                    "variant": variant, "event": dataset.event_ids[end],
                    "input_timestamp": dataset.timestamps[end],
                    "root": dataset.services[root], "new_top": dataset.services[new_top],
                    "baseline_top_probability": float(base["node_scores"][row, old_top]),
                    "baseline_margin_to_new_top": float(old[root]-old[new_top]),
                    "updated_margin_to_new_top": float(new[root]-new[new_top]),
                    "root_residual": float(new[root]-old[root]),
                    "competitor_residual": float(new[new_top]-old[new_top]),
                    "root_latest_deviation": float(dev[-1, root].mean()),
                    "competitor_latest_deviation": float(dev[-1, new_top].mean()),
                })
        counts["variant"] = variant
        counts["high_confidence_correct_to_wrong"] = sum(
            row["variant"] == variant and row["baseline_top_probability"] >= 0.9 for row in window_changes
        )
        if variant == "raw-metric-bypass":
            raw_counts = counts
        else:
            deviation_counts = counts
    return {
        "dataset": name, "fold": fold, "seed": seed,
        "root_coverage": {split: root_coverage(dataset, ends) for split, ends in splits.items()},
        "split_metrics": {split: {method: {
            "event_mrr": _event_rca_metrics(dataset, splits[split], output)["mrr"],
            "event_balanced_ce": validation_rank_loss(dataset, splits[split], output),
        } for method, output in methods.items()} for split, methods in outputs.items()},
        "events": event_rows(dataset, splits["test"], outputs["test"]),
        "window_counts": [raw_counts, deviation_counts],
        "harmed_windows": window_changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(name, fold, seed) for name, fold in (("ob", 0), ("tt", 3)) for seed in (42, 43)]
    args.output.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row["dataset"], row["seed"], row["window_counts"], flush=True)


if __name__ == "__main__":
    main()
