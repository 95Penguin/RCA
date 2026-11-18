#!/usr/bin/env python3
"""Paired failure-only / RCA-supervised service-to-system risk pilot."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from run_state_relation_pilot import ROOT, evaluate, first_available_rca
from data.re2 import Re2WindowDataset
from models.factory import build_model
from models.service_risk import ServiceRiskInteraction
from training.engine import fit_feature_scaler, seed_everything
from training.experiment import _configure_root_event_weights, _event_rca_metrics
from training.metrics import classification_metrics


def predict(model, loader):
    probabilities, labels, residuals = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["features"], batch["failure_logits"], batch["root_logits"])
            probabilities.append(out["failure_logits"].softmax(-1)[:, 1].numpy())
            labels.append(batch["label"].numpy())
            residuals.append(out["system_residual"].numpy())
    y, p, d = np.concatenate(labels), np.concatenate(probabilities), np.concatenate(residuals)
    return classification_metrics(y, p), p, {"mean": float(d.mean()), "max_abs": float(np.abs(d).max()),
                                           "saturated_fraction": float((np.abs(d) > 0.99).mean())}


def run(name, fold, seed, args):
    dataset = Re2WindowDataset(ROOT / f"artifacts/re2_{name}_panel_fair_v1", history=5,
                              split_policy="deployment", split_fold=fold)
    splits = dict(zip(("train", "validation", "test"), dataset.predefined_splits))
    _configure_root_event_weights(dataset, dataset.predefined_splits, True)
    base_path = ROOT / f"artifacts/re2_{name}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1/best_model.pt"
    base_state = torch.load(base_path, map_location="cpu", weights_only=False)
    mean, scale = fit_feature_scaler(dataset, splits["train"])
    np.testing.assert_allclose(mean, base_state["feature_mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scale, base_state["feature_scale"], rtol=1e-5, atol=1e-6)
    baseline = build_model("ltfp", feature_columns=dataset.feature_columns, num_services=len(dataset.services)).model.eval()
    baseline.load_state_dict(base_state["model_state_dict"])
    rca_path = Path(f"/private/tmp/metric_bypass_seed{seed}_20261118/{name}_fold{fold}_metric-deviation-bypass.pt")
    rca_checkpoint = torch.load(rca_path, map_location="cpu", weights_only=False)
    rca_model = build_model("ltfp-metric-deviation-bypass-evidence", feature_columns=dataset.feature_columns,
                            num_services=len(dataset.services)).model.eval()
    rca_model.load_state_dict(rca_checkpoint["model_state_dict"])
    for model in (baseline, rca_model):
        model.requires_grad_(False)
    index = {end: row for row, end in enumerate(dataset.sample_indices)}
    cached, base_outputs, rca_outputs = {}, {}, {}
    for split, ends in splits.items():
        subset = Subset(dataset, [index[end] for end in ends])
        base = evaluate(baseline, DataLoader(subset, batch_size=64), torch.device("cpu"))
        base_outputs[split] = base
        samples, rca_scores = [], []
        with torch.no_grad():
            for start in range(0, len(subset), 64):
                rows = [subset[i] for i in range(start, min(start+64, len(subset)))]
                x = torch.stack([r["node_features"] for r in rows])
                graph = torch.stack([r["adjacency"] for r in rows])
                root = torch.from_numpy(base["node_logits"][start:start+len(rows)])
                features = rca_model.metric_evidence_features(x, root)
                old_rca = rca_model.forward_evidence(x, graph, root)["node_scores"].numpy()
                rca_scores.append(old_rca)
                for offset, sample in enumerate(rows):
                    end = ends[start+offset]
                    samples.append({"features": features[offset],
                                    "failure_logits": torch.from_numpy(base["failure_logits"][start+offset]),
                                    "root_logits": root[offset], "label": sample["failure_target"],
                                    "root": sample["node_targets"],
                                    "weight": torch.tensor(dataset.root_event_weights[end])})
        cached[split] = samples
        rca_outputs[split] = {**base, "node_scores": np.concatenate(rca_scores)}
    loaders = {split: DataLoader(rows, batch_size=64) for split, rows in cached.items()}
    class_counts = np.bincount(base_outputs["train"]["labels"], minlength=2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, class_counts[0]/max(class_counts[1],1)], dtype=torch.float32))
    row = {"dataset": name, "fold": fold, "seed": seed,
           "baseline": {"failure": classification_metrics(base_outputs["test"]["labels"], base_outputs["test"]["probabilities"]),
                        "rca": _event_rca_metrics(dataset, splits["test"], base_outputs["test"])},
           "frozen_rca": {"metrics": _event_rca_metrics(dataset, splits["test"], rca_outputs["test"]),
                          "first_available": first_available_rca(dataset, splits["test"], rca_outputs["test"]),
                          "checkpoint": str(rca_path)}, "variants": {}}
    for mode in ("failure-only", "root-supervised"):
        seed_everything(seed)
        model = ServiceRiskInteraction(cached["train"][0]["features"].shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(cached["train"], batch_size=64, shuffle=True, generator=generator)
        initial, _, _ = predict(model, loaders["validation"])
        best_ap = initial["average_precision"]
        best_epoch, best_state = 0, copy.deepcopy(model.state_dict())
        history = [{"epoch": 0, "validation_ap": best_ap}]
        for epoch in range(1, args.epochs+1):
            model.train()
            for batch in train_loader:
                out = model(batch["features"], batch["failure_logits"], batch["root_logits"])
                loss = loss_fn(out["failure_logits"], batch["label"])
                valid = (batch["label"] == 1) & (batch["root"].sum(-1) > 0)
                if mode == "root-supervised" and valid.any():
                    roots = batch["root"][valid]
                    roots = roots/roots.sum(-1, keepdim=True)
                    rank = -(roots*out["auxiliary_root_logits"][valid].log_softmax(-1)).sum(-1)
                    loss = loss + 0.1*(rank*batch["weight"][valid]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            metrics, _, _ = predict(model, loaders["validation"])
            history.append({"epoch": epoch, "validation_ap": metrics["average_precision"]})
            if metrics["average_precision"] > best_ap:
                best_ap = metrics["average_precision"]
                best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        metrics, probabilities, residual = predict(model, loaders["test"])
        row["variants"][mode] = {"failure": metrics, "epoch": best_epoch, "validation_ap": best_ap,
                                  "parameters": sum(p.numel() for p in model.parameters()),
                                  "residual": residual, "history": history,
                                  "max_probability_change": float(np.abs(probabilities-base_outputs["test"]["probabilities"]).max())}
        torch.save({"state_dict": best_state, "mode": mode,
                    "input_dim": cached["train"][0]["features"].shape[-1], "bound": 1.0,
                    "rca_checkpoint": str(rca_path), "baseline_checkpoint": str(base_path), "epoch": best_epoch},
                   args.output_dir / f"{name}_fold{fold}_seed{seed}_{mode}.pt")
        print(name, fold, seed, mode, "AP", metrics["average_precision"], "epoch", best_epoch, flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=("ob", "tt"))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=range(5))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,43))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.output_dir.exists():
        parser.error("epochs must be positive and output directory must be new")
    args.output_dir.mkdir(parents=True)
    rows = []
    for name in args.datasets:
        for fold in args.folds:
            for seed in args.seeds:
                rows.append(run(name, fold, seed, args))
                (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
