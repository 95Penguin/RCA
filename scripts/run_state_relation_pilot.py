#!/usr/bin/env python3
"""Frozen-baseline, capacity-matched metric state/relation RCA pilot."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.re2 import Re2WindowDataset
from models.factory import build_model
from training.engine import seed_everything, choose_device, fit_feature_scaler
from training.experiment import _event_rca_metrics, _configure_root_event_weights
from training.metrics import classification_metrics, rca_metrics
from training.normal_reference import configure_metric_reference


def validation_rank_loss(dataset, ends, output):
    valid = (output["labels"] == 1) & (output["node_targets"].sum(-1) > 0)
    if not valid.any():
        raise ValueError("validation requires attributed failure windows")
    roots = output["node_targets"][valid]
    roots = roots / roots.sum(-1, keepdims=True)
    log_probs = torch.from_numpy(output["node_logits"][valid]).log_softmax(-1).numpy()
    weights = dataset.root_event_weights[ends][valid]
    return float((-(roots * log_probs).sum(-1) * weights).mean())


def first_available_rca(dataset, ends, output):
    first = {}
    for row, end in enumerate(ends):
        if dataset.event_ids[end] and output["node_targets"][row].sum() > 0:
            first.setdefault(dataset.event_ids[end], row)
    rows = list(first.values())
    result = rca_metrics(output["node_targets"][rows], output["node_scores"][rows])
    result["definition"] = "first available attributed window per event; not necessarily pre-onset"
    return result


def evaluate(model, loader, device):
    output = {key: [] for key in ("labels", "probabilities", "node_targets", "node_scores", "node_logits", "failure_logits")}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if "baseline_node_logits" in batch:
                prediction = model.forward_evidence(batch["node_features"].to(device), batch["adjacency"].to(device), batch["baseline_node_logits"].to(device))
                prediction["failure_logits"] = batch["baseline_failure_logits"].to(device)
            else:
                prediction = model(batch["node_features"].to(device), batch["adjacency"].to(device))
            output["labels"].append(batch["failure_target"].numpy())
            output["node_targets"].append(batch["node_targets"].numpy())
            output["probabilities"].append(prediction["failure_logits"].softmax(-1)[:, 1].cpu().numpy())
            output["node_scores"].append(prediction["node_scores"].cpu().numpy())
            output["node_logits"].append(prediction["node_logits"].cpu().numpy())
            output["failure_logits"].append(prediction["failure_logits"].cpu().numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def run(args, name, fold):
    device = choose_device(args.device)
    dataset = Re2WindowDataset(ROOT / f"artifacts/re2_{name}_panel_fair_v1", history=5,
                               split_policy="deployment", split_fold=fold)
    train, val, test = dataset.predefined_splits
    checkpoint_path = ROOT / f"artifacts/re2_{name}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean, scale = fit_feature_scaler(dataset, train)
    np.testing.assert_allclose(mean, checkpoint["feature_mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scale, checkpoint["feature_scale"], rtol=1e-5, atol=1e-6)
    _configure_root_event_weights(dataset, (train, val, test), True)
    indices = {end: i for i, end in enumerate(dataset.sample_indices)}
    def loader(ends):
        return DataLoader(Subset(dataset, [indices[end] for end in ends]), batch_size=64, shuffle=False)
    train_loader, val_loader, test_loader = (loader(ends) for ends in (train, val, test))
    baseline = build_model("ltfp", feature_columns=dataset.feature_columns, num_services=len(dataset.services)).model.to(device)
    baseline.load_state_dict(checkpoint["model_state_dict"])
    base_output = evaluate(baseline, test_loader, device)
    original_test_loader = test_loader
    cached_loaders = []
    for split_loader in (train_loader, val_loader, test_loader):
        frozen_output = evaluate(baseline, split_loader, device)
        samples = []
        for i in range(len(split_loader.dataset)):
            sample = dict(split_loader.dataset[i])
            sample["baseline_node_logits"] = torch.from_numpy(frozen_output["node_logits"][i])
            sample["baseline_failure_logits"] = torch.from_numpy(frozen_output["failure_logits"][i])
            samples.append(sample)
        cached_loaders.append(DataLoader(samples, batch_size=64, shuffle=False))
    train_loader, val_loader, test_loader = cached_loaders
    row = {"dataset": name, "fold": fold, "checkpoint": str(checkpoint_path),
           "graph_control": args.graph_control, "graph_seed": args.graph_seed,
           "training_seed": args.seed, "epochs": args.epochs,
           "root_selection": args.root_selection,
           "baseline": {"failure": classification_metrics(base_output["labels"], base_output["probabilities"]),
                        "rca": _event_rca_metrics(dataset, test, base_output),
                        "first_available_rca": first_available_rca(dataset, test, base_output)}, "variants": {}}
    for variant in args.variants:
        seed_everything(args.seed)
        model = build_model(f"ltfp-{variant}-evidence", feature_columns=dataset.feature_columns,
                            num_services=len(dataset.services), history=5).model.to(device)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(incompatible.unexpected_keys)
        if args.graph_control == "permuted":
            permutation = np.random.default_rng(args.graph_seed).permutation(len(dataset.services))
            if np.array_equal(permutation, np.arange(len(dataset.services))):
                raise ValueError("graph permutation unexpectedly equals identity")
            model.relation_permutation.copy_(torch.as_tensor(permutation, device=device))
        reference = configure_metric_reference(model, dataset, train)
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(key.startswith(model.root_sidecar_parameter_prefixes))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        initial = evaluate(model, test_loader, device)
        np.testing.assert_allclose(initial["node_scores"], base_output["node_scores"], atol=1e-6)
        generator = torch.Generator().manual_seed(args.seed)
        training_loader = DataLoader(train_loader.dataset, batch_size=64, shuffle=True, generator=generator)
        best_mrr = -1.0
        best_loss = float("inf")
        best_state = None
        history = []
        if args.root_selection == "validation-rank-loss":
            initial_val = evaluate(model, val_loader, device)
            best_loss = validation_rank_loss(dataset, val, initial_val)
            best_mrr = _event_rca_metrics(dataset, val, initial_val)["mrr"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = 0
            history.append({"epoch": 0, "loss": None, "val_mrr": best_mrr, "val_rank_loss": best_loss})
        for epoch in range(1, args.epochs + 1):
            # Keep pretrained dropout and batchnorm in evaluation mode.
            model.eval()
            if hasattr(model, "metric_encoder"):
                model.metric_encoder.train()
            losses = []
            for batch in training_loader:
                valid = (batch["failure_target"] == 1) & (batch["node_targets"].sum(-1) > 0)
                if not valid.any():
                    continue
                prediction = model.forward_evidence(batch["node_features"].to(device), batch["adjacency"].to(device), batch["baseline_node_logits"].to(device))
                roots = batch["node_targets"][valid].to(device)
                roots = roots / roots.sum(-1, keepdim=True)
                weights = batch["root_event_weight"][valid].to(device)
                loss = (-(roots * prediction["node_logits"][valid.to(device)].log_softmax(-1)).sum(-1) * weights).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            output = evaluate(model, val_loader, device)
            val_mrr = _event_rca_metrics(dataset, val, output)["mrr"]
            val_loss = validation_rank_loss(dataset, val, output)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mrr": val_mrr, "val_rank_loss": val_loss})
            improved = (val_loss < best_loss if args.root_selection == "validation-rank-loss"
                        else val_mrr is not None and val_mrr > best_mrr)
            if improved:
                best_loss = val_loss
                best_mrr = val_mrr
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        if best_state is None:
            raise ValueError("validation fold has no attributed events")
        model.load_state_dict(best_state)
        output = evaluate(model, test_loader, device)
        full_output = evaluate(model, original_test_loader, device)
        np.testing.assert_allclose(output["node_scores"], full_output["node_scores"], atol=1e-6)
        ap_delta = np.max(np.abs(full_output["probabilities"] - base_output["probabilities"]))
        if ap_delta > 1e-6:
            raise AssertionError(f"protected prediction changed: {ap_delta}")
        result = {"reference": reference, "best_epoch": best_epoch, "validation_mrr": best_mrr,
                  "validation_rank_loss": best_loss,
                  "prediction_max_abs_delta": float(ap_delta),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "rca": _event_rca_metrics(dataset, test, output),
                  "window_rca": rca_metrics(output["node_targets"], output["node_scores"]), "history": history}
        result["first_available_rca"] = first_available_rca(dataset, test, output)
        row["variants"][variant] = result
        torch.save({"model_state_dict": best_state, "feature_columns": dataset.feature_columns,
                    "history": 5, "variant": variant, "seed": args.seed,
                    "baseline_checkpoint": str(checkpoint_path)}, args.output_dir / f"{name}_fold{fold}_{variant}.pt")
        print(name, fold, variant, "epoch", best_epoch, "MRR", result["rca"]["mrr"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=("ob", "tt"))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=range(5))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-selection", choices=("validation-event-mrr", "validation-rank-loss"), default="validation-event-mrr")
    parser.add_argument("--variants", nargs="+", choices=("local-normal", "state-relation", "raw-metric-bypass", "metric-deviation-bypass"), default=("local-normal", "state-relation"))
    parser.add_argument("--graph-control", choices=("real", "permuted"), default="real")
    parser.add_argument("--graph-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if (args.output_dir / "results.json").exists():
        parser.error("results.json already exists; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        for fold in args.folds:
            rows.append(run(args, name, fold))
            (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
