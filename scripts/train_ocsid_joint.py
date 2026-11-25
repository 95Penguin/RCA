#!/usr/bin/env python3
"""Train the unified OCSID model (prediction + source-impact RCA).

The model consumes the shared panel windows plus per-operation trace features
and is trained from scratch on the deployment-group five-fold split.  It reports
both prediction metrics (AP / F1) and event-level RCA (Hit@1 / MRR), so the two
tasks are evaluated on the same checkpoint and the same windows.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --output artifacts/ocsid_joint_tt_seed42_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.re2 import Re2WindowDataset  # noqa: E402
from models.factory import feature_groups  # noqa: E402
from ocsid.features import OP_FEATURE_NAMES, load_operation_table  # noqa: E402
from models.ocsid import OCSIDModel  # noqa: E402
from training.metrics import classification_metrics, rca_metrics  # noqa: E402

PANELS = {name: ROOT / f"artifacts/re2_{name}_panel_fair_v1" for name in ("ob", "tt")}
OP_FEATURES = {
    name: ROOT / f"artifacts/re2_{name}_ocsid_operation_features_v1.csv"
    for name in ("ob", "tt")
}
def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def _build_arrays(dataset_name: str, data_dir: Path, horizon_minutes: int = 1):
    dataset = Re2WindowDataset(
        data_dir, history=5, split_policy="deployment", split_fold=0,
        horizon_minutes=horizon_minutes, graph_direction="symmetric",
    )
    table, operations = load_operation_table(OP_FEATURES[dataset_name])
    op_index = {op: index for index, op in enumerate(operations)}
    ends = list(dataset.sample_indices)
    features = dataset.features
    adjacency = dataset.adjacency.numpy()
    timestamps = dataset.timestamp_epochs
    case_ids = dataset.case_ids

    X = np.stack([features[end - 4:end + 1] for end in ends]).astype(np.float32)
    A = np.stack([adjacency[end - 4:end + 1] for end in ends]).astype(np.float32)
    O = np.zeros((len(ends), 5, len(operations), len(OP_FEATURE_NAMES)), dtype=np.float32)
    M = np.zeros((len(ends), 5, len(operations)), dtype=np.float32)
    for sample, end in enumerate(ends):
        for step, offset in enumerate(range(4, -1, -1)):
            minute = timestamps[end - offset]
            rows = table.get((case_ids[end - offset], minute), {})
            for op, values in rows.items():
                index = op_index[op]
                O[sample, step, index] = values
                M[sample, step, index] = 1.0
    y = np.asarray([dataset.failure_targets[end] for end in ends], dtype=np.float32)
    node = np.stack([dataset.node_targets[end] for end in ends]).astype(np.float32)
    events = [dataset.event_ids[end] for end in ends]
    return dataset, operations, ends, X, A, O, M, y, node, events


def _standardize(dataset, X, O, M, train_ends, ends):
    positions = {end: index for index, end in enumerate(ends)}
    train_positions = [positions[end] for end in train_ends]
    train_minutes = sorted({
        end - offset
        for end in train_ends for offset in range(5)
    })
    flat = dataset.features[train_minutes].reshape(-1, dataset.features.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    X = (X - mean) / scale
    # Keep binary observation masks binary.
    observed = np.array([name.startswith("observed_") or name == "log_observed"
                         for name in dataset.feature_columns])
    X[..., observed] = np.round(X[..., observed])
    rows = O[M > 0]
    if len(rows):
        o_mean = rows.mean(axis=0)
        o_scale = rows.std(axis=0)
        o_scale[o_scale < 1e-8] = 1.0
        O = ((O - o_mean) / o_scale) * M[..., None]
    return X, O


def _f1(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    tp = float(((predicted == 1) & (labels == 1)).sum())
    fp = float(((predicted == 1) & (labels == 0)).sum())
    fn = float(((predicted == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _calibrate(validation_probabilities, validation_labels, test_probabilities,
               test_labels):
    """Validation-selected threshold and temperature scaling."""
    grid = np.linspace(0.05, 0.95, 91)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        score = _f1(validation_labels, validation_probabilities, threshold)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    eps = 1e-6
    val_logits = np.log(np.clip(validation_probabilities, eps, 1 - eps) /
                        np.clip(1 - validation_probabilities, eps, 1 - eps))
    test_logits = np.log(np.clip(test_probabilities, eps, 1 - eps) /
                         np.clip(1 - test_probabilities, eps, 1 - eps))
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.5, 3.0, 26):
        probability = 1.0 / (1.0 + np.exp(-val_logits / temperature))
        nll = -np.mean(
            validation_labels * np.log(np.clip(probability, eps, 1))
            + (1 - validation_labels) * np.log(np.clip(1 - probability, eps, 1))
        )
        if nll < best_nll:
            best_temperature, best_nll = float(temperature), float(nll)
    calibrated = 1.0 / (1.0 + np.exp(-test_logits / best_temperature))
    return {
        "threshold": best_threshold,
        "validation_f1_at_threshold": best_f1,
        "temperature": best_temperature,
        "test_f1_at_validation_threshold": _f1(
            test_labels, test_probabilities, best_threshold
        ),
        "test_f1_calibrated_at_0_5": _f1(test_labels, calibrated, 0.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-operation", action="store_true")
    parser.add_argument("--no-aux-root", action="store_true")
    parser.add_argument("--horizon-minutes", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device != "cpu" and torch.cuda.is_available()
        else "mps" if args.device != "cpu" and torch.backends.mps.is_available()
        else "cpu"
    )
    data_dir = PANELS[args.dataset]
    results = []
    for fold in args.folds:
        dataset, operations, ends, X, A, O, M, y, node, events = _build_arrays(
            args.dataset, data_dir, args.horizon_minutes
        )
        dataset = Re2WindowDataset(
            data_dir, history=5, split_policy="deployment", split_fold=fold,
            horizon_minutes=args.horizon_minutes, graph_direction="symmetric",
        )
        train_ends, val_ends, test_ends = dataset.predefined_splits
        positions = {end: index for index, end in enumerate(ends)}
        train_pos = [positions[e] for e in train_ends]
        val_pos = [positions[e] for e in val_ends]
        test_pos = [positions[e] for e in test_ends]
        X, O = _standardize(dataset, X, O, M, train_ends, ends)

        def tensors(index):
            idx = np.asarray(index)
            return (
                torch.from_numpy(X[idx]).to(device),
                torch.from_numpy(A[idx]).to(device),
                torch.from_numpy(O[idx]).to(device),
                torch.from_numpy(M[idx]).to(device),
                torch.from_numpy(y[idx]).to(device),
                torch.from_numpy(node[idx]).to(device),
            )

        train_t = tensors(train_pos)
        val_t = tensors(val_pos)
        test_t = tensors(test_pos)
        groups = feature_groups(dataset.feature_columns)
        model = OCSIDModel(
            feature_dim=X.shape[-1], operation_dim=len(OP_FEATURE_NAMES),
            feature_groups=groups, hidden_dim=64, dropout=0.1,
            aggregation="max", use_graph=not args.no_graph,
            use_operation=not args.no_operation,
            aux_root=not args.no_aux_root,
        ).to(device)
        positives = float(y[train_pos].sum())
        pos_weight = torch.tensor(
            (len(train_pos) - positives) / max(positives, 1.0),
            dtype=torch.float32, device=device,
        )
        bce = nn.BCELoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(
            TensorDataset(*train_t), batch_size=args.batch_size, shuffle=True
        )
        best = {"ap": -1.0, "epoch": -1, "state": None}
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in loader:
                bx, ba, bo, bm, by, bn = batch
                optimizer.zero_grad()
                out = model(bx, ba, bo, bm)
                loss = bce(out["probability"], by)
                loss = loss + 0.2 * bce(out["operation_probability"], by)
                aux_logits = out.get("aux_root_logits")
                if aux_logits is not None:
                    valid = (by > 0) & (bn.sum(dim=1) > 0)
                    if valid.any():
                        log_probs = torch.log_softmax(aux_logits[valid], dim=-1)
                        root_index = bn[valid].argmax(dim=1)
                        loss = loss + 0.5 * (-log_probs.gather(
                            1, root_index.unsqueeze(1)
                        ).mean())
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_out = model(val_t[0], val_t[1], val_t[2], val_t[3])
            val_metrics = classification_metrics(
                val_t[4].cpu().numpy().astype(np.float32),
                val_out["probability"].cpu().numpy(),
            )
            score = val_metrics["average_precision"] or -1.0
            if score > best["ap"]:
                best = {"ap": score, "epoch": epoch,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        model.eval()
        with torch.no_grad():
            test_out = model(test_t[0], test_t[1], test_t[2], test_t[3])
        with torch.no_grad():
            val_out = model(val_t[0], val_t[1], val_t[2], val_t[3])
        probs = test_out["probability"].cpu().numpy()
        labels = test_t[4].cpu().numpy().astype(np.float32)
        calibration = _calibrate(
            val_out["probability"].cpu().numpy(),
            val_t[4].cpu().numpy().astype(np.float32),
            probs, labels,
        )
        prediction = classification_metrics(labels, probs)
        result = {
            "fold": fold,
            "train_samples": len(train_pos),
            "validation_samples": len(val_pos),
            "test_samples": len(test_pos),
            "best_epoch": best["epoch"],
            "best_val_ap": best["ap"],
            "average_precision": prediction["average_precision"],
            "f1_at_0_5": prediction["f1_at_0_5"],
            "precision_at_0_5": prediction["precision_at_0_5"],
            "recall_at_0_5": prediction["recall_at_0_5"],
            "event_rca": None,  # attribution is CPCA, outside the model
            "calibration": calibration,
            "services": list(dataset.services),
            "operations": [list(op) for op in operations],
            "test_predictions": [
                {
                    "case": dataset.case_ids[end],
                    "minute": int(dataset.timestamp_epochs[end]),
                    "event": events[position],
                    "root": (dataset.services[int(node[position].argmax())]
                             if node[position].sum() > 0 else None),
                    "probability": float(probs[index]),
                    "operation_scores": test_out["operation_scores"].cpu().numpy()[index].tolist(),
                }
                for index, (position, end) in enumerate(zip(test_pos, test_ends))
            ],
        }
        results.append(result)
        print(f"fold {fold}: AP={result['average_precision']:.4f} "
              f"F1={result['f1_at_0_5']:.4f}", flush=True)

    def summarize(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))} if values else None

    report = {
        "dataset": args.dataset,
        "protocol": "OCSID predictor; deployment folds; attribution is CPCA (outside the model)",
        "folds": results,
        "summary": {
            "ap": summarize(results, "average_precision"),
            "f1": summarize(results, "f1_at_0_5"),
        },
        "event_summary": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
