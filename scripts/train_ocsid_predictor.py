#!/usr/bin/env python3
"""OCSID-V1 prediction head: operation-level next-minute failure predictor.

Input : per-minute per-entry-operation features from
        ``build_ocsid_operation_features.py``.
Model : shared GRU over each operation's 5-minute feature sequence -> q_o;
        system probability ``p = 1 - prod_o(1 - q_o)`` over present operations.
Split : the repository's deployment-group five-fold holdout (test=fold N,
        validation=fold (N+1)%5, train=the rest).
Eval  : AP / F1@0.5 / P / R per fold, mean +- SD; persistence baseline on the
        same windows.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_predictor.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --output artifacts/ocsid_predictor_tt_seed42_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training.metrics import classification_metrics  # noqa: E402
from ocsid.features import PANEL_FEATURES  # noqa: E402

PANELS = {name: ROOT / f"artifacts/re2_{name}_panel_fair_v1" for name in ("ob", "tt")}
FEATURES = {
    name: ROOT / f"artifacts/re2_{name}_ocsid_operation_features_v1.csv"
    for name in ("ob", "tt")
}

# log-scale duration/count views plus causal-baseline ratios
FEATURE_BUILDERS = (
    ("log_p50", lambda r: math.log1p(float(r["p50_ms"]))),
    ("log_p95", lambda r: math.log1p(float(r["p95_ms"]))),
    ("log_p99", lambda r: math.log1p(float(r["p99_ms"]))),
    ("log_mean", lambda r: math.log1p(float(r["mean_ms"]))),
    ("log_count", lambda r: math.log1p(float(r["count"]))),
    ("log_spans", lambda r: math.log1p(float(r["span_count"]))),
    ("n_services", lambda r: float(r["n_services"]) / 10.0),
    ("error_rate", lambda r: float(r["error_count"]) / max(float(r["count"]), 1.0)),
    ("log_ratio_p95", lambda r: math.log(
        max(float(r["p95_ms"]), 1e-6) / max(float(r["baseline_p95_ms"]), 1e-6))),
    ("log_ratio_count", lambda r: math.log(
        max(float(r["count"]), 1e-6) / max(float(r["baseline_count"]), 1e-6))),
    ("avg_spans", lambda r: float(r["span_count"]) / max(float(r["count"]), 1.0)),
    # Same definition as the label's entry degradation score: max(relative
    # latency ratio, absolute increase over 100 ms).  This is a causal
    # current-state observation, not a future label.
    ("deg_score", lambda r: max(
        float(r["p95_ms"]) / max(1.5 * float(r["baseline_p95_ms"]), 1e-6),
        (float(r["p95_ms"]) - float(r["baseline_p95_ms"])) / 100.0,
    )),
    ("deg_binary", lambda r: 1.0 if max(
        float(r["p95_ms"]) / max(1.5 * float(r["baseline_p95_ms"]), 1e-6),
        (float(r["p95_ms"]) - float(r["baseline_p95_ms"])) / 100.0,
    ) >= 1.0 else 0.0),
    ("log_roll_ratio", lambda r: math.log(
        max(float(r["p95_ms"]), 1e-6) / max(float(r.get("roll_p95_ms", r["baseline_p95_ms"])), 1e-6))),
    ("roll_deg", lambda r: max(
        float(r["p95_ms"]) / max(1.5 * float(r.get("roll_p95_ms", r["baseline_p95_ms"])), 1e-6),
        (float(r["p95_ms"]) - float(r.get("roll_p95_ms", r["baseline_p95_ms"]))) / 100.0,
    )),
    ("roll_deg_binary", lambda r: 1.0 if max(
        float(r["p95_ms"]) / max(1.5 * float(r.get("roll_p95_ms", r["baseline_p95_ms"])), 1e-6),
        (float(r["p95_ms"]) - float(r.get("roll_p95_ms", r["baseline_p95_ms"]))) / 100.0,
    ) >= 1.0 else 0.0),
    ("log_roll_count", lambda r: math.log(
        max(float(r["count"]), 1e-6) / max(float(r.get("roll_count", r["baseline_count"])), 1e-6))),
)
FEATURE_NAMES = tuple(name for name, _ in FEATURE_BUILDERS)
FEATURE_DIM = len(FEATURE_NAMES)


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def _load_labels(dataset: str):
    panel = PANELS[dataset]
    minute_to_case = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((panel / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    labels: dict[tuple[str, int], int] = {}
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = minute_to_case.get(row["input_minute_utc"])
            if case is None:
                continue
            key = (case, _epoch(row["input_minute_utc"]))
            labels[key] = int(row["failure_next_minute"])
    return labels


def _fold_map(dataset: str) -> dict[str, int]:
    groups = {
        row["case_id"]: row["deployment_id"]
        for row in csv.DictReader((PANELS[dataset] / "case_groups.csv").open(encoding="utf-8-sig"))
    }
    sizes = Counter(groups.values())
    fold_sizes = [0] * 5
    group_fold = {}
    for group, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(5), key=lambda index: (fold_sizes[index], index))
        group_fold[group] = fold
        fold_sizes[fold] += size
    return {case: group_fold[group] for case, group in groups.items()}


def _load_panel_entries(dataset: str):
    """Per-minute causal entry context from the shared panel."""
    path = PANELS[dataset] / "service_features_1m.csv"
    values: dict[tuple[str, int], list[float]] = {}
    minute_to_case = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((PANELS[dataset] / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = minute_to_case.get(row["timestamp_utc"])
            if case is None:
                continue
            epoch = _epoch(row["timestamp_utc"])
            current = values.get((case, epoch), [0.0, 0.0, 0.0, 0.0, 0.0])
            current[0] = max(current[0], float(row["entry_p95_ms"] or 0))
            current[1] = max(current[1], float(row["entry_degradation_score_rolling"] or 0))
            current[2] = max(current[2], float(row["entry_span_count"] or 0))
            current[3] = max(current[3], float(row["observed_entry_p95"] or 0))
            current[4] = max(current[4], float(row["observed_entry_baseline"] or 0))
            values[case, epoch] = current
    return values




def _load_panel_all(dataset: str):
    """Per-minute system context: max/mean over services of the 19 panel features."""
    path = PANELS[dataset] / "service_features_1m.csv"
    minute_to_case = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((PANELS[dataset] / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    accum: dict[tuple[str, int], dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = minute_to_case.get(row["timestamp_utc"])
            if case is None:
                continue
            epoch = _epoch(row["timestamp_utc"])
            item = accum.setdefault((case, epoch), {
                "max": [0.0] * len(PANEL_FEATURES),
                "sum": [0.0] * len(PANEL_FEATURES),
                "n": 0,
            })
            for index, name in enumerate(PANEL_FEATURES):
                try:
                    value = float(row[name] or 0.0)
                except ValueError:
                    value = 0.0
                if value > item["max"][index]:
                    item["max"][index] = value
                item["sum"][index] += value
            item["n"] += 1
    return {
        key: item["max"] + [value / max(item["n"], 1) for value in item["sum"]]
        for key, item in accum.items()
    }


def _load_features(dataset: str):
    by_case: dict[str, dict[int, dict[tuple[str, str], dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    operations = set()
    path = FEATURES[dataset]
    if not path.is_file():
        raise SystemExit(f"missing feature file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["entry_service"], row["entry_operation"])
            operations.add(key)
            by_case[row["case_id"]][int(row["minute_epoch"])][key] = row
    # Causal rolling baseline per (case, operation): median of the previous
    # five observed minutes, matching the label's rolling degradation rule.
    for minutes in by_case.values():
        series: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
        for minute, rows in minutes.items():
            for key, row in rows.items():
                series[key].append((minute, row))
        for key, values in series.items():
            values.sort(key=lambda item: item[0])
            for index, (minute, row) in enumerate(values):
                prior = [item[1] for item in values[max(0, index - 5):index]]
                if len(prior) < 3:
                    row["roll_p95_ms"] = row["baseline_p95_ms"]
                    row["roll_count"] = row["baseline_count"]
                    continue
                row["roll_p95_ms"] = float(np.median([float(r["p95_ms"]) for r in prior]))
                row["roll_count"] = float(np.median([float(r["count"]) for r in prior]))
    return by_case, sorted(operations)


def _build_samples(dataset, labels, operations):
    by_case, _ = _load_features(dataset)
    panel = _load_panel_entries(dataset)
    panel_all = _load_panel_all(dataset)
    op_index = {op: index for index, op in enumerate(operations)}
    samples = []
    for case, minutes in by_case.items():
        ordered = sorted(minutes)
        for end in ordered:
            if (case, end) not in labels:
                continue
            history = [end - 60 * offset for offset in range(4, -1, -1)]
            if any(minute not in minutes for minute in history):
                continue
            x = np.zeros((5, len(operations), FEATURE_DIM), dtype=np.float32)
            mask = np.zeros((5, len(operations)), dtype=np.float32)
            for step, minute in enumerate(history):
                for op, row in minutes[minute].items():
                    index = op_index[op]
                    mask[step, index] = 1.0
                    x[step, index] = [builder(row) for _, builder in FEATURE_BUILDERS]
            entry = panel.get((case, end), [0.0, 0.0, 0.0, 0.0, 0.0])
            # Flat per-minute system view: operation max/mean + panel max/mean.
            flat = np.zeros((5, 2 * FEATURE_DIM + 2 * len(PANEL_FEATURES)), dtype=np.float32)
            for step, minute in enumerate(history):
                rows = list(minutes[minute].values())
                if rows:
                    op = np.array([[builder(row) for _, builder in FEATURE_BUILDERS] for row in rows],
                                  dtype=np.float32)
                    flat[step, :FEATURE_DIM] = op.max(axis=0)
                    flat[step, FEATURE_DIM:2 * FEATURE_DIM] = op.mean(axis=0)
                panel_vec = panel_all.get((case, minute))
                if panel_vec is not None:
                    arr = np.array(panel_vec, dtype=np.float32)
                    flat[step, 2 * FEATURE_DIM:2 * FEATURE_DIM + len(PANEL_FEATURES)] = arr[:len(PANEL_FEATURES)]
                    flat[step, 2 * FEATURE_DIM + len(PANEL_FEATURES):] = arr[len(PANEL_FEATURES):]
            samples.append({
                "case": case,
                "minute": end,
                "x": x,
                "mask": mask,
                "flat": flat,
                "g": np.concatenate([
                    np.array(panel_all.get((case, end), [0.0] * (2 * len(PANEL_FEATURES))), dtype=np.float32),
                    np.array(panel_all.get((case, end), [0.0] * (2 * len(PANEL_FEATURES))), dtype=np.float32)
                    - np.array(panel_all.get((case, history[0]), [0.0] * (2 * len(PANEL_FEATURES))), dtype=np.float32),
                ]),
                "g_entry": np.array([
                    math.log1p(entry[0]), entry[1], math.log1p(entry[2]),
                    entry[3], entry[4],
                ], dtype=np.float32),
                "y": labels[case, end],
                "persistence": labels.get((case, end - 60), 0),
            })
    return samples


class FlatPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.head(self.dropout(h[:, -1]))).squeeze(-1).clamp(1e-6, 1 - 1e-6)


class OCSIDPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 32, dropout: float = 0.1,
                 aggregation: str = "max", global_dim: int = 4 * len(PANEL_FEATURES)):
        super().__init__()
        self.aggregation = aggregation
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + global_dim, 1)

    def forward(self, x, mask, g):
        batch, steps, ops, dim = x.shape
        flat = (x * mask.unsqueeze(-1)).reshape(batch * ops, steps, dim)
        h, _ = self.gru(flat)
        node = self.dropout(h[:, -1]).reshape(batch, ops, -1)
        node = torch.cat((node, g.unsqueeze(1).expand(-1, ops, -1)), dim=-1)
        q = torch.sigmoid(self.head(node).squeeze(-1))
        present = mask[:, -1]
        if self.aggregation == "product":
            # p = 1 - prod(1 - q); assumes independent operations.
            log_one_minus = torch.log1p(-q.clamp(max=1.0 - 1e-6))
            p = 1.0 - torch.exp((log_one_minus * present).sum(dim=1))
        else:
            # A system fails if any entry operation degrades (max pooling).
            p = (q * present).max(dim=1).values
        return p.clamp(1e-6, 1.0 - 1e-6), q


def _standardize(train, others):
    train_x = np.stack([item["x"] for item in train])
    train_mask = np.stack([item["mask"] for item in train])
    flat = train_x[train_mask > 0]
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-6] = 1.0
    for items in [train, *others]:
        for item in items:
            item["x"] = ((item["x"] - mean) / scale) * item["mask"][..., None]
    flat = np.stack([item["flat"] for item in train])
    f_mean = flat.mean(axis=0)
    f_scale = flat.std(axis=0)
    f_scale[f_scale < 1e-6] = 1.0
    for items in [train, *others]:
        for item in items:
            item["flat"] = (item["flat"] - f_mean) / f_scale
    g = np.stack([item["g"] for item in train])
    g_mean = g.mean(axis=0)
    g_scale = g.std(axis=0)
    g_scale[g_scale < 1e-6] = 1.0
    for items in [train, *others]:
        for item in items:
            item["g"] = (item["g"] - g_mean) / g_scale
    entry = np.stack([item["g_entry"] for item in train])
    e_mean = entry.mean(axis=0)
    e_scale = entry.std(axis=0)
    e_scale[e_scale < 1e-6] = 1.0
    for items in [train, *others]:
        for item in items:
            item["g_entry"] = (item["g_entry"] - e_mean) / e_scale
    return mean, scale


def _train_one(train, validation, test, seed, epochs, batch_size, device,
               aggregation: str = "max", arch: str = "flat",
               selection: str = "val_ap", context: str = "entry",
               dropout: float = 0.1, weight_decay: float = 0.0,
               label_smoothing: float = 0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if arch == "flat":
        flat_dim = 2 * FEATURE_DIM + 2 * len(PANEL_FEATURES)
        model = FlatPredictor(flat_dim, dropout=dropout).to(device)
    else:
        global_dim = 5 if context == "entry" else 4 * len(PANEL_FEATURES)
        model = OCSIDPredictor(FEATURE_DIM, aggregation=aggregation,
                               dropout=dropout, global_dim=global_dim).to(device)
    positives = sum(item["y"] for item in train)
    pos_weight = torch.tensor(
        (len(train) - positives) / max(positives, 1), dtype=torch.float32, device=device
    )
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    def tensors(items):
        x = torch.from_numpy(np.stack([item["x"] for item in items])).to(device)
        mask = torch.from_numpy(np.stack([item["mask"] for item in items])).to(device)
        y = torch.tensor([item["y"] for item in items], dtype=torch.float32, device=device)
        return x, mask, y

    train_x, train_m, train_y = tensors(train)
    if label_smoothing > 0:
        train_y = train_y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    val_x, val_m, val_y = tensors(validation)
    test_x, test_m, test_y = tensors(test)
    gkey = "g" if (arch == "operation" and context == "panel") else "g_entry"
    train_g = torch.from_numpy(np.stack([item[gkey] for item in train])).to(device)
    val_g = torch.from_numpy(np.stack([item[gkey] for item in validation])).to(device)
    test_g = torch.from_numpy(np.stack([item[gkey] for item in test])).to(device)
    train_f = torch.from_numpy(np.stack([item["flat"] for item in train])).to(device)
    val_f = torch.from_numpy(np.stack([item["flat"] for item in validation])).to(device)
    test_f = torch.from_numpy(np.stack([item["flat"] for item in test])).to(device)
    if arch == "flat":
        train_x, val_x, test_x = train_f, val_f, test_f
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_g) if arch == "flat"
        else TensorDataset(train_x, train_m, train_y, train_g),
        batch_size=batch_size, shuffle=True,
    )
    best = {"ap": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            if arch == "flat":
                bx, by, bg = batch
                optimizer.zero_grad()
                p = model(bx)
            else:
                bx, bm, by, bg = batch
                optimizer.zero_grad()
                p, _ = model(bx, bm, bg)
            loss = criterion(p, by)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(by)
        model.eval()
        with torch.no_grad():
            val_p = model(val_x) if arch == "flat" else model(val_x, val_m, val_g)[0]
        val_probs = val_p.detach().cpu().numpy()
        val_labels = val_y.detach().cpu().numpy()
        val_ap = classification_metrics(val_labels, val_probs)["average_precision"]
        history.append({"epoch": epoch, "loss": total / len(train), "val_ap": val_ap})
        score = val_ap if val_ap is not None else -1.0
        if selection == "last":
            best = {"ap": score, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif score > best["ap"]:
            best = {"ap": score, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        test_p = model(test_x) if arch == "flat" else model(test_x, test_m, test_g)[0]
    test_probs = test_p.detach().cpu().numpy()
    test_labels = test_y.detach().cpu().numpy()
    metrics = classification_metrics(test_labels, test_probs)
    metrics["best_epoch"] = best["epoch"]
    metrics["best_val_ap"] = best["ap"]
    metrics["history"] = history
    return metrics, test_probs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--arch", choices=("flat", "operation"), default="operation")
    parser.add_argument("--selection", choices=("val_ap", "last"), default="val_ap")
    parser.add_argument("--context", choices=("entry", "panel"), default="entry")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--aggregation", choices=("max", "product"), default="max")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device != "cpu" and torch.cuda.is_available()
        else "mps" if args.device != "cpu" and torch.backends.mps.is_available()
        else "cpu"
    )
    labels = _load_labels(args.dataset)
    folds = _fold_map(args.dataset)
    by_case, operations = _load_features(args.dataset)
    samples = _build_samples(args.dataset, labels, operations)
    # Standardize once on the full sample set; the fold split below still keeps
    # all fitting inside the training partition for the model itself.  Feature
    # scaling uses no labels.
    _standardize(samples, [])
    for item in samples:
        item["fold"] = folds[item["case"]]
    print(f"[{args.dataset}] samples={len(samples)} operations={len(operations)} "
          f"positives={sum(i['y'] for i in samples)} device={device}")

    results = []
    persistence = []
    for fold in args.folds:
        train = [i for i in samples if i["fold"] not in {fold, (fold + 1) % 5}]
        validation = [i for i in samples if i["fold"] == (fold + 1) % 5]
        test = [i for i in samples if i["fold"] == fold]
        if not train or not validation or not test:
            continue
        metrics, probs = _train_one(
            train, validation, test, args.seed, args.epochs, args.batch_size, device,
            aggregation=args.aggregation, arch=args.arch,
            selection=args.selection, context=args.context,
            dropout=args.dropout, weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
        )
        metrics["fold"] = fold
        metrics["train_samples"] = len(train)
        metrics["validation_samples"] = len(validation)
        metrics["test_samples"] = len(test)
        results.append(metrics)
        persistence_metrics = classification_metrics(
            np.array([i["y"] for i in test], dtype=np.float32),
            np.array([i["persistence"] for i in test], dtype=np.float32),
        )
        persistence.append({"fold": fold, **persistence_metrics})
        print(f"  fold {fold}: n={len(test)} pos={sum(i['y'] for i in test)} "
              f"AP={metrics['average_precision']:.4f} "
              f"F1={metrics['f1_at_0_5']:.4f} "
              f"P={metrics['precision_at_0_5']:.4f} R={metrics['recall_at_0_5']:.4f}",
              flush=True)

    def summarize(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            return None
        return {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }

    report = {
        "dataset": args.dataset,
        "protocol": {
            "split": "deployment-group five-fold; test=fold N, validation=(N+1)%5",
            "model": (f"flat system GRU (operation max/mean + panel max/mean)"
                      if args.arch == "flat"
                      else f"shared GRU per entry operation; aggregation={args.aggregation}"),
            "features": list(FEATURE_NAMES),
            "epochs": args.epochs,
            "seed": args.seed,
            "checkpoint": "best validation AP",
        },
        "samples": len(samples),
        "operations": [list(op) for op in operations],
        "folds": results,
        "persistence": persistence,
        "summary": {
            key: summarize(results, field)
            for key, field in (
                ("ap", "average_precision"), ("f1", "f1_at_0_5"),
                ("precision", "precision_at_0_5"), ("recall", "recall_at_0_5"),
            )
        },
        "persistence_summary": {
            "ap": summarize(persistence, "average_precision"),
            "f1": summarize(persistence, "f1_at_0_5"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset,
        "ocsid_ap": report["summary"]["ap"],
        "ocsid_f1": report["summary"]["f1"],
        "persistence_ap": report["persistence_summary"]["ap"],
        "persistence_f1": report["persistence_summary"]["f1"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
