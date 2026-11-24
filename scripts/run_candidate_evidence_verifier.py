#!/usr/bin/env python3
"""Screen a constrained Top-K evidence verifier on frozen formal LTFP folds.

This is phase A of Forecast-then-Verify, not the final raw-evidence method.  It
uses only causal five-minute panel evidence to decide whether candidate
verification is worth the much more expensive request-trace/raw-log adapter.

For every outer deployment fold:
  * replay the frozen formal LTFP checkpoint on train/validation/test;
  * form Top-K candidates from its root logits;
  * fit a shared candidate verifier on positive TRAIN windows only;
  * select the bounded residual magnitude using validation event MRR;
  * evaluate the untouched test fold once.

The LTFP failure path is never modified, so failure AP is exactly preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "artifacts/baselines/.dgl"))

from baselines.adapters.data import PreparedData
from baselines.adapters.frameworks import build_framework
from training.metrics import rca_metrics


PANELS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}
FORMAL = ROOT / "artifacts/baselines/comparison_2026-11-19_seed42/frameworks"
METRICS = ("metric_cpu", "metric_mem", "metric_diskio", "metric_socket")
OBSERVED = ("observed_cpu", "observed_mem", "observed_diskio", "observed_socket")
ACTIVITY = (
    "trace_span_count", "trace_error_count", "trace_duration_mean_ms",
    "log_count", "log_error_count", "entry_span_count", "entry_p95_ms",
    "entry_degradation_score_rolling",
)


def _checkpoint(dataset: str, fold: int) -> Path:
    return FORMAL / f"{dataset}_ltfp_fold{fold}_seed42/best_model.pt"


def _prediction_file(dataset: str, fold: int) -> Path:
    return FORMAL / f"{dataset}_ltfp_fold{fold}_seed42/predictions.jsonl"


def _normal_reference(data: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    d = data.dataset
    metric = [d.feature_columns.index(name) for name in METRICS]
    observed = [d.feature_columns.index(name) for name in OBSERVED]
    # The complete history and next-minute target must be normal.  Only outer
    # training endpoints can contribute reference minutes.
    healthy_minutes: set[int] = set()
    for end in data.train:
        start = end - d.history + 1
        if d.failure_targets[end] != 0:
            continue
        if d.failure_targets[start : end + 1].any():
            continue
        healthy_minutes.update(range(start, end + 1))
    indices = sorted(healthy_minutes)
    values = d.features[indices][:, :, metric]
    masks = d.features[indices][:, :, observed] > 0
    center = np.zeros((len(d.services), len(metric)), np.float32)
    scale = np.ones_like(center)
    for service in range(len(d.services)):
        for feature in range(len(metric)):
            selected = values[:, service, feature][masks[:, service, feature]]
            if not len(selected):
                continue
            median = float(np.median(selected))
            mad = float(np.median(np.abs(selected - median)) * 1.4826)
            std = float(selected.std())
            center[service, feature] = median
            scale[service, feature] = mad if mad > 1e-3 else std if std > 1e-3 else 1.0
    return center, scale


def _window_evidence(
    data: PreparedData,
    end: int,
    reference: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return service evidence [nodes, evidence features] from history only."""
    d = data.dataset
    window = d.features[end - d.history + 1 : end + 1]
    metric = [d.feature_columns.index(name) for name in METRICS]
    observed = [d.feature_columns.index(name) for name in OBSERVED]
    values = window[:, :, metric]
    masks = window[:, :, observed] > 0
    center, scale = reference
    signed = np.clip((values - center[None]) / scale[None], -10.0, 10.0)
    absolute = np.where(masks, np.abs(signed), 0.0)
    signed = np.where(masks, signed, 0.0)
    count = np.maximum(masks.sum(axis=0), 1)
    deviation = np.concatenate(
        [absolute[-1], absolute.max(axis=0), absolute.sum(axis=0) / count], axis=1
    )
    dynamics = np.concatenate([signed[-1], signed[-1] - signed[0]], axis=1)
    coverage = masks.mean(axis=0)

    activity_columns = [d.feature_columns.index(name) for name in ACTIVITY]
    activity = np.log1p(np.maximum(window[:, :, activity_columns], 0.0))
    summaries = np.concatenate(
        [activity[-1], activity.max(axis=0), activity.mean(axis=0), activity[-1] - activity[0]],
        axis=1,
    )
    # Candidate-versus-system contrasts help distinguish a local source from a
    # system-wide symptom without doing another graph pass.
    latest_deviation = absolute[-1].mean(axis=1, keepdims=True)
    relative = np.concatenate(
        [latest_deviation - latest_deviation.mean(axis=0, keepdims=True),
         latest_deviation - latest_deviation.max(axis=0, keepdims=True)], axis=1
    )
    return np.concatenate([deviation, dynamics, coverage, summaries, relative], axis=1).astype(np.float32)


@torch.no_grad()
def _scout_scores(model: nn.Module, data: PreparedData, ends: list[int]) -> tuple[np.ndarray, np.ndarray]:
    risks, roots = [], []
    model.eval()
    for batch in DataLoader(data.windows(ends), batch_size=64, shuffle=False):
        output = model(batch["x"], batch["graph"])
        risks.append(output["failure_logit"].sigmoid().cpu().numpy())
        roots.append(output["root_logits"].cpu().numpy())
    return np.concatenate(risks), np.concatenate(roots)


def _assert_formal_replay(dataset: str, fold: int, data: PreparedData, scores: np.ndarray) -> None:
    expected = {}
    with _prediction_file(dataset, fold).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            expected[row["input_minute"]] = np.asarray(row["root_scores"], np.float32)
    replayed = np.stack([expected[data.dataset.timestamps[end]] for end in data.test])
    if not np.allclose(scores, replayed, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{dataset} fold {fold}: checkpoint replay differs from formal predictions")


def _candidate_rows(
    data: PreparedData,
    ends: list[int],
    root_scores: np.ndarray,
    evidence_by_end: dict[int, np.ndarray],
    top_k: int,
) -> dict[str, object]:
    d = data.dataset
    rows = []
    for row, end in enumerate(ends):
        target = d.node_targets[end]
        if d.failure_targets[end] != 1 or target.sum() <= 0:
            continue
        base = root_scores[row]
        order = np.argsort(-base, kind="stable")
        candidates = order[:top_k]
        true_service = int(target.argmax())
        positions = np.flatnonzero(candidates == true_service)
        target_class = int(positions[0]) if len(positions) else top_k
        # Keep the verifier evidence-only and permutation equivariant.  In
        # particular, do not expose Scout scores or rank-position features
        # during verifier training: outer-train Scout predictions are
        # in-sample and otherwise make "candidate 0" an almost perfect
        # shortcut.  Scout scores enter only through the bounded residual at
        # validation/test time.
        features = evidence_by_end[end][candidates]
        rows.append(
            {
                "end": end,
                "case_id": d.case_ids[end],
                "event_id": d.event_ids[end],
                "candidates": candidates,
                "target_class": target_class,
                "true_service": true_service,
                "base_scores": base,
                "features": features,
            }
        )
    if not rows:
        raise ValueError("split has no attributable positive windows")
    return {"rows": rows, "feature_dim": rows[0]["features"].shape[1]}


def _fit_standardizer(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([row["features"] for row in rows], axis=0)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


class CandidateVerifier(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU())
        self.delta = nn.Linear(hidden, 1)
        self.none = nn.Sequential(
            nn.Linear(feature_dim * 2 + top_k, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.none[-1].weight)
        nn.init.constant_(self.none[-1].bias, -4.0)

    def forward(self, features: torch.Tensor, centered_base: torch.Tensor, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.tanh(self.delta(self.encoder(features)).squeeze(-1))
        pooled = torch.cat(
            [features.mean(dim=1), features.max(dim=1).values, centered_base], dim=1
        )
        none_logit = self.none(pooled).squeeze(-1)
        gate = 1.0 - torch.sigmoid(none_logit)
        corrected = centered_base + rho * gate[:, None] * delta
        return torch.cat([corrected, none_logit[:, None]], dim=1), delta


def _batch(rows, mean, scale, device="cpu"):
    features = torch.tensor(
        np.stack([(row["features"] - mean) / scale for row in rows]),
        dtype=torch.float32, device=device,
    )
    base = torch.tensor(
        np.stack([row["base_scores"][row["candidates"]] for row in rows]),
        dtype=torch.float32, device=device,
    )
    base = base - base[:, :1]
    targets = torch.tensor([row["target_class"] for row in rows], dtype=torch.long, device=device)
    return features, base, targets


def _event_weights(rows: list[dict[str, object]]) -> np.ndarray:
    counts = Counter(row["event_id"] or row["case_id"] for row in rows)
    values = np.asarray(
        [1.0 / counts[row["event_id"] or row["case_id"]] for row in rows], np.float32
    )
    return values / values.mean()


def _evidence_objective(
    model: CandidateVerifier,
    features: torch.Tensor,
    centered_base: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Learn evidence quality without letting in-sample Scout ranks leak in."""
    logits, delta = model(features, centered_base, rho=1.0)
    inside = targets < top_k
    if not bool(inside.any()):
        raise ValueError("training split has no root cause inside the candidate set")
    ranking = torch.nn.functional.cross_entropy(
        delta[inside], targets[inside], reduction="none"
    )
    ranking_loss = (ranking * weights[inside]).sum() / weights[inside].sum()

    # None-of-the-above is a separate calibrated decision.  Its class weight
    # is square-root bounded because candidate misses are very sparse.
    none_target = (targets == top_k).float()
    positives = float(none_target.sum())
    negatives = float(len(none_target) - positives)
    pos_weight = math.sqrt(negatives / positives) if positives else 1.0
    none = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, -1], none_target,
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32),
        reduction="none",
    )
    none_loss = (none * weights).sum() / weights.sum()
    return ranking_loss + 0.25 * none_loss + 0.01 * delta.square().mean()


def _corrected_scores(
    model: CandidateVerifier,
    rows: list[dict[str, object]],
    mean: np.ndarray,
    scale: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    features, base, _ = _batch(rows, mean, scale)
    model.eval()
    with torch.no_grad():
        logits, _ = model(features, base, rho)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    corrected = []
    for index, row in enumerate(rows):
        scores = row["base_scores"].copy()
        candidates = row["candidates"]
        # If none-of-the-above dominates, preserve the scout instead of forcing
        # an unsupported reorder. Otherwise use the bounded candidate residual.
        if int(probabilities[index].argmax()) != model.top_k:
            scores[candidates] = row["base_scores"][candidates] + (
                logits[index, : model.top_k].cpu().numpy() - base[index].cpu().numpy()
            )
        corrected.append(scores)
    return np.stack(corrected), probabilities


def _event_metrics(rows: list[dict[str, object]], scores: np.ndarray, services: int) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["event_id"] or row["case_id"]].append(index)
    targets, event_scores = [], []
    for members in grouped.values():
        target = np.zeros(services, np.float32)
        target[rows[members[0]]["true_service"]] = 1.0
        targets.append(target)
        event_scores.append(scores[members].mean(axis=0))
    result = rca_metrics(np.stack(targets), np.stack(event_scores))
    result["positive_events"] = result.pop("positive_windows")
    return result


def _metrics(rows: list[dict[str, object]], scores: np.ndarray, services: int) -> dict[str, object]:
    targets = np.zeros((len(rows), services), np.float32)
    for index, row in enumerate(rows):
        targets[index, row["true_service"]] = 1.0
    return {
        "window": rca_metrics(targets, scores),
        "event": _event_metrics(rows, scores, services),
    }


def _train_verifier(
    train_rows, validation_rows, mean, scale, top_k, hidden, epochs, seed
) -> tuple[CandidateVerifier, float, dict[str, object]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CandidateVerifier(train_rows[0]["features"].shape[1], hidden, top_k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    features, base, targets = _batch(train_rows, mean, scale)
    weights = torch.tensor(_event_weights(train_rows), dtype=torch.float32)
    best_state = None
    best = (float("inf"), 0)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = _evidence_objective(model, features, base, targets, weights, top_k)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_features, val_base, val_targets = _batch(validation_rows, mean, scale)
        model.eval()
        with torch.no_grad():
            val_weights = torch.tensor(_event_weights(validation_rows), dtype=torch.float32)
            val_loss = float(_evidence_objective(
                model, val_features, val_base, val_targets, val_weights, top_k
            ))
        history.append({"epoch": epoch + 1, "train_loss": float(loss), "validation_loss": val_loss})
        if val_loss < best[0] - 1e-8:
            best = (val_loss, epoch + 1)
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Select only the residual bound on validation event MRR.  Prefer smaller
    # corrections on ties, including the exact baseline rho=0.
    rho_grid = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    choices = []
    services = len(validation_rows[0]["base_scores"])
    for rho in rho_grid:
        scores, probabilities = _corrected_scores(model, validation_rows, mean, scale, rho)
        event = _event_metrics(validation_rows, scores, services)
        choices.append({"rho": rho, "event_mrr": event["mrr"],
                        "none_rate": float((probabilities.argmax(axis=1) == top_k).mean())})
    selected = max(choices, key=lambda row: (row["event_mrr"], -row["rho"]))["rho"]
    return model, float(selected), {
        "selected_epoch": best[1], "rho_choices": choices,
        "objective": "shared evidence-only candidate ranking plus none calibration",
        "target_class_counts": {
            str(key): Counter(int(row["target_class"]) for row in train_rows).get(key, 0)
            for key in range(top_k + 1)
        },
        "history": history,
    }


def _run_fold(dataset_name: str, fold: int, args) -> dict[str, object]:
    data = PreparedData(PANELS[dataset_name], fold, history=5)
    checkpoint = torch.load(_checkpoint(dataset_name, fold), map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    scout = build_framework(
        "ltfp", data, hidden=config["hidden"], dropout=config["dropout"], top_k=config["top_k"]
    )
    scout.load_state_dict(checkpoint["model_state_dict"])
    split_scores = {}
    for name, ends in data.splits.items():
        risk, roots = _scout_scores(scout, data, ends)
        split_scores[name] = {"risk": risk, "roots": roots}
    _assert_formal_replay(dataset_name, fold, data, split_scores["test"]["roots"])

    reference = _normal_reference(data)
    needed = sorted(set(data.train + data.validation + data.test))
    evidence = {end: _window_evidence(data, end, reference) for end in needed}
    row_sets = {
        name: _candidate_rows(data, ends, split_scores[name]["roots"], evidence, args.top_k)["rows"]
        for name, ends in data.splits.items()
    }
    mean, scale = _fit_standardizer(row_sets["train"])
    verifier, rho, training = _train_verifier(
        row_sets["train"], row_sets["validation"], mean, scale,
        args.top_k, args.hidden, args.epochs, args.seed,
    )
    base_scores = np.stack([row["base_scores"] for row in row_sets["test"]])
    corrected, probabilities = _corrected_scores(
        verifier, row_sets["test"], mean, scale, rho
    )
    services = len(data.dataset.services)
    baseline = _metrics(row_sets["test"], base_scores, services)
    verified = _metrics(row_sets["test"], corrected, services)
    base_top = base_scores.argmax(axis=1)
    verified_top = corrected.argmax(axis=1)
    truth = np.asarray([row["true_service"] for row in row_sets["test"]])
    candidate_coverage = np.mean(
        [row["true_service"] in row["candidates"] for row in row_sets["test"]]
    )
    return {
        "dataset": dataset_name,
        "fold": fold,
        "positive_test_windows": len(row_sets["test"]),
        "candidate_top_k": args.top_k,
        "candidate_coverage": float(candidate_coverage),
        "selected_rho": rho,
        "baseline": baseline,
        "verified": verified,
        "baseline_correct_to_verified_wrong": int(np.logical_and(base_top == truth, verified_top != truth).sum()),
        "baseline_wrong_to_verified_correct": int(np.logical_and(base_top != truth, verified_top == truth).sum()),
        "none_rate": float((probabilities.argmax(axis=1) == args.top_k).mean()),
        "training": training,
    }


def _summarize(dataset_name: str, folds: list[dict[str, object]]) -> dict[str, object]:
    summary = {"dataset": dataset_name, "folds": len(folds)}
    for level in ("window", "event"):
        summary[level] = {}
        for method in ("baseline", "verified"):
            summary[level][method] = {}
            for metric in ("hit_at_1", "hit_at_3", "mrr"):
                values = [fold[method][level][metric] for fold in folds]
                summary[level][method][metric] = {
                    "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))
                }
    summary["candidate_coverage"] = float(np.mean([fold["candidate_coverage"] for fold in folds]))
    summary["wrong_to_correct"] = sum(fold["baseline_wrong_to_verified_correct"] for fold in folds)
    summary["correct_to_wrong"] = sum(fold["baseline_correct_to_verified_wrong"] for fold in folds)
    summary["selected_rhos"] = [fold["selected_rho"] for fold in folds]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=["ob", "tt"])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/candidate_evidence_verifier_phase_a.json")
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("top-k must be at least 2")
    result = {
        "experiment": "forecast_then_verify_phase_a_panel_evidence",
        "boundary": "screening only; formal raw Metric/Log/Trace verifier requires a positive result",
        "failure_path": "frozen formal LTFP; AP exactly unchanged",
        "configuration": {
            "datasets": args.datasets,
            "folds": args.folds,
            "top_k": args.top_k,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "seed": args.seed,
            "objective": "shared evidence-only candidate ranking plus none calibration",
        },
        "datasets": {},
    }
    for dataset_name in args.datasets:
        folds = []
        for fold in args.folds:
            print(f"running {dataset_name} fold {fold}", flush=True)
            folds.append(_run_fold(dataset_name, fold, args))
        summary = _summarize(dataset_name, folds)
        result["datasets"][dataset_name] = {"summary": summary, "folds": folds}
        b = summary["event"]["baseline"]
        v = summary["event"]["verified"]
        print(
            f"{dataset_name.upper()} event MRR {b['mrr']['mean']:.3f} -> {v['mrr']['mean']:.3f}; "
            f"Hit@1 {b['hit_at_1']['mean']:.3f} -> {v['hit_at_1']['mean']:.3f}; "
            f"fix/harm {summary['wrong_to_correct']}/{summary['correct_to_wrong']}", flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
