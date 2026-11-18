#!/usr/bin/env python3
"""Run the leakage-controlled LTFP failure-onset protocol.

One invocation handles one scenario-disjoint test fold.  ``--mode full`` first
selects hyperparameters using seed 7 and validation AUPRC only, then trains the
five requested initialization seeds.  Test labels are never used for model,
temperature, or threshold selection.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import load_rcaeval_cases  # noqa: E402
from ltfp.dataset_v2 import (  # noqa: E402
    build_failure_windows,
    feature_exclusions_from_manifest,
    fit_feature_schema,
    indices_for_cases,
    service_signal,
    standardize_train_only,
)
from ltfp.evaluation_v2 import (  # noqa: E402
    evaluate_events,
    evaluate_first_alert_roots,
    window_prediction_frame,
)
from ltfp.metrics import (  # noqa: E402
    assert_nested_probabilities,
    average_precision,
    evaluate_multi_horizon_v2,
    fit_shared_temperature,
    sigmoid_temperature,
    tune_shared_threshold,
)
from ltfp.protocol import (  # noqa: E402
    check_horizon_support,
    combine_topology_and_correlation,
    fold_roles,
    label_quality_gate,
    load_label_manifest,
    load_split_manifest,
    load_topology_manifest,
    normalize_system,
    runtime_metadata,
    topology_adjacency,
)
from ltfp.torch_models import LTFPConfig, LTFPModel  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int
    graph_layers: int
    correlation_top_k: int
    dropout: float
    learning_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--topology-manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7, help="Single-seed mode and fold-construction audit seed.")
    parser.add_argument("--initialization-seeds", default="7,17,27,37,47")
    parser.add_argument("--grid-seconds", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--horizons", default="60,180,300")
    parser.add_argument("--primary-horizon", type=float, default=300.0)
    parser.add_argument("--max-features", type=int, default=180)
    parser.add_argument("--allowed-modalities", default="metric,log,trace")
    parser.add_argument("--case-limit-per-role", type=int, default=None)
    parser.add_argument("--allow-degenerate-smoke", action="store_true")
    parser.add_argument("--mode", choices=("train", "search", "full"), default="train")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--correlation-top-k", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tcn-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--fusion-mode", choices=("gated", "concat"), default="gated")
    parser.add_argument("--independent-horizons", action="store_true")
    parser.add_argument("--variant-name", default="main")
    parser.add_argument(
        "--graph-mode",
        choices=("combined", "topology-only", "correlation-only", "no-graph"),
        default="combined",
    )
    parser.add_argument("--exclude-target-sli", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/failure_onset_v2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _resolve_topology_path(argument: str | Path, system: str) -> Path:
    path = Path(argument)
    if path.is_file():
        return path
    candidates = sorted(path.glob(f"{normalize_system(system)}*.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one topology manifest for {system} under {path}, found {candidates}")
    return candidates[0]


def _select_role_cases(cases, roles: dict[str, set[str]], limit_per_role: int | None):
    by_id = {str(case.case_id): case for case in cases}
    missing = sorted(set().union(*roles.values()) - set(by_id))
    # A global split manifest normally contains all seven subsets. Missing IDs
    # outside the selected --case-root are expected, so filter each role.
    selected_roles = {role: sorted(ids & set(by_id)) for role, ids in roles.items()}
    if limit_per_role is not None:
        selected_roles = {role: ids[: int(limit_per_role)] for role, ids in selected_roles.items()}
    if any(not ids for ids in selected_roles.values()):
        raise ValueError(
            "selected case root has an empty train/validation/test role; "
            f"role counts={ {key: len(value) for key, value in selected_roles.items()} }, "
            f"manifest IDs absent from this root={len(missing)}"
        )
    selected_ids = set().union(*map(set, selected_roles.values()))
    selected_cases = [by_id[case_id] for case_id in sorted(selected_ids)]
    return selected_cases, {key: set(value) for key, value in selected_roles.items()}


def _model_adjacency(dependency, correlation, graph_mode: str):
    if graph_mode == "combined":
        return np.maximum(dependency, correlation).astype(np.float32)
    if graph_mode == "topology-only":
        return dependency.astype(np.float32)
    if graph_mode == "correlation-only":
        return correlation.astype(np.float32)
    if graph_mode == "no-graph":
        return np.eye(len(dependency), dtype=np.float32)
    raise ValueError(graph_mode)


def _set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def _masked_failure_root_loss(
    outputs,
    labels,
    label_mask,
    root_targets,
    root_active,
    pos_weight,
    root_weight: float,
):
    import torch.nn.functional as F

    element = F.binary_cross_entropy_with_logits(
        outputs["fault_logits"],
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    valid = label_mask.to(element.dtype)
    failure_loss = (element * valid).sum() / valid.sum().clamp_min(1.0)
    root_mask = root_active & (root_targets >= 0)
    if root_mask.any():
        root_loss = F.cross_entropy(outputs["root_logits"][root_mask], root_targets[root_mask])
    else:
        root_loss = failure_loss.new_zeros(())
    return failure_loss + float(root_weight) * root_loss, failure_loss.detach(), root_loss.detach()


def _predict(model, X, availability, adjacency, indices, batch_size, device):
    import torch

    logits = []
    root_logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            xb = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
            mb = torch.as_tensor(availability[idx], dtype=torch.bool, device=device)
            output = model(xb, adjacency, modality_mask=mb)
            logits.append(output["fault_logits"].cpu().numpy())
            root_logits.append(output["root_logits"].cpu().numpy())
    return np.vstack(logits), np.vstack(root_logits)


def _validation_primary_auprc(dataset, val_idx, logits, primary_horizon: float) -> float:
    matches = np.flatnonzero(np.isclose(np.asarray(dataset.horizons), float(primary_horizon)))
    if len(matches) != 1:
        raise ValueError(f"primary horizon {primary_horizon:g}s is not present in {dataset.horizons}")
    horizon_idx = int(matches[0])
    valid = dataset.label_mask[val_idx, horizon_idx]
    if not valid.any():
        return 0.0
    scores = sigmoid_temperature(logits[valid, horizon_idx])
    return average_precision(dataset.Y[val_idx[valid], horizon_idx], scores)


def train_one(
    dataset,
    X,
    adjacency_np,
    indices,
    config: TrainConfig,
    args,
    seed: int,
):
    import torch

    _set_seed(seed)
    device = torch.device(args.device)
    adjacency = torch.as_tensor(adjacency_np, dtype=torch.float32, device=device)
    model = LTFPModel(
        LTFPConfig(
            input_dim=dataset.schema.input_dim,
            hidden_dim=config.hidden_dim,
            n_fault_types=0,
            n_horizons=len(dataset.horizons),
            n_tcn_layers=args.tcn_layers,
            n_graph_layers=config.graph_layers,
            dropout=config.dropout,
            modality_dims=dataset.schema.modality_dims,
            enable_type_head=False,
            enable_auxiliary_heads=False,
            enable_root_prior=False,
            fusion_mode=args.fusion_mode,
            monotonic_horizons=not args.independent_horizons,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    train_idx = indices["train"]
    train_y = dataset.Y[train_idx]
    train_mask = dataset.label_mask[train_idx]
    positives = (train_y * train_mask).sum(axis=0)
    negatives = ((1.0 - train_y) * train_mask).sum(axis=0)
    pos_weight = np.divide(negatives, np.maximum(positives, 1.0))
    pos_weight = torch.as_tensor(np.clip(pos_weight, 0.25, 20.0), dtype=torch.float32, device=device)
    y = torch.as_tensor(dataset.Y, dtype=torch.float32, device=device)
    label_mask = torch.as_tensor(dataset.label_mask, dtype=torch.bool, device=device)
    roots = torch.as_tensor(dataset.root_targets, dtype=torch.long, device=device)
    root_active = torch.as_tensor(dataset.root_active, dtype=torch.bool, device=device)

    best_score = -math.inf
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(int(args.max_epochs)):
        model.train()
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        batch_losses = []
        failure_losses = []
        root_losses = []
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            xb = torch.as_tensor(X[batch], dtype=torch.float32, device=device)
            mb = torch.as_tensor(dataset.availability_mask[batch], dtype=torch.bool, device=device)
            output = model(xb, adjacency, modality_mask=mb)
            loss, failure_loss, root_loss = _masked_failure_root_loss(
                output,
                y[batch],
                label_mask[batch],
                roots[batch],
                root_active[batch],
                pos_weight,
                args.root_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            failure_losses.append(float(failure_loss.cpu()))
            root_losses.append(float(root_loss.cpu()))

        validation_logits, _ = _predict(
            model,
            X,
            dataset.availability_mask,
            adjacency,
            indices["validation"],
            args.batch_size,
            device,
        )
        validation_score = _validation_primary_auprc(
            dataset,
            indices["validation"],
            validation_logits,
            args.primary_horizon,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(batch_losses)),
                "failure_loss": float(np.mean(failure_losses)),
                "root_loss": float(np.mean(root_losses)),
                "validation_primary_auprc": float(validation_score),
            }
        )
        if validation_score > best_score + 1e-12:
            best_score = float(validation_score)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_score, best_epoch, pos_weight.detach().cpu().numpy()


def _prepare(args):
    label_frame = load_label_manifest(
        args.label_manifest,
        require_confirmatory=not args.allow_degenerate_smoke,
    )
    if "review_status" in label_frame:
        pending = label_frame["review_status"].astype(str).str.lower().isin(
            {"pending", "unreviewed", "candidate"}
        )
        if pending.any():
            raise ValueError(
                "label manifest contains unreviewed onset candidates; the confirmatory runner "
                "accepts only independently reviewed/adjudicated failure_time labels"
            )
    if label_frame["failure_time"].notna().sum() == 0:
        raise ValueError("label manifest contains no accepted failure onsets")
    quality_gate = label_quality_gate(label_frame)
    if quality_gate["route_status"] != "eligible_for_human_audit" and not args.allow_degenerate_smoke:
        raise ValueError(
            "failure-onset label manifest fails the fixed >=60% onset / >=4-subset route gate; "
            "it cannot enter a confirmatory run"
        )
    split_frame = load_split_manifest(args.split_manifest, label_records=label_frame)
    roles_all = fold_roles(split_frame, args.fold)
    label_records = _json_records(label_frame)
    cases = load_rcaeval_cases(
        args.case_root,
        label_manifest={"cases": label_records},
        interval_seconds=args.grid_seconds,
    )
    if not cases:
        raise ValueError(f"no cases found under {args.case_root}")
    systems = {normalize_system(case.system or "") for case in cases}
    if len(systems) != 1:
        raise ValueError(f"one invocation must cover one system, found {sorted(systems)}")
    system = next(iter(systems))
    topology_path = _resolve_topology_path(args.topology_manifest, system)
    topology = load_topology_manifest(topology_path, expected_system=system)
    cases, role_ids = _select_role_cases(cases, roles_all, args.case_limit_per_role)
    train_cases = [case for case in cases if str(case.case_id) in role_ids["train"]]
    selected_label_records = [row for row in label_records if str(row["case_id"]) in set().union(*role_ids.values())]
    excluded = feature_exclusions_from_manifest(selected_label_records) if args.exclude_target_sli else ()
    allowed_modalities = tuple(
        item.strip() for item in args.allowed_modalities.split(",") if item.strip()
    )
    schema = fit_feature_schema(
        train_cases,
        topology,
        max_features=args.max_features,
        excluded_features=excluded,
        allowed_modalities=allowed_modalities,
    )
    horizons = tuple(float(item) for item in args.horizons.split(",") if item.strip())
    if not any(np.isclose(float(args.primary_horizon), horizon) for horizon in horizons):
        raise ValueError(
            f"--primary-horizon {args.primary_horizon:g} must be exactly one of --horizons {horizons}"
        )
    dataset = build_failure_windows(
        cases,
        schema,
        topology,
        horizons=horizons,
        window_size=args.window_size,
        stride=args.stride,
        grid_seconds=args.grid_seconds,
    )
    indices = {role: indices_for_cases(dataset, ids) for role, ids in role_ids.items()}
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError(f"one or more roles produced no windows: { {k: len(v) for k, v in indices.items()} }")
    X, mean, scale = standardize_train_only(dataset, indices["train"])
    if not args.allow_degenerate_smoke:
        for horizon_idx, horizon in enumerate(dataset.horizons):
            valid = dataset.label_mask[indices["test"], horizon_idx]
            classes = np.unique(dataset.Y[indices["test"]][valid, horizon_idx])
            if len(classes) < 2:
                raise ValueError(
                    f"main test horizon {horizon:g}s is single-class; this run cannot enter the paper"
                )
    dependency = topology_adjacency(topology, bidirectional=True)
    return {
        "labels": label_frame,
        "splits": split_frame,
        "cases": cases,
        "role_ids": role_ids,
        "topology": topology,
        "topology_path": topology_path,
        "dataset": dataset,
        "indices": indices,
        "X": X,
        "mean": mean,
        "scale": scale,
        "dependency": dependency,
        "selected_label_records": selected_label_records,
        "label_quality_gate": quality_gate,
    }


def _adjacency_for(prepared, top_k: int, graph_mode: str):
    dataset = prepared["dataset"]
    train_signal = service_signal(dataset, prepared["X"], prepared["indices"]["train"])
    _, correlation = combine_topology_and_correlation(prepared["dependency"], train_signal, top_k=top_k)
    return _model_adjacency(prepared["dependency"], correlation, graph_mode), correlation


def _hyperparameter_grid():
    for values in itertools.product(
        (16, 32),
        (1, 2),
        (2, 4),
        (0.1, 0.2),
        (1e-3, 3e-4),
    ):
        yield TrainConfig(*values)


def run_search(prepared, args, output_root: Path) -> TrainConfig:
    rows = []
    for search_id, config in enumerate(_hyperparameter_grid()):
        adjacency, _ = _adjacency_for(prepared, config.correlation_top_k, args.graph_mode)
        started = time.time()
        _, history, score, best_epoch, _ = train_one(
            prepared["dataset"],
            prepared["X"],
            adjacency,
            prepared["indices"],
            config,
            args,
            seed=7,
        )
        rows.append(
            {
                "search_id": search_id,
                **asdict(config),
                "validation_primary_auprc": score,
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "runtime_seconds": time.time() - started,
            }
        )
        pd.DataFrame(rows).to_csv(output_root / "validation_search.csv", index=False)
    search = pd.DataFrame(rows).sort_values(
        ["validation_primary_auprc", "hidden_dim", "graph_layers", "correlation_top_k"],
        ascending=[False, True, True, True],
    )
    best = search.iloc[0]
    config = TrainConfig(
        hidden_dim=int(best.hidden_dim),
        graph_layers=int(best.graph_layers),
        correlation_top_k=int(best.correlation_top_k),
        dropout=float(best.dropout),
        learning_rate=float(best.learning_rate),
    )
    (output_root / "best_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return config


def _evaluate_seed(prepared, args, config: TrainConfig, seed: int, output_root: Path) -> dict:
    import torch

    dataset = prepared["dataset"]
    model_name = "LTFP" if args.variant_name == "main" else f"LTFP[{args.variant_name}]"
    indices = prepared["indices"]
    adjacency, correlation = _adjacency_for(prepared, config.correlation_top_k, args.graph_mode)
    model, history, validation_score, best_epoch, pos_weight = train_one(
        dataset,
        prepared["X"],
        adjacency,
        indices,
        config,
        args,
        seed=seed,
    )
    device = torch.device(args.device)
    adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    validation_logits, _ = _predict(
        model,
        prepared["X"],
        dataset.availability_mask,
        adjacency_tensor,
        indices["validation"],
        args.batch_size,
        device,
    )
    validation_mask = dataset.label_mask[indices["validation"]]
    temperature = fit_shared_temperature(
        validation_logits[validation_mask],
        dataset.Y[indices["validation"]][validation_mask],
    )
    validation_probability = sigmoid_temperature(validation_logits, temperature)
    threshold = tune_shared_threshold(
        dataset.Y[indices["validation"]],
        validation_probability,
        label_mask=validation_mask,
    )

    test_logits, test_root_logits = _predict(
        model,
        prepared["X"],
        dataset.availability_mask,
        adjacency_tensor,
        indices["test"],
        args.batch_size,
        device,
    )
    raw_probability = sigmoid_temperature(test_logits, 1.0)
    calibrated_probability = sigmoid_temperature(test_logits, temperature)
    if not args.independent_horizons:
        assert_nested_probabilities(raw_probability)
        assert_nested_probabilities(calibrated_probability)
    alarms = calibrated_probability >= threshold
    if (
        not args.independent_horizons
        and alarms.shape[1] > 1
        and np.any(np.diff(alarms.astype(int), axis=1) < 0)
    ):
        raise AssertionError("shared threshold did not preserve nested binary alarms")

    point_rows = evaluate_multi_horizon_v2(
        dataset.Y[indices["test"]],
        calibrated_probability,
        dataset.horizons,
        threshold,
        label_mask=dataset.label_mask[indices["test"]],
    )
    raw_rows = evaluate_multi_horizon_v2(
        dataset.Y[indices["test"]],
        raw_probability,
        dataset.horizons,
        threshold,
        label_mask=dataset.label_mask[indices["test"]],
    )
    for calibrated, raw in zip(point_rows, raw_rows):
        calibrated["raw_auprc"] = raw["auprc"]
        calibrated["raw_brier"] = raw["brier"]
        calibrated["raw_ece"] = raw["ece"]
        calibrated["seed"] = int(seed)
        calibrated["fold"] = int(args.fold)
        calibrated["model"] = model_name

    event_rows, per_case_events = evaluate_events(
        dataset,
        indices["test"],
        calibrated_probability,
        threshold,
    )
    for row in event_rows:
        row.update({"seed": int(seed), "fold": int(args.fold), "model": model_name})
    root_summary, root_cases = evaluate_first_alert_roots(
        dataset,
        indices["test"],
        calibrated_probability,
        test_root_logits,
        threshold,
        primary_horizon=args.primary_horizon,
    )
    predictions = window_prediction_frame(
        dataset,
        indices["test"],
        raw_probability,
        calibrated_probability,
        test_root_logits,
    )
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(seed_dir / "point_metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(seed_dir / "event_metrics.csv", index=False)
    per_case_events.to_csv(seed_dir / "case_events.csv", index=False)
    root_cases.to_csv(seed_dir / "case_roots.csv", index=False)
    predictions.to_csv(seed_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), seed_dir / "model.pt")
    np.savez_compressed(
        seed_dir / "graph_and_preprocessing.npz",
        adjacency=adjacency,
        dependency=prepared["dependency"],
        correlation=correlation,
        mean=prepared["mean"],
        scale=prepared["scale"],
    )
    summary = {
        "seed": int(seed),
        "fold": int(args.fold),
        "model": model_name,
        "config": asdict(config),
        "validation_primary_auprc": float(validation_score),
        "best_epoch": int(best_epoch),
        "temperature": float(temperature),
        "shared_threshold": float(threshold),
        "pos_weight": pos_weight.tolist(),
        "point_metrics": point_rows,
        "event_metrics": event_rows,
        "root_metrics": root_summary,
    }
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return summary


def run() -> dict:
    args = parse_args()
    if args.grid_seconds != 15:
        raise ValueError("submission protocol fixes --grid-seconds to 15")
    if args.window_size * args.grid_seconds != 360:
        raise ValueError("submission protocol requires a 24-step / 6-minute observation window")
    if args.stride * args.grid_seconds != 30:
        raise ValueError("submission protocol requires a 30-second prediction cadence")
    prepared = _prepare(args)
    dataset = prepared["dataset"]
    subset = Path(args.case_root).name.lower()
    decoder_name = "independent" if args.independent_horizons else "monotonic"
    target_sli_name = "without-target-sli" if args.exclude_target_sli else "with-target-sli"
    run_name = args.run_name or "_".join(
        (
            subset,
            f"fold{args.fold}",
            args.variant_name,
            args.graph_mode,
            args.fusion_mode,
            decoder_name,
            target_sli_name,
        )
    )
    output_root = Path(args.output_dir) / run_name
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty v2 run directory without --overwrite: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "version": "failure-onset-v2",
        "task": "fault-to-service-failure-onset",
        "case_root": str(Path(args.case_root)),
        "fold": int(args.fold),
        "roles": {role: sorted(ids) for role, ids in prepared["role_ids"].items()},
        "horizons": list(dataset.horizons),
        "grid_seconds": args.grid_seconds,
        "window_size": args.window_size,
        "stride": args.stride,
        "prediction_cadence_seconds": args.stride * args.grid_seconds,
        "feature_schema": dataset.schema.to_dict(),
        "allowed_modalities": [
            item.strip() for item in args.allowed_modalities.split(",") if item.strip()
        ],
        "fusion_mode": args.fusion_mode,
        "monotonic_horizons": not args.independent_horizons,
        "variant_name": args.variant_name,
        "label_quality_gate": prepared["label_quality_gate"],
        "topology": {
            "path": str(prepared["topology_path"]),
            "version": prepared["topology"].version,
            "services": list(prepared["topology"].services),
            "n_root_candidates": len(prepared["topology"].services),
        },
        "window_counts": {role: int(len(indices)) for role, indices in prepared["indices"].items()},
        "case_counts": {role: int(len(ids)) for role, ids in prepared["role_ids"].items()},
        "horizon_support_train": check_horizon_support(
            dataset.Y[prepared["indices"]["train"]],
            dataset.horizons,
            np.asarray(dataset.case_ids)[prepared["indices"]["train"]],
        ),
        "runtime": runtime_metadata(
            sys.argv,
            [args.label_manifest, args.split_manifest, prepared["topology_path"]],
            repository_root=ROOT / "6a39ee5ca99c677533eea511",
            source_paths=[ROOT / "src", ROOT / "experiments", ROOT / "scripts"],
        ),
    }
    unsupported_long = [
        row
        for row in protocol["horizon_support_train"]
        if row["horizon"] > 300.0 and not row["supported"]
    ]
    if unsupported_long and not args.allow_degenerate_smoke:
        raise ValueError(
            "appendix horizons violate the pre-registered prevalence/event support gate: "
            f"{unsupported_long}"
        )
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output_root / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")

    default_config = TrainConfig(
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        correlation_top_k=args.correlation_top_k,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    if args.mode in {"search", "full"}:
        best_config = run_search(prepared, args, output_root)
        if args.mode == "search":
            return {"mode": "search", "best_config": asdict(best_config), "output": str(output_root)}
    else:
        best_config = default_config

    seeds = (
        [int(item) for item in args.initialization_seeds.split(",") if item.strip()]
        if args.mode == "full"
        else [int(args.seed)]
    )
    summaries = [_evaluate_seed(prepared, args, best_config, seed, output_root) for seed in seeds]
    point_rows = [row for summary in summaries for row in summary["point_metrics"]]
    event_rows = [row for summary in summaries for row in summary["event_metrics"]]
    pd.DataFrame(point_rows).to_csv(output_root / "point_metrics_all_seeds.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output_root / "event_metrics_all_seeds.csv", index=False)
    final = {
        "mode": args.mode,
        "run_name": run_name,
        "fold": int(args.fold),
        "seeds": seeds,
        "best_config": asdict(best_config),
        "output": str(output_root),
        "completed": True,
    }
    (output_root / "run_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


if __name__ == "__main__":
    run()
