"""Dataset-independent experiment orchestration for aligned graph windows."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.dataset import chronological_split, purged_event_split
from data.dataset import GraphWindowDataset
from models.factory import build_model, training_policy
from rca.ranking import masking_importance
from training.engine import (
    choose_device,
    class_pos_weight,
    fit_feature_scaler,
    run_epoch,
    seed_everything,
)
from training.metrics import classification_metrics, multiclass_metrics, rca_metrics
from training.normal_reference import configure_metric_reference


def _utc_boundary(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if len(value) != 10:
            raise ValueError("split timestamps need an explicit UTC offset")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _configure_root_event_weights(
    dataset: GraphWindowDataset,
    splits: tuple[list[int], list[int], list[int]],
    enabled: bool,
) -> dict[str, object]:
    """Make each labeled failure event contribute equal total rank weight."""
    dataset.root_event_weights = np.ones(len(dataset.timestamps), dtype=np.float32)
    report: dict[str, object] = {"enabled": enabled, "splits": {}}
    if not enabled:
        return report

    for split_name, ends in zip(("train", "validation", "test"), splits):
        event_rows: dict[str, list[int]] = {}
        for end in ends:
            event_id = dataset.event_ids[end]
            if (
                event_id
                and dataset.failure_targets[end] == 1
                and dataset.node_targets[end].sum() > 0
            ):
                event_rows.setdefault(event_id, []).append(end)
        attributed_windows = sum(map(len, event_rows.values()))
        average_length = attributed_windows / max(len(event_rows), 1)
        for rows in event_rows.values():
            weight = average_length / len(rows)
            dataset.root_event_weights[rows] = weight
        lengths = [len(rows) for rows in event_rows.values()]
        report["splits"][split_name] = {
            "events": len(event_rows),
            "attributed_windows": attributed_windows,
            "minimum_event_windows": min(lengths, default=0),
            "maximum_event_windows": max(lengths, default=0),
        }
    return report


def _event_rca_metrics(
    dataset: GraphWindowDataset,
    ends: list[int],
    output: dict[str, np.ndarray],
) -> dict[str, float | int | str | None]:
    """Aggregate window scores exactly once per attributed failure event."""
    if len(ends) != len(output["node_scores"]):
        raise ValueError("split endpoints and RCA outputs must have equal length")
    event_rows: dict[str, list[int]] = {}
    for row, end in enumerate(ends):
        event_id = dataset.event_ids[end]
        if event_id and output["node_targets"][row].sum() > 0:
            event_rows.setdefault(event_id, []).append(row)

    event_targets = []
    event_scores = []
    for event_id, rows in event_rows.items():
        targets = output["node_targets"][rows]
        if not np.all(targets == targets[0]):
            raise ValueError(f"root labels change within failure event {event_id}")
        event_targets.append(targets[0])
        event_scores.append(output["node_scores"][rows].mean(axis=0))

    services = len(dataset.services)
    metrics = rca_metrics(
        np.stack(event_targets) if event_targets else np.empty((0, services)),
        np.stack(event_scores) if event_scores else np.empty((0, services)),
    )
    metrics["positive_events"] = metrics.pop("positive_windows")
    metrics["aggregation"] = (
        "mean scores over available attributed windows in each event; retrospective"
    )
    return metrics


def run_training(
    dataset: GraphWindowDataset,
    args: argparse.Namespace,
    graph_source: str,
    label_definition: str | dict,
) -> dict:
    """Train one dataset with train-only normalization and held-out evaluation.

    Adapters must provide binary failure/node labels with documented semantics.
    Different datasets train separate weights and never share fitted scalers.
    """
    if getattr(dataset, "failure_labels_available", True) is False:
        raise ValueError("application failure labels are unavailable for this dataset")
    if getattr(args, "model", None) in {
        "ltfp-local-normal-evidence", "ltfp-state-relation-evidence",
        "ltfp-raw-metric-bypass-evidence", "ltfp-metric-deviation-bypass-evidence",
    }:
        raise ValueError(
            "normal-evidence pilot requires a frozen pretrained LTFP; "
            "use scripts/run_state_relation_pilot.py"
        )

    # Scaling replaces the feature array on a shallow copy, so repeated runs
    # on the same adapter still fit against the original raw feature values.
    dataset = copy.copy(dataset)
    seed_everything(args.seed)
    device = choose_device(args.device)
    split_report = {"policy": "chronological"}
    train_until = getattr(args, "train_until", None)
    validation_until = getattr(args, "validation_until", None)
    if bool(train_until) != bool(validation_until):
        raise ValueError("train-until and validation-until must be supplied together")
    if train_until:
        if not isinstance(label_definition, dict):
            raise ValueError("calendar split requires a versioned label schema")
        if dataset.adjacency.ndim != 3:
            raise ValueError("calendar split requires the temporal graph")
        cuts = (_utc_boundary(train_until), _utc_boundary(validation_until))
        (train_ends, val_ends, test_ends), split_report = purged_event_split(
            dataset, label_definition, cuts
        )
    elif hasattr(dataset, "predefined_splits"):
        train_ends, val_ends, test_ends = dataset.predefined_splits
        split_report = dataset.split_report
    elif isinstance(label_definition, dict) and label_definition.get("split_policy") == "purged_events":
        if dataset.adjacency.ndim != 3:
            raise ValueError("event protocol requires the temporal graph; cached static graph may use later evidence")
        (train_ends, val_ends, test_ends), split_report = purged_event_split(dataset, label_definition)
    else:
        train_ends, val_ends, test_ends = chronological_split(dataset.sample_indices)
    root_event_weighting = _configure_root_event_weights(
        dataset,
        (train_ends, val_ends, test_ends),
        bool(getattr(args, "event_balanced_root_loss", False)),
    )
    baseline = None
    if (isinstance(label_definition, dict)
            and {"trace_span_count", "trace_5xx_count"}.issubset(dataset.feature_columns)):
        entry = [dataset.services.index(s) for s in label_definition["entry_services"]]
        count_col = dataset.feature_columns.index("trace_span_count")
        error_col = dataset.feature_columns.index("trace_5xx_count")
        current = dataset.features[test_ends][:, entry, :]
        counts = current[:, :, count_col].sum(axis=1)
        errors = current[:, :, error_col].sum(axis=1)
        observed = counts >= label_definition["min_entry_spans"]
        rates = errors / np.maximum(counts, 1)
        flags = rates >= label_definition["error_rate_threshold"]
        baseline = {
            "definition": "current-minute entry error threshold predicts next-minute failure; no future confirmation",
            "evaluated_windows": int(observed.sum()),
            "metrics": classification_metrics(dataset.failure_targets[test_ends][observed], flags[observed].astype(float))
                if observed.any() else None,
            "all_test_windows": {
                "missing_current_entry_windows": int((~observed).sum()),
                "missing_current_entry_policy": "predict normal when current entry count is below minimum",
                "metrics": classification_metrics(
                    dataset.failure_targets[test_ends],
                    np.where(observed, flags, False).astype(float),
                ),
            },
        }
    mean, scale = fit_feature_scaler(dataset, train_ends)
    train_index = {end: index for index, end in enumerate(dataset.sample_indices)}
    subsets = [
        Subset(dataset, [train_index[end] for end in ends])
        for ends in (train_ends, val_ends, test_ends)
    ]
    loaders = [
        DataLoader(subset, batch_size=args.batch_size, shuffle=(index == 0), num_workers=0)
        for index, subset in enumerate(subsets)
    ]
    model_name = getattr(args, "model", "gcn-gru")
    train_labels = torch.tensor(
        [int(dataset.failure_targets[end]) for end in train_ends], dtype=torch.float32
    )
    if train_labels.unique().numel() < 2:
        raise ValueError("training split needs both normal and fault windows")
    train_node_labels = torch.from_numpy(dataset.node_targets[train_ends])
    known_roots = (dataset.failure_targets[train_ends] == 0) | (dataset.node_targets[train_ends].max(axis=1) > 0)
    train_node_labels = train_node_labels[known_roots]
    failure_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, class_pos_weight(train_labels, torch.device("cpu")).item()],
            dtype=torch.float32,
            device=device,
        )
    )
    node_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(train_node_labels, device))
    fault_type_loss_weight = float(getattr(args, "fault_type_loss_weight", 0.0))
    fault_type_loss_fn: nn.Module | None = None
    fault_type_names = tuple(getattr(dataset, "fault_type_names", ()))
    if model_name == "ltfp-type-aux":
        if not fault_type_names or not hasattr(dataset, "fault_type_targets"):
            raise ValueError("ltfp-type-aux requires RE2 case fault-type labels")
        if fault_type_loss_weight <= 0:
            raise ValueError("ltfp-type-aux requires a positive fault-type loss weight")
        fault_ends = [end for end in train_ends if dataset.failure_targets[end] == 1]
        fault_targets = torch.tensor(
            dataset.fault_type_targets[fault_ends], dtype=torch.long
        )
        counts = torch.bincount(fault_targets, minlength=len(fault_type_names)).float()
        observed = counts > 0
        class_weights = torch.zeros_like(counts)
        class_weights[observed] = (
            counts[observed].sum() / (int(observed.sum()) * counts[observed])
        )
        fault_type_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif fault_type_loss_weight:
        raise ValueError("fault-type loss is only supported by ltfp-type-aux")
    joint_service_pooling = getattr(args, "joint_service_pooling", False)
    policy = training_policy(
        model_name,
        node_loss_weight=args.node_loss_weight,
        rank_loss_weight=getattr(args, "rank_loss_weight", 0.0),
        rca_method=getattr(args, "rca_method", "masking"),
    )
    node_loss_weight = policy.node_loss_weight
    rank_loss_weight = policy.rank_loss_weight
    telemetry_loss_weight = float(getattr(args, "telemetry_loss_weight", 0.0))
    evidence_loss_weight = float(getattr(args, "evidence_loss_weight", 0.0))
    if min(telemetry_loss_weight, evidence_loss_weight) < 0:
        raise ValueError("auxiliary loss weights must be nonnegative")
    built = build_model(
        model_name,
        feature_columns=dataset.feature_columns,
        num_services=len(dataset.services),
        history=dataset.history,
        num_fault_types=len(fault_type_names),
        joint_service_pooling=joint_service_pooling,
    )
    model = built.model.to(device)
    model_config = built.checkpoint_config
    if model_name in {"joint-granularity", "joint-granularity-encoded-local",
                "joint-granularity-null-selection"}:
        if rank_loss_weight <= 0:
            raise ValueError(f"{model_name} requires a positive root ranking weight")
        model_config["normal_reference_fit"] = configure_metric_reference(model, dataset, train_ends)
    if model_name == "ltfp-type-aux":
        model_config["fault_type_names"] = list(fault_type_names)
        model_config["fault_type_train_positive_counts"] = {
            name: int(count) for name, count in zip(fault_type_names, counts)
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    root_checkpoint_path = args.output_dir / "best_root_sidecar.pt"
    root_prefixes = tuple(
        getattr(model, "root_sidecar_parameter_prefixes", ())
    )
    checkpoint_selection = getattr(args, "checkpoint_selection", "validation-loss")
    if checkpoint_selection not in {"validation-loss", "validation-ap"}:
        raise ValueError("checkpoint_selection must be validation-loss or validation-ap")
    best_val_loss = float("inf")
    best_val_ap = float("-inf")
    best_val_root_mrr = float("-inf")
    best_root_epoch: int | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(
            model,
            loaders[0],
            device,
            failure_loss_fn,
            node_loss_fn,
            node_loss_weight,
            optimizer,
            rank_loss_weight=rank_loss_weight,
            telemetry_loss_weight=telemetry_loss_weight,
            evidence_loss_weight=evidence_loss_weight,
            fault_type_loss_fn=fault_type_loss_fn,
            fault_type_loss_weight=fault_type_loss_weight,
        )
        val_loss, val_output = run_epoch(
            model,
            loaders[1],
            device,
            failure_loss_fn,
            node_loss_fn,
            node_loss_weight,
            rank_loss_weight=rank_loss_weight,
            telemetry_loss_weight=telemetry_loss_weight,
            evidence_loss_weight=evidence_loss_weight,
            fault_type_loss_fn=fault_type_loss_fn,
            fault_type_loss_weight=fault_type_loss_weight,
        )
        val_ap = classification_metrics(
            val_output["labels"], val_output["probabilities"]
        )["average_precision"]
        val_root_mrr = _event_rca_metrics(dataset, val_ends, val_output)["mrr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ap": val_ap,
            "val_root_event_mrr": val_root_mrr,
        })
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_ap={val_ap:.5f} "
            f"val_root_mrr={val_root_mrr if val_root_mrr is not None else float('nan'):.5f}"
        )
        improved = (
            val_loss < best_val_loss
            if checkpoint_selection == "validation-loss"
            else val_ap > best_val_ap
        )
        if improved:
            best_val_loss = val_loss
            best_val_ap = val_ap
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": mean,
                    "feature_scale": scale,
                    "services": list(dataset.services),
                    "feature_columns": list(dataset.feature_columns),
                    "history": args.history,
                    "graph_source": graph_source,
                    "coverage_policy": dataset.coverage_policy,
                    "adjacency": dataset.adjacency.cpu(),
                    "model_config": model_config,
                    "seed": args.seed,
                    "label_definition": label_definition,
                },
                checkpoint_path,
            )
        if (
            root_prefixes
            and val_root_mrr is not None
            and val_root_mrr > best_val_root_mrr
        ):
            best_val_root_mrr = val_root_mrr
            best_root_epoch = epoch
            root_state = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
                if name.startswith(root_prefixes)
            }
            torch.save(
                {
                    "model_state_dict": root_state,
                    "validation_mrr": val_root_mrr,
                    "epoch": epoch,
                },
                root_checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if root_prefixes:
        root_checkpoint = torch.load(
            root_checkpoint_path, map_location=device, weights_only=False
        )
        combined_state = model.state_dict()
        combined_state.update(root_checkpoint["model_state_dict"])
        model.load_state_dict(combined_state)
    test_loss, test_output = run_epoch(
        model, loaders[2], device, failure_loss_fn, node_loss_fn, node_loss_weight,
        rank_loss_weight=rank_loss_weight,
        telemetry_loss_weight=telemetry_loss_weight,
        evidence_loss_weight=evidence_loss_weight,
        fault_type_loss_fn=fault_type_loss_fn,
        fault_type_loss_weight=fault_type_loss_weight,
    )
    method = policy.rca_method
    if method == "masking":
        scores = []
        model.eval()
        for batch in loaders[2]:
            scores.append(
                masking_importance(
                    model, batch["node_features"].to(device), batch["adjacency"].to(device)
                )
                .cpu()
                .numpy()
            )
        test_output["node_scores"] = np.concatenate(scores)
    predictions = []
    test_event_ids = [dataset.event_ids[end] for end in test_ends]
    for index, (end, event_id) in enumerate(zip(test_ends, test_event_ids)):
        timestamp = dataset.timestamps[end]
        scores = test_output["node_scores"][index]
        order = sorted(range(len(dataset.services)), key=lambda node: (-float(scores[node]), node))
        predictions.append(
            {
                "input_minute_utc": timestamp,
                "forecast": {
                    "mode": getattr(dataset, "forecast_mode", "point"),
                    "horizon_minutes": getattr(dataset, "forecast_horizon_minutes", 1),
                    "end_minute_utc": dataset.forecast_end_timestamps.get(end),
                },
                "failure_probability": float(test_output["probabilities"][index]),
                "failure_target": int(test_output["labels"][index]),
                "event_id": event_id,
                "rca_method": method,
                "true_services": [
                    service
                    for service, label in zip(dataset.services, test_output["node_targets"][index])
                    if label
                ],
                "ranking": [
                    {"service": dataset.services[node], "score": float(scores[node])}
                    for node in order
                ],
                "modality_routing": (
                    {
                        "names": model_config["modality_names"],
                        "failure": test_output["failure_modality_weights"][index]
                        .mean(axis=0)
                        .tolist(),
                        "rca": test_output["root_modality_weights"][index]
                        .mean(axis=0)
                        .tolist(),
                    }
                    if "failure_modality_weights" in test_output
                    else None
                ),
            }
        )
    event_rca = _event_rca_metrics(dataset, test_ends, test_output)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    metrics = {
        "rca_method": method,
        "split_protocol": split_report,
        "current_error_baseline": baseline,
        "root_label_coverage": {
            "test_failure_windows": int(test_output["labels"].sum()),
            "test_attributed_failure_windows": int((test_output["node_targets"].max(axis=1) > 0).sum()),
        },
        "rca_score_definition": (
            "nonnegative target-probability drop" if method == "masking" else "node sigmoid"
        ),
        "coverage": {
            "policy": dataset.coverage_policy,
            "invalid_panel_minutes": dataset.invalid_minutes,
            "resource_complete_minutes": int(dataset.complete_minutes.sum()),
            "any_modality_minutes": int(dataset.available_minutes.sum()),
            "usable_windows": len(dataset),
        },
        "rca_ties": {
            "all_services_equal_windows": int((
                test_output["node_scores"].max(axis=1)
                == test_output["node_scores"].min(axis=1)
            ).sum()),
            "tie_break": "service order; Hit/MRR may depend on this order",
        },
        "run_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "effective_loss_weights": {
            "node_bce": node_loss_weight,
            "root_ranking": rank_loss_weight,
            "telemetry_forecast": telemetry_loss_weight,
            "service_evidence": evidence_loss_weight,
            "fault_type_auxiliary": fault_type_loss_weight,
        },
        "root_event_weighting": root_event_weighting,
        "model": model_name,
        "forecast_task": {
            "mode": getattr(dataset, "forecast_mode", "point"),
            "horizon_minutes": getattr(dataset, "forecast_horizon_minutes", 1),
        },
        "split_positive_samples": {
            name: int(dataset.failure_targets[ends].sum())
            for name, ends in zip(
                ("train", "validation", "test"), (train_ends, val_ends, test_ends)
            )
        },
        "test_loss": test_loss,
        "failure_prediction": classification_metrics(
            test_output["labels"], test_output["probabilities"]
        ),
        "root_cause_proxy": rca_metrics(test_output["node_targets"], test_output["node_scores"]),
        "root_cause_event_proxy": event_rca,
        "graph_source": graph_source,
        "label_definition": label_definition,
        "split_sizes": {
            "train": len(train_ends),
            "validation": len(val_ends),
            "test": len(test_ends),
        },
        "training_history": history,
        "checkpoint_selection": checkpoint_selection,
        "root_checkpoint_selection": (
            {
                "metric": "validation-event-mrr",
                "best_value": best_val_root_mrr,
                "epoch": best_root_epoch,
                "path": str(root_checkpoint_path),
            }
            if root_prefixes
            else None
        ),
    }
    if "failure_modality_weights" in test_output:
        failure_routes = test_output["failure_modality_weights"]
        root_routes = test_output["root_modality_weights"]
        metrics["modality_routing"] = {
            "names": model_config["modality_names"],
            "failure_mean": failure_routes.mean(axis=(0, 1)).tolist(),
            "rca_mean": root_routes.mean(axis=(0, 1)).tolist(),
            "mean_absolute_task_gap": float(
                np.abs(failure_routes - root_routes).mean()
            ),
        }
    if "fault_type_probabilities" in test_output:
        metrics["fault_type_auxiliary"] = {
            "supervision_scope": "future user-visible failure windows only",
            "class_names": list(fault_type_names),
            **multiclass_metrics(
                test_output["fault_type_targets"],
                test_output["fault_type_probabilities"],
                fault_type_names,
            ),
        }
    if "telemetry_predictions" in test_output:
        mask = test_output["telemetry_masks"].astype(bool)
        prediction_error = (
            test_output["telemetry_predictions"] - test_output["telemetry_targets"]
        )[mask]
        persistence_error = (
            test_output["telemetry_current"] - test_output["telemetry_targets"]
        )[mask]
        metrics["telemetry_forecast"] = {
            "horizon_minutes": 1,
            "scale": "train-standardized feature units",
            "target_columns": model_config["telemetry_forecast_columns"],
            "observed_values": int(mask.sum()),
            "model_mae": float(np.abs(prediction_error).mean()),
            "model_rmse": float(np.sqrt(np.mean(prediction_error ** 2))),
            "persistence_mae": float(np.abs(persistence_error).mean()),
            "persistence_rmse": float(np.sqrt(np.mean(persistence_error ** 2))),
        }
    if "evidence_probabilities" in test_output:
        metrics["service_evidence"] = {
            "auxiliary_weight": evidence_loss_weight,
            "residual_scale": float(
                model.evidence_residual_scale.detach().cpu().item()
            ),
            "global_branch": classification_metrics(
                test_output["labels"], test_output["global_probabilities"]
            ),
            "evidence_branch": classification_metrics(
                test_output["labels"], test_output["evidence_probabilities"]
            ),
        }
    if "collaboration_attention" in test_output:
        attention = test_output["collaboration_attention"]
        entropy = -(attention * np.log(np.maximum(attention, 1e-12))).sum(axis=1)
        metrics["context_evidence_collaboration"] = {
            "residual_scale": float(
                model.collaboration_residual_scale.detach().cpu().item()
            ),
            "reliability_mean": float(
                test_output["collaboration_reliability"].mean()
            ),
            "reliability_std": float(
                test_output["collaboration_reliability"].std()
            ),
            "attention_entropy_mean": float(entropy.mean()),
            "attention_max_mean": float(attention.max(axis=1).mean()),
        }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    if graph_source.startswith("identity matrix"):
        print("WARNING: no call graph supplied; this run is a temporal-only baseline.")
    return metrics
