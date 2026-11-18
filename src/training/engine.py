from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import GraphWindowDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fit_feature_scaler(
    dataset: GraphWindowDataset, train_ends: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on train-only observed values, keeping missing inputs and masks at zero."""
    train_times: set[int] = set()
    for end in train_ends:
        train_times.update(range(end - dataset.history + 1, end + 1))
    train = dataset.features[sorted(train_times)]
    names = {name: index for index, name in enumerate(dataset.feature_columns)}
    mean = np.zeros(len(names), dtype=np.float32)
    scale = np.ones(len(names), dtype=np.float32)
    scaled = dataset.features.copy()
    for name, column in names.items():
        if name.startswith("observed_") or name == "log_observed":
            continue  # observation masks remain binary after scaling
        if name.startswith("metric_"):
            observed = names.get("observed_" + name.removeprefix("metric_"))
        elif name.startswith("log_"):
            observed = names.get("log_observed")
        elif name == "entry_p95_ms":
            observed = names.get("observed_entry_p95")
        elif name == "entry_degradation_score_rolling":
            observed = names.get("observed_entry_baseline")
        else:
            observed = None
        observed_train = train[:, :, observed] > 0 if observed is not None else None
        values = train[:, :, column][observed_train] if observed_train is not None else train[:, :, column].ravel()
        if values.size:
            mean[column] = values.mean(dtype=np.float64)
            std = values.std(dtype=np.float64)
            scale[column] = std if std >= 1e-8 else 1.0
        scaled[:, :, column] = (dataset.features[:, :, column] - mean[column]) / scale[column]
        if observed is not None:
            scaled[:, :, column][dataset.features[:, :, observed] == 0] = 0.0
    dataset.features = scaled
    return mean, scale


def class_pos_weight(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    weight = negatives / positives if positives else 1.0
    return torch.tensor(weight, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    failure_loss_fn: nn.Module,
    node_loss_fn: nn.Module,
    node_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    rank_loss_weight: float = 0.0,
    telemetry_loss_weight: float = 0.0,
    evidence_loss_weight: float = 0.0,
    fault_type_loss_fn: nn.Module | None = None,
    fault_type_loss_weight: float = 0.0,
) -> tuple[float, dict[str, np.ndarray]]:
    if failure_loss_fn.reduction != "mean" or node_loss_fn.reduction != "mean":
        raise ValueError("epoch aggregation requires mean-reduced losses")
    training = optimizer is not None
    model.train(training)
    failure_numerator = failure_denominator = 0.0
    node_numerator = node_denominator = 0.0
    rank_numerator = rank_denominator = 0.0
    telemetry_numerator = telemetry_denominator = 0.0
    evidence_numerator = evidence_denominator = 0.0
    fault_type_numerator = fault_type_denominator = 0.0
    collected: dict[str, list[np.ndarray]] = {
        "labels": [],
        "probabilities": [],
        "node_targets": [],
        "node_scores": [],
        "telemetry_predictions": [],
        "telemetry_targets": [],
        "telemetry_current": [],
        "telemetry_masks": [],
        "global_probabilities": [],
        "evidence_probabilities": [],
        "failure_modality_weights": [],
        "root_modality_weights": [],
        "collaboration_attention": [],
        "collaboration_reliability": [],
        "collaboration_residual": [],
        "fault_type_targets": [],
        "fault_type_probabilities": [],
    }
    for batch in loader:
        features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["failure_target"].to(device)
        node_targets = batch["node_targets"].to(device)
        node_valid = batch.get("node_label_valid", torch.ones_like(labels, dtype=torch.bool)).to(device)
        with torch.set_grad_enabled(training):
            output = model(features, adjacency)
            failure_loss = failure_loss_fn(output["failure_logits"], labels)
            node_loss = (
                node_loss_fn(output["node_logits"][node_valid], node_targets[node_valid])
                if node_valid.any() else output["node_logits"].sum() * 0
            )
            rank_node_logits = output.get("rank_node_logits", output["node_logits"])
            root_event_weights = batch.get(
                "root_event_weight", torch.ones_like(labels, dtype=torch.float32)
            ).to(device)
            rank_valid = node_valid & (labels == 1) & (node_targets.sum(dim=1) > 0)
            if rank_valid.any():
                roots = node_targets[rank_valid]
                roots = roots / roots.sum(dim=1, keepdim=True)
                log_probs = torch.log_softmax(rank_node_logits[rank_valid], dim=1)
                per_window_rank_loss = -(roots * log_probs).sum(dim=1)
                rank_loss = (
                    per_window_rank_loss * root_event_weights[rank_valid]
                ).mean()
            else:
                rank_loss = rank_node_logits.sum() * 0
            telemetry_prediction = output.get("telemetry_prediction")
            if telemetry_prediction is not None:
                if "next_node_features" not in batch:
                    raise ValueError("telemetry forecast requires next_node_features")
                target_indices = tuple(getattr(model, "forecast_target_indices", ()))
                observation_indices = tuple(
                    getattr(model, "forecast_observation_indices", ())
                )
                if (
                    not target_indices
                    or len(target_indices) != telemetry_prediction.size(-1)
                    or len(observation_indices) != len(target_indices)
                ):
                    raise ValueError("invalid telemetry forecast target specification")
                next_features = batch["next_node_features"].to(device)
                telemetry_target = next_features[:, :, list(target_indices)]
                telemetry_mask = torch.ones_like(telemetry_target, dtype=torch.bool)
                for column, observed in enumerate(observation_indices):
                    if observed is not None:
                        telemetry_mask[:, :, column] = next_features[:, :, observed] > 0
                telemetry_elements = int(telemetry_mask.sum().item())
                telemetry_loss = (
                    (telemetry_prediction - telemetry_target).square()[telemetry_mask].mean()
                    if telemetry_elements
                    else telemetry_prediction.sum() * 0
                )
            else:
                if telemetry_loss_weight:
                    raise ValueError(
                        "telemetry_loss_weight is positive but model has no forecast output"
                    )
                telemetry_target = telemetry_mask = None
                telemetry_elements = 0
                telemetry_loss = output["failure_logits"].sum() * 0
            evidence_failure_logits = output.get("evidence_failure_logits")
            if evidence_failure_logits is not None:
                evidence_loss = failure_loss_fn(evidence_failure_logits, labels)
            else:
                if evidence_loss_weight:
                    raise ValueError(
                        "evidence_loss_weight is positive but model has no evidence output"
                    )
                evidence_loss = output["failure_logits"].sum() * 0
            fault_type_logits = output.get("fault_type_logits")
            if fault_type_logits is not None:
                if fault_type_loss_fn is None:
                    raise ValueError("fault-type output requires a configured loss")
                if "fault_type_target" not in batch or "fault_type_valid" not in batch:
                    raise ValueError("fault-type output requires dataset targets and masks")
                fault_type_targets = batch["fault_type_target"].to(device)
                fault_type_valid = batch["fault_type_valid"].to(device).bool()
                fault_type_loss_valid = fault_type_valid
                fault_weights = getattr(fault_type_loss_fn, "weight", None)
                if fault_weights is not None:
                    fault_type_loss_valid = fault_type_loss_valid & (
                        fault_weights[fault_type_targets] > 0
                    )
                fault_type_loss = (
                    fault_type_loss_fn(
                        fault_type_logits[fault_type_loss_valid],
                        fault_type_targets[fault_type_loss_valid],
                    )
                    if fault_type_loss_valid.any()
                    else fault_type_logits.sum() * 0
                )
            else:
                if fault_type_loss_weight:
                    raise ValueError(
                        "fault_type_loss_weight is positive but model has no fault-type output"
                    )
                fault_type_targets = fault_type_valid = None
                fault_type_loss = output["failure_logits"].sum() * 0
            loss = (
                failure_loss
                + node_loss_weight * node_loss
                + rank_loss_weight * rank_loss
                + telemetry_loss_weight * telemetry_loss
                + evidence_loss_weight * evidence_loss
                + fault_type_loss_weight * fault_type_loss
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss; check features and class weights")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        size = labels.size(0)
        # CrossEntropyLoss(mean) divides by target-class weight, not batch size.
        weight = getattr(failure_loss_fn, "weight", None)
        denominator = float(weight[labels].sum().item()) if weight is not None else size
        failure_numerator += failure_loss.item() * denominator
        failure_denominator += denominator
        node_elements = node_targets[node_valid].numel()
        node_numerator += node_loss.item() * node_elements
        node_denominator += node_elements
        rank_examples = int(rank_valid.sum().item())
        rank_numerator += rank_loss.item() * rank_examples
        rank_denominator += rank_examples
        telemetry_numerator += telemetry_loss.item() * telemetry_elements
        telemetry_denominator += telemetry_elements
        evidence_numerator += evidence_loss.item() * denominator
        evidence_denominator += denominator
        if fault_type_logits is not None and fault_type_valid is not None:
            loss_fault_targets = fault_type_targets[fault_type_loss_valid]
            fault_weight = getattr(fault_type_loss_fn, "weight", None)
            fault_denominator = (
                float(fault_weight[loss_fault_targets].sum().item())
                if fault_weight is not None and loss_fault_targets.numel()
                else int(loss_fault_targets.numel())
            )
            fault_type_numerator += fault_type_loss.item() * fault_denominator
            fault_type_denominator += fault_denominator
            valid_fault_targets = fault_type_targets[fault_type_valid]
            if valid_fault_targets.numel():
                collected["fault_type_targets"].append(
                    valid_fault_targets.detach().cpu().numpy()
                )
                collected["fault_type_probabilities"].append(
                    fault_type_logits[fault_type_valid]
                    .softmax(dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
        probabilities = output["failure_logits"].softmax(dim=-1)[:, 1]
        collected["labels"].append(labels.detach().cpu().numpy())
        collected["probabilities"].append(probabilities.detach().cpu().numpy())
        if "global_failure_logits" in output:
            collected["global_probabilities"].append(
                output["global_failure_logits"].softmax(dim=-1)[:, 1]
                .detach()
                .cpu()
                .numpy()
            )
        if evidence_failure_logits is not None:
            collected["evidence_probabilities"].append(
                evidence_failure_logits.softmax(dim=-1)[:, 1]
                .detach()
                .cpu()
                .numpy()
            )
        collected["node_targets"].append(node_targets.detach().cpu().numpy())
        collected["node_scores"].append(output["node_scores"].detach().cpu().numpy())
        for key in ("failure_modality_weights", "root_modality_weights"):
            if key in output:
                collected[key].append(output[key].detach().cpu().numpy())
        for key in (
            "collaboration_attention",
            "collaboration_reliability",
            "collaboration_residual",
        ):
            if key in output:
                collected[key].append(output[key].detach().cpu().numpy())
        if telemetry_prediction is not None and telemetry_target is not None:
            target_indices = tuple(model.forecast_target_indices)
            collected["telemetry_predictions"].append(
                telemetry_prediction.detach().cpu().numpy()
            )
            collected["telemetry_targets"].append(
                telemetry_target.detach().cpu().numpy()
            )
            collected["telemetry_current"].append(
                features[:, -1, :, list(target_indices)].detach().cpu().numpy()
            )
            collected["telemetry_masks"].append(
                telemetry_mask.detach().cpu().numpy()
            )
    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in collected.items()
        if value
    }
    epoch_loss = (
        failure_numerator / max(failure_denominator, 1e-12)
        + node_loss_weight * node_numerator / max(node_denominator, 1)
        + rank_loss_weight * rank_numerator / max(rank_denominator, 1)
        + telemetry_loss_weight
        * telemetry_numerator
        / max(telemetry_denominator, 1)
        + evidence_loss_weight
        * evidence_numerator
        / max(evidence_denominator, 1e-12)
        + fault_type_loss_weight
        * fault_type_numerator
        / max(fault_type_denominator, 1)
    )
    return epoch_loss, arrays
