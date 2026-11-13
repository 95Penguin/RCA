from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import GraphWindowDataset
from models.instantops import InstantOpsGNNGRU


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
    model: InstantOpsGNNGRU,
    loader: DataLoader,
    device: torch.device,
    failure_loss_fn: nn.Module,
    node_loss_fn: nn.Module,
    node_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    rank_loss_weight: float = 0.0,
) -> tuple[float, dict[str, np.ndarray]]:
    if failure_loss_fn.reduction != "mean" or node_loss_fn.reduction != "mean":
        raise ValueError("epoch aggregation requires mean-reduced losses")
    training = optimizer is not None
    model.train(training)
    failure_numerator = failure_denominator = 0.0
    node_numerator = node_denominator = 0.0
    rank_numerator = rank_denominator = 0.0
    collected: dict[str, list[np.ndarray]] = {
        "labels": [],
        "probabilities": [],
        "node_targets": [],
        "node_scores": [],
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
            rank_valid = node_valid & (labels == 1) & (node_targets.sum(dim=1) > 0)
            if rank_valid.any():
                roots = node_targets[rank_valid]
                roots = roots / roots.sum(dim=1, keepdim=True)
                log_probs = torch.log_softmax(output["node_logits"][rank_valid], dim=1)
                rank_loss = -(roots * log_probs).sum(dim=1).mean()
            else:
                rank_loss = output["node_logits"].sum() * 0
            loss = failure_loss + node_loss_weight * node_loss + rank_loss_weight * rank_loss
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
        probabilities = output["failure_logits"].softmax(dim=-1)[:, 1]
        collected["labels"].append(labels.detach().cpu().numpy())
        collected["probabilities"].append(probabilities.detach().cpu().numpy())
        collected["node_targets"].append(node_targets.detach().cpu().numpy())
        collected["node_scores"].append(output["node_scores"].detach().cpu().numpy())
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    epoch_loss = (
        failure_numerator / max(failure_denominator, 1e-12)
        + node_loss_weight * node_numerator / max(node_denominator, 1)
        + rank_loss_weight * rank_numerator / max(rank_denominator, 1)
    )
    return epoch_loss, arrays
