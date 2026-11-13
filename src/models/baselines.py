"""Comparable non-graph baselines for the shared next-minute task."""

from __future__ import annotations

import torch
from torch import nn

from .common import CausalConv1d


class _ServiceHeads(nn.Module):
    """Shared application and service heads used by simple baselines."""

    def __init__(self, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.failure_head = nn.Linear(hidden_dim, num_classes)
        self.node_score_head = nn.Linear(hidden_dim, 1)

    def output(self, final_nodes: torch.Tensor) -> dict[str, torch.Tensor]:
        node_logits = self.node_score_head(final_nodes).squeeze(-1)
        app_state = final_nodes.mean(dim=1)
        return {
            "failure_logits": self.failure_head(app_state),
            "node_logits": node_logits,
            "node_scores": torch.sigmoid(node_logits),
            "node_embeddings": final_nodes,
        }


class LastValueMLP(nn.Module):
    """Use only the latest observed minute; no temporal or graph modeling."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = _ServiceHeads(hidden_dim, num_classes)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del adjacency
        if node_features.ndim != 4:
            raise ValueError("node_features must have shape [B, T, N, F]")
        final_nodes = self.encoder(node_features[:, -1])
        return self.heads.output(final_nodes)


class ServiceTCN(nn.Module):
    """Per-service causal TCN; captures time but deliberately ignores the graph."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            CausalConv1d(feature_dim, hidden_dim, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, dilation=2),
            nn.ReLU(),
        )
        self.heads = _ServiceHeads(hidden_dim, num_classes)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del adjacency
        if node_features.ndim != 4:
            raise ValueError("node_features must have shape [B, T, N, F]")
        batch, steps, services, feature_dim = node_features.shape
        sequences = (
            node_features.permute(0, 2, 3, 1)
            .reshape(batch * services, feature_dim, steps)
        )
        encoded = self.temporal(sequences)
        final_nodes = encoded[:, :, -1].reshape(batch, services, -1)
        return self.heads.output(final_nodes)
