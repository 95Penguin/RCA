"""Eadro architecture adapted to future failure prediction labels."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .common import CausalConv1d, normalize_feature_groups, validate_graph_inputs


class _SimpleTemporalBranch(nn.Module):
    """Eadro-style causal convolution stack with final-step readout."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(input_dim, hidden_dim, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.network(sequence.transpose(1, 2))[:, :, -1]


class _DenseGATv2(nn.Module):
    """Dependency-masked, single-head dense equivalent of GATv2 propagation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.left = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.right = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.left.weight)
        nn.init.xavier_uniform_(self.right.weight)
        nn.init.xavier_uniform_(self.message.weight)
        nn.init.normal_(self.attention, std=0.02)

    def forward(self, node_state: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        services = node_state.size(1)
        eye = torch.eye(services, device=adjacency.device, dtype=adjacency.dtype)
        mask = torch.maximum(adjacency, eye.unsqueeze(0)) > 0
        pair_state = F.leaky_relu(
            self.left(node_state).unsqueeze(2) + self.right(node_state).unsqueeze(1),
            negative_slope=0.2,
        )
        logits = torch.einsum("bijh,h->bij", pair_state, self.attention)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = self.dropout(torch.softmax(logits, dim=-1))
        return F.elu(torch.bmm(weights, self.message(node_state)))


class EadroAdapted(nn.Module):
    """Eadro's reactive architecture adapted to future prediction labels."""

    def __init__(
        self,
        feature_dim: int,
        num_services: int,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        graph_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.groups = normalize_feature_groups(feature_dim, feature_groups)
        self.branches = nn.ModuleList(
            _SimpleTemporalBranch(len(group), hidden_dim, dropout) for group in self.groups
        )
        self.fusion = nn.Linear(hidden_dim * len(self.groups), hidden_dim * 2)
        self.graph_layers = nn.ModuleList(
            _DenseGATv2(hidden_dim, dropout) for _ in range(graph_layers)
        )
        self.graph_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(graph_layers)
        )
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.failure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_services)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        validate_graph_inputs(node_features, adjacency)
        batch, steps, services, _ = node_features.shape
        flattened = node_features.permute(0, 2, 1, 3).reshape(batch * services, steps, -1)
        modalities = [
            branch(flattened[:, :, group])
            for branch, group in zip(self.branches, self.groups)
        ]
        node_state = F.glu(self.fusion(torch.cat(modalities, dim=-1)), dim=-1)
        node_state = node_state.reshape(batch, services, -1)
        graph = adjacency[:, -1]
        for layer, norm in zip(self.graph_layers, self.graph_norms):
            node_state = norm(node_state + self.dropout(layer(node_state, graph)))
        pooling_weights = torch.softmax(self.pool_gate(node_state).squeeze(-1), dim=1)
        graph_state = (pooling_weights.unsqueeze(-1) * node_state).sum(dim=1)
        node_logits = self.location_head(graph_state)
        return {
            "failure_logits": self.failure_head(graph_state),
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": node_state,
        }


__all__ = ["EadroAdapted"]
