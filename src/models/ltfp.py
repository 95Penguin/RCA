"""Task-aligned LTFP baseline for future failure prediction and RCA."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .common import CausalConv1d, normalize_feature_groups, validate_graph_inputs


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(hidden_dim, hidden_dim, dilation)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.dropout(F.relu(self.norm(self.conv(values))))
        return values + update


class _TemporalAttentionBranch(nn.Module):
    """LTFP-style TCN branch with latest-state and attention readout."""

    def __init__(
        self, input_dim: int, hidden_dim: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _ResidualTemporalBlock(hidden_dim, 2**index, dropout)
            for index in range(layers)
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        values = self.project(sequence).transpose(1, 2)
        for block in self.blocks:
            values = block(values)
        values = values.transpose(1, 2)
        weights = torch.softmax(self.attention(values).squeeze(-1), dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return self.readout(torch.cat((values[:, -1], pooled), dim=-1))


class _WeightedGraphAttention(nn.Module):
    """Dense edge-weight-aware graph attention used by the LTFP baseline."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(hidden_dim))
        self.target_attention = nn.Parameter(torch.empty(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.source_attention, std=0.02)
        nn.init.normal_(self.target_attention, std=0.02)

    def forward(self, node_state: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        services = node_state.size(1)
        eye = torch.eye(services, device=adjacency.device, dtype=adjacency.dtype)
        weights = torch.maximum(adjacency, eye.unsqueeze(0))
        transformed = self.project(node_state)
        source = torch.einsum("bnh,h->bn", transformed, self.source_attention)
        target = torch.einsum("bnh,h->bn", transformed, self.target_attention)
        logits = F.leaky_relu(
            source.unsqueeze(2) + target.unsqueeze(1) + torch.log(weights + 1e-12),
            negative_slope=0.2,
        )
        logits = logits.masked_fill(weights <= 0, float("-inf"))
        attention = self.dropout(torch.softmax(logits, dim=-1))
        return F.elu(torch.bmm(attention, transformed))


class LTFPBaseline(nn.Module):
    """Single-horizon LTFP architecture under the shared prediction protocol."""

    def __init__(
        self,
        feature_dim: int,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.groups = normalize_feature_groups(feature_dim, feature_groups)
        self.branches = nn.ModuleList(
            _TemporalAttentionBranch(len(group), hidden_dim, temporal_layers, dropout)
            for group in self.groups
        )
        self.modality_gate = nn.Linear(hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_layers = nn.ModuleList(
            _WeightedGraphAttention(hidden_dim, dropout) for _ in range(graph_layers)
        )
        self.graph_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(graph_layers)
        )
        self.graph_dropout = nn.Dropout(dropout)
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.failure_head = nn.Linear(hidden_dim, num_classes)
        self.root_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        validate_graph_inputs(node_features, adjacency)
        batch, steps, services, _ = node_features.shape
        flattened = node_features.permute(0, 2, 1, 3).reshape(batch * services, steps, -1)
        states = torch.stack(
            [
                branch(flattened[:, :, group])
                for branch, group in zip(self.branches, self.groups)
            ],
            dim=1,
        )
        modality_weights = torch.softmax(self.modality_gate(states).squeeze(-1), dim=1)
        weighted = (states * modality_weights.unsqueeze(-1)).sum(dim=1)
        maximum = states.max(dim=1).values
        node_state = self.fusion(torch.cat((weighted, maximum), dim=-1))
        node_state = node_state.reshape(batch, services, -1)

        graph = adjacency[:, -1]
        for layer, norm in zip(self.graph_layers, self.graph_norms):
            node_state = norm(node_state + self.graph_dropout(layer(node_state, graph)))
        graph_state = self.pool(
            torch.cat((node_state.mean(dim=1), node_state.max(dim=1).values), dim=-1)
        )
        context = graph_state.unsqueeze(1).expand(-1, services, -1)
        node_logits = self.root_head(
            torch.cat((node_state, context, node_state * context), dim=-1)
        ).squeeze(-1)
        return {
            "failure_logits": self.failure_head(graph_state),
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": node_state,
        }


__all__ = ["LTFPBaseline"]
