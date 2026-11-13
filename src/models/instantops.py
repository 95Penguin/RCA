"""InstantOps-style GCN-GRU failure prediction model."""

from __future__ import annotations

import torch
from torch import nn

from .common import validate_graph_inputs


class GraphConvolution(nn.Module):
    """One normalized graph-convolution layer for batched service graphs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Args: node_features [..., nodes, features], adjacency [..., nodes, nodes]."""
        nodes = adjacency.size(-1)
        eye = torch.eye(nodes, device=adjacency.device, dtype=adjacency.dtype)
        adjacency = adjacency + eye
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)
        return self.linear(torch.matmul(normalized, node_features))


class InstantOpsGNNGRU(nn.Module):
    """Spatial graph encoder followed by a per-service temporal GRU.

    The prediction horizon is defined by the dataset labels, so this model can
    be reused for one-, three-, or five-minute targets without architecture
    changes.
    """

    def __init__(
        self,
        feature_dim: int,
        graph_hidden: int = 64,
        temporal_hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
        gcn_layers: int = 2,
        joint_service_pooling: bool = False,
    ) -> None:
        super().__init__()
        if gcn_layers < 0:
            raise ValueError("gcn_layers must be nonnegative")
        self.graph_convs = nn.ModuleList(
            [
                GraphConvolution(feature_dim if index == 0 else graph_hidden, graph_hidden)
                for index in range(gcn_layers)
            ]
        )
        self.graph_norm = nn.LayerNorm(graph_hidden) if gcn_layers else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.temporal = nn.GRU(
            graph_hidden if gcn_layers else feature_dim,
            temporal_hidden,
            batch_first=True,
        )
        self.failure_head = nn.Linear(temporal_hidden, num_classes)
        self.node_score_head = nn.Linear(temporal_hidden, 1)
        self.joint_service_pooling = joint_service_pooling

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        validate_graph_inputs(node_features, adjacency)
        batch, steps, services, _ = node_features.shape
        spatial = node_features.reshape(batch * steps, services, -1)
        graphs = adjacency.reshape(batch * steps, services, services)
        for convolution in self.graph_convs:
            spatial = self.dropout(torch.relu(self.graph_norm(convolution(spatial, graphs))))
        spatial = spatial.reshape(batch, steps, services, -1)

        per_service = spatial.permute(0, 2, 1, 3).reshape(batch * services, steps, -1)
        temporal, _ = self.temporal(per_service)
        final_nodes = temporal[:, -1].reshape(batch, services, -1)

        node_logits = self.node_score_head(final_nodes).squeeze(-1)
        app_state = final_nodes.mean(dim=1)
        if self.joint_service_pooling:
            service_weights = torch.softmax(node_logits, dim=1)
            attended = (service_weights.unsqueeze(-1) * final_nodes).sum(dim=1)
            app_state = (app_state + attended) * 0.5
        return {
            "failure_logits": self.failure_head(app_state),
            "node_logits": node_logits,
            "node_scores": torch.sigmoid(node_logits),
            "node_embeddings": final_nodes,
        }


__all__ = ["GraphConvolution", "InstantOpsGNNGRU"]
