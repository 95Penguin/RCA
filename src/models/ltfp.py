"""LTFP backbone and historical feasibility variants.

The active coarse/fine research models live in :mod:`models.coarse_fine`.
Compatibility exports remain at the bottom so older experiment code and
checkpoints can keep importing them from :mod:`models.ltfp`.
"""

from __future__ import annotations

import copy
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

    def _encode_local(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Encode each service history before cross-service propagation."""
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
        local_state = self.fusion(torch.cat((weighted, maximum), dim=-1))
        return local_state.reshape(batch, services, -1)

    def _propagate_state(
        self,
        local_state: torch.Tensor,
        adjacency: torch.Tensor,
        graph_layers: nn.ModuleList | None = None,
        graph_norms: nn.ModuleList | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate one local state through a selected graph expert."""
        layers = self.graph_layers if graph_layers is None else graph_layers
        norms = self.graph_norms if graph_norms is None else graph_norms
        node_state = local_state
        graph = adjacency[:, -1]
        for layer, norm in zip(layers, norms):
            node_state = norm(node_state + self.graph_dropout(layer(node_state, graph)))
        graph_state = self.pool(
            torch.cat((node_state.mean(dim=1), node_state.max(dim=1).values), dim=-1)
        )
        return node_state, graph_state

    def _encode_paths(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local, propagated service, and application states."""
        local_state = self._encode_local(node_features, adjacency)
        node_state, graph_state = self._propagate_state(local_state, adjacency)
        return local_state, node_state, graph_state

    def _encode(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return propagated service and application states for legacy heads."""
        _, node_state, graph_state = self._encode_paths(node_features, adjacency)
        return node_state, graph_state

    def _output(
        self,
        node_state: torch.Tensor,
        graph_state: torch.Tensor,
        failure_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the shared RCA head and package the model output contract."""
        services = node_state.size(1)
        context = graph_state.unsqueeze(1).expand(-1, services, -1)
        node_logits = self.root_head(
            torch.cat((node_state, context, node_state * context), dim=-1)
        ).squeeze(-1)
        return {
            "failure_logits": failure_logits,
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": node_state,
        }

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        node_state, graph_state = self._encode(node_features, adjacency)
        return self._output(
            node_state,
            graph_state,
            self.failure_head(graph_state),
        )


class LTFPFaultTypeAux(LTFPBaseline):
    """LTFP with a positive-window fault-type auxiliary classifier.

    The auxiliary head reads the same application representation as the
    failure head. Its loss is masked by the training engine on windows whose
    future application-failure label is positive, so normal windows are not
    assigned the injected case's fault type.
    """

    def __init__(
        self,
        feature_dim: int,
        num_fault_types: int,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        if num_fault_types < 2:
            raise ValueError("fault-type auxiliary head needs at least two classes")
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.fault_type_head = nn.Linear(hidden_dim, num_fault_types)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        node_state, graph_state = self._encode(node_features, adjacency)
        output = self._output(
            node_state,
            graph_state,
            self.failure_head(graph_state),
        )
        output["fault_type_logits"] = self.fault_type_head(graph_state)
        return output


class _TaskSpecificModalityRouter(nn.Module):
    """Select modality evidence using both branch state and incident context."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = states.mean(dim=1, keepdim=True).expand_as(states)
        weights = torch.softmax(
            self.score(torch.cat((states, context), dim=-1)).squeeze(-1), dim=1
        )
        routed = states * weights.unsqueeze(-1)
        summary = torch.cat((routed.sum(dim=1), routed.max(dim=1).values), dim=-1)
        return summary, weights


class LTFPDualRouter(LTFPBaseline):
    """Use incident-adaptive, task-specific modality routes.

    Failure prediction and RCA share modality encoders, fusion weights, and
    graph layers. They differ only in the learned evidence routers, isolating
    the routing hypothesis from a wholesale backbone change.
    """

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
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        del self.modality_gate
        self.failure_router = _TaskSpecificModalityRouter(hidden_dim)
        self.root_router = _TaskSpecificModalityRouter(hidden_dim)

    def _propagate(
        self, local_state: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_state = local_state
        graph = adjacency[:, -1]
        for layer, norm in zip(self.graph_layers, self.graph_norms):
            node_state = norm(node_state + self.graph_dropout(layer(node_state, graph)))
        graph_state = self.pool(
            torch.cat((node_state.mean(dim=1), node_state.max(dim=1).values), dim=-1)
        )
        return node_state, graph_state

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        validate_graph_inputs(node_features, adjacency)
        batch, steps, services, _ = node_features.shape
        flattened = node_features.permute(0, 2, 1, 3).reshape(
            batch * services, steps, -1
        )
        states = torch.stack(
            [
                branch(flattened[:, :, group])
                for branch, group in zip(self.branches, self.groups)
            ],
            dim=1,
        )
        failure_summary, failure_weights = self.failure_router(states)
        root_summary, root_weights = self.root_router(states)
        failure_local = self.fusion(failure_summary).reshape(batch, services, -1)
        root_local = self.fusion(root_summary).reshape(batch, services, -1)
        failure_nodes, failure_graph = self._propagate(failure_local, adjacency)
        root_nodes, root_graph = self._propagate(root_local, adjacency)

        context = root_graph.unsqueeze(1).expand(-1, services, -1)
        node_logits = self.root_head(
            torch.cat((root_nodes, context, root_nodes * context), dim=-1)
        ).squeeze(-1)
        return {
            "failure_logits": self.failure_head(failure_graph),
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": root_nodes,
            "failure_node_embeddings": failure_nodes,
            "failure_modality_weights": failure_weights.reshape(batch, services, -1),
            "root_modality_weights": root_weights.reshape(batch, services, -1),
        }


class _LTFPTwoGraphExperts(LTFPBaseline):
    """Common capacity-matched backbone for the task-graph ablation."""

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
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.secondary_graph_layers = copy.deepcopy(self.graph_layers)
        self.secondary_graph_norms = copy.deepcopy(self.graph_norms)

    def _expert_states(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_state = self._encode_local(node_features, adjacency)
        first_nodes, first_graph = self._propagate_state(local_state, adjacency)
        second_nodes, second_graph = self._propagate_state(
            local_state,
            adjacency,
            self.secondary_graph_layers,
            self.secondary_graph_norms,
        )
        return first_nodes, first_graph, second_nodes, second_graph


class LTFPTaskIndependentGraph(_LTFPTwoGraphExperts):
    """Assign independent graph experts to failure prediction and RCA."""

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        failure_nodes, failure_graph, root_nodes, root_graph = self._expert_states(
            node_features, adjacency
        )
        context = root_graph.unsqueeze(1).expand(-1, root_nodes.size(1), -1)
        node_logits = self.root_head(
            torch.cat((root_nodes, context, root_nodes * context), dim=-1)
        ).squeeze(-1)
        return {
            "failure_logits": self.failure_head(failure_graph),
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": root_nodes,
            "failure_node_embeddings": failure_nodes,
        }


class LTFPSharedGraphExperts(_LTFPTwoGraphExperts):
    """Capacity control: both tasks consume the same two-expert average."""

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        first_nodes, _, second_nodes, _ = self._expert_states(
            node_features, adjacency
        )
        shared_nodes = 0.5 * (first_nodes + second_nodes)
        shared_graph = self.pool(
            torch.cat(
                (shared_nodes.mean(dim=1), shared_nodes.max(dim=1).values), dim=-1
            )
        )
        return self._output(
            shared_nodes,
            shared_graph,
            self.failure_head(shared_graph),
        )


class LTFPNoGraph(LTFPBaseline):
    """Temporal-only LTFP control with no cross-service message passing."""

    def __init__(
        self,
        feature_dim: int,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=0,
            dropout=dropout,
            num_classes=num_classes,
        )


class LTFPTransitionResidual(LTFPBaseline):
    """Decompose future risk into a latest-state prior and temporal residual.

    The state branch is structurally restricted to the latest observed minute.
    The residual branch uses LTFP's complete causal temporal representation.
    Their class logits are added, while RCA keeps the original LTFP service
    representation and head.  This isolates the proposed forecasting change
    from the localization architecture.
    """

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
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.current_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.current_failure_head = nn.Linear(hidden_dim, num_classes)

        # Start as a latest-state model and let optimization add only the
        # temporal correction supported by the data.
        nn.init.zeros_(self.failure_head.weight)
        nn.init.zeros_(self.failure_head.bias)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        node_state, graph_state = self._encode(node_features, adjacency)
        current_nodes = self.current_encoder(node_features[:, -1])
        current_state = self.current_pool(
            torch.cat(
                (current_nodes.mean(dim=1), current_nodes.max(dim=1).values),
                dim=-1,
            )
        )
        current_logits = self.current_failure_head(current_state)
        residual_logits = self.failure_head(graph_state)
        output = self._output(
            node_state,
            graph_state,
            current_logits + residual_logits,
        )
        output.update({
            "current_failure_logits": current_logits,
            "transition_residual_logits": residual_logits,
        })
        return output


class LTFPRootCauseConsistent(LTFPBaseline):
    """Derive failure prediction and RCA from one service-risk representation.

    Each service risk combines its pre-propagation local evidence, its
    graph-propagated state, and their difference.  A size-normalized log-sum-exp
    aggregates those same logits into the application failure logit, preventing
    the prediction and localization heads from relying on contradictory evidence.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=2,
        )
        self.temperature = float(temperature)
        self.service_risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # These independent baseline heads are deliberately unused: both tasks
        # must pass through service_risk_head for the consistency claim to hold.
        del self.failure_head
        del self.root_head

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, _ = self._encode_paths(node_features, adjacency)
        node_logits = self.service_risk_head(
            torch.cat((local_state, node_state, node_state - local_state), dim=-1)
        ).squeeze(-1)
        services = node_logits.size(1)
        system_logit = self.temperature * (
            torch.logsumexp(node_logits / self.temperature, dim=1)
            - torch.log(node_logits.new_tensor(float(services)))
        )
        failure_logits = torch.stack((torch.zeros_like(system_logit), system_logit), dim=1)
        return {
            "failure_logits": failure_logits,
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": node_state,
            "local_node_embeddings": local_state,
            "system_risk_logit": system_logit,
        }


class LTFPForecastAux(LTFPBaseline):
    """Forecast the next service state before failure prediction and RCA.

    The decoder is supervised with dense next-minute telemetry.  Its predicted
    service state is projected back into the graph representation used by both
    downstream heads, so the auxiliary task directly shapes the decision path.
    """

    def __init__(
        self,
        feature_dim: int,
        forecast_dim: int,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        if forecast_dim < 1:
            raise ValueError("forecast_dim must be positive")
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.telemetry_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_dim),
        )
        self.future_projection = nn.Linear(forecast_dim, hidden_dim)
        self.future_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        node_state, _ = self._encode(node_features, adjacency)
        telemetry_prediction = self.telemetry_decoder(node_state)
        future_node_state = self.future_norm(
            node_state + self.future_projection(telemetry_prediction)
        )
        future_graph_state = self.pool(
            torch.cat(
                (
                    future_node_state.mean(dim=1),
                    future_node_state.max(dim=1).values,
                ),
                dim=-1,
            )
        )
        output = self._output(
            future_node_state,
            future_graph_state,
            self.failure_head(future_graph_state),
        )
        output["telemetry_prediction"] = telemetry_prediction
        return output


def __getattr__(name: str) -> type[nn.Module]:
    """Resolve moved coarse/fine models without creating a circular import."""
    if name in {"LTFPNullAnchoredEvidence", "LTFPResidualEvidence"}:
        # New code should import these from ``models.coarse_fine``.  Loading
        # them lazily keeps the former ``models.ltfp`` path working for older
        # scripts while allowing coarse_fine itself to import LTFPBaseline.
        from .coarse_fine import LTFPNullAnchoredEvidence, LTFPResidualEvidence

        return {
            "LTFPNullAnchoredEvidence": LTFPNullAnchoredEvidence,
            "LTFPResidualEvidence": LTFPResidualEvidence,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LTFPBaseline",
    "LTFPDualRouter",
    "LTFPForecastAux",
    "LTFPNoGraph",
    "LTFPNullAnchoredEvidence",
    "LTFPResidualEvidence",
    "LTFPRootCauseConsistent",
    "LTFPSharedGraphExperts",
    "LTFPTaskIndependentGraph",
    "LTFPTransitionResidual",
]
