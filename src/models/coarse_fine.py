"""Coarse system risk and fine-grained service evidence models.

This module contains only the models used to test the current research
hypothesis.  Stable LTFP encoding remains in :mod:`models.ltfp`; historical
ablation variants remain separate from the coarse/fine evidence flow here.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn

from .common import validate_graph_inputs
from .ltfp import LTFPBaseline, _TemporalAttentionBranch, _WeightedGraphAttention


class _RootModalityRouter(nn.Module):
    """Build a task-specific service summary from detached modality states."""

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


def _service_evidence(
    local_state: torch.Tensor,
    node_state: torch.Tensor,
    graph_state: torch.Tensor,
    service_head: nn.Module,
    no_failure_head: nn.Module,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-service logits, aggregate energy, null logit and binary logits."""
    service_logits = service_head(
        torch.cat((local_state, node_state, node_state - local_state), dim=-1)
    ).squeeze(-1)
    service_energy, no_failure_logit, evidence_failure_logits = (
        _aggregate_service_evidence(
            service_logits,
            graph_state,
            no_failure_head,
            temperature,
        )
    )
    return (
        service_logits,
        service_energy,
        no_failure_logit,
        evidence_failure_logits,
    )


def _aggregate_service_evidence(
    service_logits: torch.Tensor,
    graph_state: torch.Tensor,
    no_failure_head: nn.Module,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate service anomaly logits into a normalized system-level energy."""
    services = service_logits.size(1)
    service_energy = temperature * (
        torch.logsumexp(service_logits / temperature, dim=1)
        - torch.log(service_logits.new_tensor(float(services)))
    )
    no_failure_logit = no_failure_head(graph_state).squeeze(-1)
    evidence_failure_logits = torch.stack(
        (no_failure_logit, service_energy), dim=1
    )
    return service_energy, no_failure_logit, evidence_failure_logits


class LTFPNullAnchoredEvidence(LTFPBaseline):
    """Use one null-anchored service evidence space for both tasks.

    This is the intentionally restrictive ablation that removes LTFP's global
    risk head.  It is retained as a documented negative result, not the current
    recommended architecture.
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
        self.service_evidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.no_failure_head = nn.Linear(hidden_dim, 1)
        del self.failure_head
        del self.root_head

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, graph_state = self._encode_paths(
            node_features, adjacency
        )
        service_logits, service_energy, no_failure_logit, failure_logits = (
            _service_evidence(
                local_state,
                node_state,
                graph_state,
                self.service_evidence_head,
                self.no_failure_head,
                self.temperature,
            )
        )
        return {
            "failure_logits": failure_logits,
            "node_logits": service_logits,
            "node_scores": torch.softmax(service_logits, dim=-1),
            "node_embeddings": node_state,
            "local_node_embeddings": local_state,
            "service_evidence_logits": service_logits,
            "service_evidence_energy": service_energy,
            "no_failure_logit": no_failure_logit,
        }


class LTFPResidualEvidence(LTFPBaseline):
    """Retain coarse global risk and add fine service evidence as a residual.

    Gradient paths in the current implementation are important:

    * failure loss trains the shared LTFP encoder and global head;
    * evidence loss trains the shared encoder and service evidence branch;
    * root loss also trains the shared encoder and service evidence branch;
    * the learned scalar controls only the forward residual.  It does not
      prevent root gradients from changing the shared coarse representation.

    The last point is the main architectural issue exposed by the current OB/TT
    feasibility results and is documented explicitly for the next redesign.
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
        self.service_evidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.no_failure_head = nn.Linear(hidden_dim, 1)
        self.evidence_residual_scale = nn.Parameter(torch.zeros(()))
        del self.root_head

    def _coarse_fine_output(
        self,
        local_state: torch.Tensor,
        node_state: torch.Tensor,
        graph_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Combine already encoded coarse and fine representations."""
        global_failure_logits = self.failure_head(graph_state)
        (
            service_logits,
            service_energy,
            no_failure_logit,
            evidence_failure_logits,
        ) = _service_evidence(
            local_state,
            node_state,
            graph_state,
            self.service_evidence_head,
            self.no_failure_head,
            self.temperature,
        )
        evidence_log_odds = service_energy - no_failure_logit
        failure_logits = global_failure_logits.clone()
        failure_logits[:, 1] = (
            failure_logits[:, 1]
            + self.evidence_residual_scale * evidence_log_odds
        )
        return {
            "failure_logits": failure_logits,
            "global_failure_logits": global_failure_logits,
            "evidence_failure_logits": evidence_failure_logits,
            "node_logits": service_logits,
            "node_scores": torch.softmax(service_logits, dim=-1),
            "node_embeddings": node_state,
            "local_node_embeddings": local_state,
            "service_evidence_logits": service_logits,
            "service_evidence_energy": service_energy,
            "no_failure_logit": no_failure_logit,
            "evidence_residual_scale": self.evidence_residual_scale,
        }

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, graph_state = self._encode_paths(
            node_features, adjacency
        )
        return self._coarse_fine_output(local_state, node_state, graph_state)


class LTFPProtectedRootEvidence(LTFPResidualEvidence):
    """Protect the shared backbone from the root-ranking gradient only.

    The forward prediction, evaluated RCA scores and parameter count are the
    same as :class:`LTFPResidualEvidence`.  A second service-head evaluation on
    detached representations is exposed only to the training engine's root
    ranking loss.  Root supervision can therefore update the service evidence
    head, but not the temporal encoder, modality fusion or graph layers.

    This is a controlled failure-mechanism test rather than the final model.
    """

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, graph_state = self._encode_paths(
            node_features, adjacency
        )
        output = self._coarse_fine_output(local_state, node_state, graph_state)
        protected_features = torch.cat(
            (
                local_state.detach(),
                node_state.detach(),
                (node_state - local_state).detach(),
            ),
            dim=-1,
        )
        output["rank_node_logits"] = self.service_evidence_head(
            protected_features
        ).squeeze(-1)
        return output


class LTFPLocalAdapterEvidence(LTFPBaseline):
    """Keep global prediction protected while making local evidence adaptable.

    A service adapter is shared by two semantically distinct fine-grained heads:

    * ``anomaly_head`` supplies predictive service evidence;
    * ``root_responsibility_head`` supplies RCA rankings.

    The root-ranking path recomputes the local adapter on detached backbone
    features.  Root supervision can therefore shape the local adapter and root
    head, indirectly changing predictive local evidence, without modifying the
    temporal/GAT backbone used by the coarse global branch.
    """

    detach_fine_backbone = False

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
        self.service_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.anomaly_head = nn.Linear(hidden_dim, 1)
        self.root_responsibility_head = nn.Linear(hidden_dim, 1)
        self.no_failure_head = nn.Linear(hidden_dim, 1)
        self.evidence_residual_scale = nn.Parameter(torch.zeros(()))
        del self.root_head

    @staticmethod
    def _service_features(
        local_state: torch.Tensor, node_state: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (local_state, node_state, node_state - local_state), dim=-1
        )

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, graph_state = self._encode_paths(
            node_features, adjacency
        )
        service_features = self._service_features(local_state, node_state)
        fine_input = (
            service_features.detach()
            if self.detach_fine_backbone
            else service_features
        )
        fine_graph_state = (
            graph_state.detach() if self.detach_fine_backbone else graph_state
        )
        fine_state = self.service_adapter(fine_input)
        anomaly_logits = self.anomaly_head(fine_state).squeeze(-1)
        root_logits = self.root_responsibility_head(fine_state).squeeze(-1)
        service_energy, no_failure_logit, evidence_failure_logits = (
            _aggregate_service_evidence(
                anomaly_logits,
                fine_graph_state,
                self.no_failure_head,
                self.temperature,
            )
        )

        global_failure_logits = self.failure_head(graph_state)
        evidence_log_odds = service_energy - no_failure_logit
        failure_logits = global_failure_logits.clone()
        failure_logits[:, 1] = (
            failure_logits[:, 1]
            + self.evidence_residual_scale * evidence_log_odds
        )

        if self.detach_fine_backbone:
            rank_node_logits = root_logits
        else:
            protected_fine_state = self.service_adapter(service_features.detach())
            rank_node_logits = self.root_responsibility_head(
                protected_fine_state
            ).squeeze(-1)
        return {
            "failure_logits": failure_logits,
            "global_failure_logits": global_failure_logits,
            "evidence_failure_logits": evidence_failure_logits,
            "node_logits": root_logits,
            "rank_node_logits": rank_node_logits,
            "node_scores": torch.softmax(root_logits, dim=-1),
            "node_embeddings": node_state,
            "local_node_embeddings": local_state,
            "fine_node_embeddings": fine_state,
            "service_anomaly_logits": anomaly_logits,
            "service_evidence_logits": anomaly_logits,
            "service_evidence_energy": service_energy,
            "no_failure_logit": no_failure_logit,
            "evidence_residual_scale": self.evidence_residual_scale,
        }


class LTFPIsolatedLocalAdapterEvidence(LTFPLocalAdapterEvidence):
    """Detach the complete fine branch while retaining coarse-to-fine fusion.

    This capacity-matched control protects the temporal/GAT backbone from the
    evidence, residual and root-ranking paths.  All fine losses still train the
    service adapter and its two heads; the coarse backbone receives gradients
    only through the system failure logits.
    """

    detach_fine_backbone = True


class LTFPProtectedRootSidecar(LTFPBaseline):
    """Preserve the LTFP prediction path and learn RCA in a detached sidecar.

    Failure prediction uses the original modality gate, fusion, graph layers,
    pooling and failure head without any RCA gradient. RCA reuses detached
    temporal modality states, then learns its own modality router, fusion and
    graph propagation.
    """

    root_sidecar_parameter_prefixes = (
        "root_router.",
        "root_fusion.",
        "root_graph_layers.",
        "root_graph_norms.",
        "root_pool.",
        "root_head.",
    )

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
        # Extra sidecar initialization must not perturb the random sequence
        # seen by the unchanged LTFP path in a seed-matched control run.
        rng_state = torch.random.get_rng_state()
        self.root_router = _RootModalityRouter(hidden_dim)
        self.root_fusion = copy.deepcopy(self.fusion)
        self.root_graph_layers = copy.deepcopy(self.graph_layers)
        self.root_graph_norms = copy.deepcopy(self.graph_norms)
        self.root_graph_dropout = nn.Dropout(0.0)
        self.root_pool = copy.deepcopy(self.pool)
        for module in (
            self.root_fusion,
            self.root_graph_layers,
            self.root_pool,
        ):
            for child in module.modules():
                if isinstance(child, nn.Dropout):
                    child.p = 0.0
        torch.random.set_rng_state(rng_state)

    def _root_propagate(
        self, local_state: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_state = local_state
        graph = adjacency[:, -1]
        for layer, norm in zip(self.root_graph_layers, self.root_graph_norms):
            node_state = norm(
                node_state + self.root_graph_dropout(layer(node_state, graph))
            )
        graph_state = self.root_pool(
            torch.cat(
                (node_state.mean(dim=1), node_state.max(dim=1).values), dim=-1
            )
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
        modality_states = torch.stack(
            [
                branch(flattened[:, :, group])
                for branch, group in zip(self.branches, self.groups)
            ],
            dim=1,
        )

        coarse_weights = torch.softmax(
            self.modality_gate(modality_states).squeeze(-1), dim=1
        )
        coarse_summary = torch.cat(
            (
                (modality_states * coarse_weights.unsqueeze(-1)).sum(dim=1),
                modality_states.max(dim=1).values,
            ),
            dim=-1,
        )
        coarse_local = self.fusion(coarse_summary).reshape(batch, services, -1)
        coarse_nodes, coarse_graph = self._propagate_state(coarse_local, adjacency)

        root_summary, root_weights = self.root_router(modality_states.detach())
        root_local = self.root_fusion(root_summary).reshape(batch, services, -1)
        root_nodes, root_graph = self._root_propagate(root_local, adjacency)
        context = root_graph.unsqueeze(1).expand(-1, services, -1)
        node_logits = self.root_head(
            torch.cat((root_nodes, context, root_nodes * context), dim=-1)
        ).squeeze(-1)
        return {
            "failure_logits": self.failure_head(coarse_graph),
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": root_nodes,
            "failure_node_embeddings": coarse_nodes,
            "failure_modality_weights": coarse_weights.reshape(
                batch, services, -1
            ),
            "root_modality_weights": root_weights.reshape(batch, services, -1),
        }


class LTFPContextEvidenceCollaboration(LTFPBaseline):
    """Protected asymmetric collaboration between system and service evidence.

    The original LTFP path remains responsible for system risk.  RCA learns a
    service evidence space from detached LTFP representations.  A separate
    failure-supervised collaboration path uses global context to select among
    detached service evidence and contributes only a zero-initialized residual
    to the original failure logit.
    """

    root_sidecar_parameter_prefixes = (
        "local_evidence_adapter.",
        "root_head.",
    )

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
        self.local_evidence_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.collaboration_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.collaboration_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.collaboration_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.collaboration_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.evidence_reliability = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.collaboration_residual_scale = nn.Parameter(torch.zeros(()))
        self.attention_scale = hidden_dim**-0.5

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, graph_state = self._encode_paths(
            node_features, adjacency
        )
        global_failure_logits = self.failure_head(graph_state)

        evidence_input = torch.cat(
            (local_state, node_state, node_state - local_state), dim=-1
        ).detach()
        service_evidence = self.local_evidence_adapter(evidence_input)
        root_context = graph_state.detach().unsqueeze(1).expand_as(service_evidence)
        node_logits = self.root_head(
            torch.cat(
                (
                    service_evidence,
                    root_context,
                    service_evidence * root_context,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        # Failure supervision trains only the collaboration modules here.  It
        # cannot reshape the RCA evidence space or the global representation.
        global_for_collaboration = graph_state.detach()
        evidence_for_collaboration = service_evidence.detach()
        query = self.collaboration_query(global_for_collaboration).unsqueeze(1)
        keys = self.collaboration_key(evidence_for_collaboration)
        values = self.collaboration_value(evidence_for_collaboration)
        attention = torch.softmax(
            (query * keys).sum(dim=-1) * self.attention_scale, dim=1
        )
        selected_evidence = (attention.unsqueeze(-1) * values).sum(dim=1)
        collaboration_state = torch.cat(
            (global_for_collaboration, selected_evidence), dim=-1
        )
        reliability = torch.sigmoid(
            self.evidence_reliability(collaboration_state).squeeze(-1)
        )
        residual = self.collaboration_residual(collaboration_state).squeeze(-1)
        failure_logits = global_failure_logits.clone()
        failure_logits[:, 1] = (
            failure_logits[:, 1]
            + self.collaboration_residual_scale * reliability * residual
        )
        return {
            "failure_logits": failure_logits,
            "global_failure_logits": global_failure_logits,
            "node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "node_embeddings": node_state,
            "local_node_embeddings": local_state,
            "service_evidence_embeddings": service_evidence,
            "collaboration_attention": attention,
            "collaboration_reliability": reliability,
            "collaboration_residual": residual,
            "collaboration_residual_scale": self.collaboration_residual_scale,
        }


class LTFPStateRelationEvidence(LTFPBaseline):
    """Keep normal-anchored metric state separate from graph relation context.

    The original LTFP failure path is unchanged.  RCA is learned in a detached
    sidecar from raw metric histories and their train-only normal deviations.
    The capacity-matched local control uses an identity graph; the relation
    variant uses the observed trace-derived adjacency.
    """

    root_sidecar_parameter_prefixes = (
        "metric_encoder.",
        "deviation_projection.",
        "local_norm.",
        "relation_layer.",
        "relation_norm.",
        "fine_root_head.",
    )

    def __init__(
        self,
        feature_dim: int,
        num_services: int,
        history: int,
        metric_value_indices: Sequence[int],
        metric_mask_indices: Sequence[int | None],
        use_relation_graph: bool,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        if history < 1 or num_services < 1:
            raise ValueError("history and num_services must be positive")
        if not metric_value_indices:
            raise ValueError("state-relation evidence requires metric values")
        if len(metric_value_indices) != len(metric_mask_indices):
            raise ValueError("metric value and mask specifications must align")
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.num_services = int(num_services)
        self.history = int(history)
        self.metric_value_indices = tuple(int(index) for index in metric_value_indices)
        self.metric_mask_indices = tuple(
            None if index is None else int(index) for index in metric_mask_indices
        )
        self.use_relation_graph = bool(use_relation_graph)

        # Do not let sidecar initialization change the random stream used by
        # the seed-matched original LTFP failure path.
        rng_state = torch.random.get_rng_state()
        metric_dim = len(self.metric_value_indices)
        self.metric_encoder = _TemporalAttentionBranch(
            metric_dim * 3, hidden_dim, temporal_layers, 0.0
        )
        self.deviation_projection = nn.Sequential(
            nn.Linear(metric_dim * 3, hidden_dim),
            nn.ReLU(),
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.relation_layer = _WeightedGraphAttention(hidden_dim, 0.0)
        self.relation_norm = nn.LayerNorm(hidden_dim)
        self.fine_root_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.fine_root_head[-1].weight)
        nn.init.zeros_(self.fine_root_head[-1].bias)
        torch.random.set_rng_state(rng_state)

        reference_shape = (history, num_services, metric_dim)
        self.register_buffer("normal_median", torch.zeros(reference_shape))
        self.register_buffer("normal_scale", torch.ones(reference_shape))
        self.register_buffer("normal_reference_ready", torch.tensor(False))
        self.register_buffer("relation_permutation", torch.arange(num_services))

    def configure_normal_reference(
        self, median: torch.Tensor, scale: torch.Tensor
    ) -> None:
        """Install train-only robust metric references for every history slot."""
        expected = self.normal_median.shape
        if median.shape != expected or scale.shape != expected:
            raise ValueError(
                f"normal reference must have shape {tuple(expected)}; "
                f"got {tuple(median.shape)} and {tuple(scale.shape)}"
            )
        if not torch.isfinite(median).all() or not torch.isfinite(scale).all():
            raise ValueError("normal reference must be finite")
        if (scale <= 0).any():
            raise ValueError("normal reference scales must be positive")
        self.normal_median.copy_(median.to(self.normal_median))
        self.normal_scale.copy_(scale.to(self.normal_scale))
        self.normal_reference_ready.fill_(True)

    def _metric_masks(self, node_features: torch.Tensor) -> torch.Tensor:
        masks = []
        for index in self.metric_mask_indices:
            if index is None:
                masks.append(torch.ones_like(node_features[..., 0]))
            else:
                masks.append(node_features[..., index])
        return torch.stack(masks, dim=-1).clamp(0.0, 1.0)

    def _local_metric_evidence(
        self, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.normal_reference_ready):
            raise RuntimeError("normal metric reference has not been configured")
        values = node_features[..., list(self.metric_value_indices)]
        masks = self._metric_masks(node_features)
        deviation = (
            (values - self.normal_median.unsqueeze(0)).abs()
            / self.normal_scale.unsqueeze(0)
        ).clamp(max=10.0) * masks
        batch, steps, services, _ = values.shape
        sequence = torch.cat((values, deviation, masks), dim=-1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            batch * services, steps, -1
        )
        metric_state = self.metric_encoder(sequence).reshape(batch, services, -1)
        deviation_summary = torch.cat(
            (
                deviation[:, -1],
                deviation.max(dim=1).values,
                deviation.mean(dim=1),
            ),
            dim=-1,
        )
        local_state = self.local_norm(
            metric_state + self.deviation_projection(deviation_summary)
        )
        return local_state, deviation

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        local_state, node_state, graph_state = self._encode_paths(
            node_features, adjacency
        )
        failure_logits = self.failure_head(graph_state)

        with torch.no_grad():
            context = graph_state.unsqueeze(1).expand_as(node_state)
            baseline_logits = self.root_head(
                torch.cat((node_state, context, node_state * context), dim=-1)
            ).squeeze(-1)
        output = self.forward_evidence(node_features, adjacency, baseline_logits)
        output.update({
            "failure_logits": failure_logits,
            "node_embeddings": node_state,
            "local_node_embeddings": local_state,
        })
        return output

    def forward_evidence(self, node_features, adjacency, baseline_logits):
        """Train on cached frozen logits without recomputing the backbone."""
        evidence_local, deviation = self._local_metric_evidence(node_features.detach())
        batch, services, _ = evidence_local.shape
        relation_graph = adjacency[:, -1]
        relation_graph = relation_graph.index_select(
            1, self.relation_permutation
        ).index_select(2, self.relation_permutation)
        if not self.use_relation_graph:
            relation_graph = torch.eye(
                services,
                dtype=adjacency.dtype,
                device=adjacency.device,
            ).expand(batch, -1, -1)
        relation_state = self.relation_norm(
            evidence_local
            + self.relation_layer(evidence_local, relation_graph)
        )
        residual = self.fine_root_head(
            torch.cat(
                (
                    evidence_local,
                    relation_state,
                    evidence_local * relation_state,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        # The pretrained RCA head is a protected reference. Only the sidecar
        # learns corrections; zero initialization exactly recovers its scores.
        node_logits = baseline_logits.detach() + residual
        return {
            "node_logits": node_logits,
            "rank_node_logits": node_logits,
            "node_scores": torch.softmax(node_logits, dim=-1),
            "evidence_local_embeddings": evidence_local,
            "evidence_relation_embeddings": relation_state,
            "metric_normal_deviation": deviation,
        }


class LTFPMetricBypass(LTFPStateRelationEvidence):
    """Small shared RCA calibrator with explicit metric/deviation summaries.

    Raw control duplicates the raw summary in the auxiliary input slots;
    deviation mode replaces those slots with train-normal deviations. Both
    have the same trainable parameter count and start at the frozen baseline.
    """

    root_sidecar_parameter_prefixes = ("bypass_head.",)

    def __init__(self, *args, use_normal_deviation: bool, **kwargs):
        super().__init__(*args, use_relation_graph=False, **kwargs)
        self.use_normal_deviation = bool(use_normal_deviation)
        for name in ("metric_encoder", "deviation_projection", "local_norm",
                     "relation_layer", "relation_norm", "fine_root_head"):
            delattr(self, name)
        rng_state = torch.random.get_rng_state()
        self.bypass_head = nn.Sequential(
            nn.Linear(len(self.metric_value_indices) * 9 + 2, 16),
            nn.ReLU(), nn.Linear(16, 1),
        )
        nn.init.zeros_(self.bypass_head[-1].weight)
        nn.init.zeros_(self.bypass_head[-1].bias)
        torch.random.set_rng_state(rng_state)

    @staticmethod
    def _summary(values, masks):
        observed = masks.sum(dim=1).clamp(min=1)
        mean = (values * masks).sum(dim=1) / observed
        maximum = values.masked_fill(masks <= 0, float("-inf")).max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat((values[:, -1] * masks[:, -1], mean, maximum), dim=-1)

    def metric_evidence_features(self, node_features, baseline_logits):
        """Explicit per-service inputs, before either learned correction head."""
        values = node_features[..., list(self.metric_value_indices)].detach()
        masks = self._metric_masks(node_features)
        raw_summary = self._summary(values, masks)
        if self.use_normal_deviation:
            if not bool(self.normal_reference_ready):
                raise RuntimeError("normal reference has not been configured")
            deviation = ((values - self.normal_median.unsqueeze(0)).abs()
                         / self.normal_scale.unsqueeze(0)).clamp(max=10.0)
            extra = self._summary(deviation, masks)
        else:
            extra = raw_summary
        mask_summary = torch.cat((masks[:, -1], masks.mean(dim=1), masks.max(dim=1).values), dim=-1)
        logits = baseline_logits.detach()
        confidence = torch.stack((logits - logits.mean(dim=1, keepdim=True), logits.softmax(dim=1)), dim=-1)
        features = torch.cat((raw_summary, mask_summary, extra, confidence), dim=-1)
        return features

    def forward_evidence(self, node_features, adjacency, baseline_logits):
        features = self.metric_evidence_features(node_features, baseline_logits)
        logits = baseline_logits.detach()
        residual = self.bypass_head(features).squeeze(-1)
        node_logits = logits + residual
        return {"node_logits": node_logits, "rank_node_logits": node_logits,
                "node_scores": node_logits.softmax(dim=1), "bypass_residual": residual}


__all__ = [
    "LTFPIsolatedLocalAdapterEvidence",
    "LTFPLocalAdapterEvidence",
    "LTFPNullAnchoredEvidence",
    "LTFPProtectedRootEvidence",
    "LTFPProtectedRootSidecar",
    "LTFPContextEvidenceCollaboration",
    "LTFPStateRelationEvidence",
    "LTFPMetricBypass",
    "LTFPResidualEvidence",
]
