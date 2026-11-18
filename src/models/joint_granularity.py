"""End-to-end system/service prediction from a shared telemetry backbone."""

import torch
from torch import nn

from .common import validate_graph_inputs
from .ltfp import LTFPBaseline


class JointGranularityModel(LTFPBaseline):
    """One trainable model; final system and RCA heads share service evidence.

    The historical variant keeps a direct normal-deviation path. The encoded-local
    control disables only its direct input to the failure head; deviations still
    shape the shared service evidence and the RCA head. Logs/traces contribute through
    the shared global path. Global context selects service evidence for failure
    prediction and conditions the final root head. Optional null selection permits
    attention mass on a zero value without changing either task's output semantics.
    No backbone is frozen.
    """

    def __init__(self, feature_dim, num_services, history, metric_value_indices,
                 metric_mask_indices, feature_groups=None, hidden_dim=64,
                 temporal_layers=2, graph_layers=2, dropout=0.1,
                 *, direct_failure_deviation=True, allow_null_evidence=False):
        if not metric_value_indices or len(metric_value_indices) != len(metric_mask_indices):
            raise ValueError("aligned metric values and masks are required")
        super().__init__(feature_dim, feature_groups, hidden_dim, temporal_layers, graph_layers, dropout)
        self.direct_failure_deviation = bool(direct_failure_deviation)
        self.allow_null_evidence = bool(allow_null_evidence)
        self.metric_value_indices = tuple(metric_value_indices)
        self.metric_mask_indices = tuple(metric_mask_indices)
        self.metric_branch = next(i for i, group in enumerate(self.groups)
                                  if set(metric_value_indices).issubset(group))
        dim = len(metric_value_indices) * 3
        self.service_evidence = nn.Sequential(nn.Linear(hidden_dim + dim, hidden_dim), nn.ReLU())
        self.context_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.service_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.failure_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )
        self.root_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.attention_scale = hidden_dim ** -0.5
        # A learned key and an exactly zero value let attention decline local input.
        # Zero initialization does not shift any existing module's RNG stream.
        self.register_parameter(
            "null_evidence_key",
            nn.Parameter(torch.zeros(hidden_dim)) if self.allow_null_evidence else None,
        )
        shape = (history, num_services, len(metric_value_indices))
        self.register_buffer("normal_median", torch.zeros(shape))
        self.register_buffer("normal_scale", torch.ones(shape))
        self.register_buffer("normal_reference_ready", torch.tensor(False))

    def configure_normal_reference(self, median, scale):
        if median.shape != self.normal_median.shape or scale.shape != self.normal_scale.shape:
            raise ValueError("normal reference shape mismatch")
        if not torch.isfinite(median).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("normal reference must be finite with positive scales")
        self.normal_median.copy_(median.to(self.normal_median))
        self.normal_scale.copy_(scale.to(self.normal_scale))
        self.normal_reference_ready.fill_(True)

    def _metric_deviation(self, features):
        if not bool(self.normal_reference_ready):
            raise RuntimeError("normal reference has not been configured")
        values = features[..., list(self.metric_value_indices)]
        masks = torch.stack([
            torch.ones_like(features[..., 0]) if index is None else features[..., index]
            for index in self.metric_mask_indices
        ], dim=-1).clamp(0, 1)
        deviation = ((values - self.normal_median.unsqueeze(0)).abs()
                     / self.normal_scale.unsqueeze(0)).clamp(max=10) * masks
        return torch.cat((deviation[:, -1], deviation.max(1).values,
                          deviation.sum(1)/masks.sum(1).clamp(min=1)), dim=-1)

    def _service_attention(self, global_state, evidence):
        query = self.context_query(global_state).unsqueeze(1)
        logits = (query * self.service_key(evidence)).sum(-1) * self.attention_scale
        if self.allow_null_evidence:
            null_logit = (query * self.null_evidence_key).sum(-1) * self.attention_scale
            distribution = torch.cat((null_logit, logits), dim=1).softmax(1)
            # Do not renormalize service weights: their missing mass is the null value.
            return distribution[:, 1:], distribution[:, 0]
        return logits.softmax(1), logits.new_zeros(logits.size(0))

    def forward(self, node_features, adjacency):
        validate_graph_inputs(node_features, adjacency)
        batch, steps, services, _ = node_features.shape
        flat = node_features.permute(0, 2, 1, 3).reshape(batch*services, steps, -1)
        states = torch.stack([branch(flat[..., group]) for branch, group in zip(self.branches, self.groups)], dim=1)
        weights = self.modality_gate(states).squeeze(-1).softmax(1)
        local = self.fusion(torch.cat(((states*weights.unsqueeze(-1)).sum(1), states.max(1).values), -1))
        local = local.reshape(batch, services, -1)
        nodes, global_state = self._propagate_state(local, adjacency)
        metric_state = states[:, self.metric_branch].reshape(batch, services, -1)
        deviation = self._metric_deviation(node_features)
        evidence = self.service_evidence(torch.cat((metric_state, deviation), -1))
        attention, null_attention = self._service_attention(global_state, evidence)
        selected = (attention.unsqueeze(-1)*evidence).sum(1)
        selected_deviation = (attention.unsqueeze(-1)*deviation).sum(1)
        # Keep the same dimensions, parameter initialization and compute graph.
        # The ablation removes only the raw bypass, during both training and inference.
        failure_deviation = (selected_deviation if self.direct_failure_deviation
                             else torch.zeros_like(selected_deviation))
        failure_logits = self.failure_head(torch.cat((global_state, selected, failure_deviation), -1))
        context = global_state.unsqueeze(1).expand(-1, services, -1)
        root_logits = self.root_head(torch.cat((evidence, nodes, context, deviation), -1)).squeeze(-1)
        return {"failure_logits": failure_logits, "node_logits": root_logits,
                "rank_node_logits": root_logits, "node_scores": root_logits.softmax(-1),
                "node_embeddings": nodes, "service_evidence_embeddings": evidence,
                "cross_granularity_attention": attention,
                "null_evidence_attention": null_attention}
