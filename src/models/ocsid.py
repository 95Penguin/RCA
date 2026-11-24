"""OCSID: operation-conditioned, source-impact decomposed RCA model.

The model has three cooperating parts, each with a distinct job:

1. **Service temporal + graph encoder** (reused LTFP building blocks, trained
   from scratch here): encodes each service's five-minute telemetry and lets
   callers/callees exchange context.  This is the part that predicts whether
   the system is about to fail; the graph is what the no-graph ablation showed
   to be essential.
2. **Operation risk head**: a shared GRU over each entry operation's trace
   features produces ``q_o``.  Operations are where the label is defined
   (TT: sustained same-operation entry p95 degradation), so keeping them
   explicit makes the prediction interpretable and gives the RCA head a
   conditioned context.
3. **Source-impact RCA head**: for each service, a source representation
   (local evidence) and an impact representation (graph-propagated context)
   are fused into ``a_i``.  This is the learned counterpart of CPCA's
   structural counterfactual; the two can be combined as
   ``R_i = a_i + lambda * r_i`` when the structural score is available.

The class is deliberately a subclass of :class:`LTFPBaseline` so the stable
temporal/graph components are not re-implemented; all weights are trained from
random initialization and no LTFP checkpoint is loaded.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .ltfp import LTFPBaseline


class OCSIDModel(LTFPBaseline):
    def __init__(
        self,
        feature_dim: int,
        operation_dim: int = 17,
        feature_groups: Sequence[Sequence[int]] | None = None,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        operation_hidden: int = 32,
        aggregation: str = "max",
        use_graph: bool = True,
        use_operation: bool = True,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            feature_groups=feature_groups,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            graph_layers=graph_layers,
            dropout=dropout,
            num_classes=2,
        )
        if aggregation not in {"max", "product"}:
            raise ValueError("aggregation must be 'max' or 'product'")
        self.aggregation = aggregation
        self.use_graph = use_graph
        self.use_operation = use_operation

        # Operation risk: one shared encoder, one scalar per operation.
        self.operation_encoder = nn.GRU(operation_dim, operation_hidden, batch_first=True)
        self.operation_norm = nn.LayerNorm(operation_hidden)
        self.operation_head = nn.Linear(operation_hidden, 1)

        # Fuse the system (graph) and operation views in logit space.
        self.system_fuse = nn.Linear(2, 1)

        # Source / impact decomposition for RCA.
        self.source_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.impact_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        # The inherited root_head has the same 3*H -> H -> 1 shape; reuse it
        # explicitly so older checkpoints and the LTFP comparison stay aligned.
        self.root_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        operation_features: torch.Tensor,
        operation_mask: torch.Tensor,
        cpca: torch.Tensor | None = None,
        cpca_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if self.use_graph:
            local_state, node_state, graph_state = self._encode_paths(
                node_features, adjacency
            )
        else:
            local_state = self._encode_local(node_features, adjacency)
            node_state = local_state
            graph_state = self.pool(torch.cat((
                local_state.mean(dim=1), local_state.max(dim=1).values
            ), dim=-1))

        # ---- operation risk ----
        batch, steps, operations, dim = operation_features.shape
        masked_ops = operation_features * operation_mask.unsqueeze(-1)
        hidden, _ = self.operation_encoder(
            masked_ops.reshape(batch * operations, steps, dim)
        )
        q = torch.sigmoid(
            self.operation_head(self.operation_norm(hidden[:, -1])).reshape(batch, operations)
        )
        present = operation_mask[:, -1]
        if self.aggregation == "product":
            log_one_minus = torch.log1p(-q.clamp(max=1.0 - 1e-6))
            p_operation = 1.0 - torch.exp((log_one_minus * present).sum(dim=1))
        else:
            p_operation = (q * present).max(dim=1).values

        # ---- system (graph) risk ----
        system_logits = self.failure_head(graph_state)
        p_system = torch.softmax(system_logits, dim=-1)[:, 1]
        eps = 1e-6
        if self.use_operation:
            fused = self.system_fuse(torch.stack((
                torch.log(p_system.clamp(eps, 1 - eps) / (1 - p_system).clamp(eps, 1 - eps)),
                torch.log(p_operation.clamp(eps, 1 - eps) / (1 - p_operation).clamp(eps, 1 - eps)),
            ), dim=-1)).squeeze(-1)
            probability = torch.sigmoid(fused)
        else:
            p_operation = p_system
            probability = p_system

        # ---- source / impact RCA ----
        source = self.source_mlp(local_state)
        impact = self.impact_mlp(node_state)
        node_logits = self.root_head(
            torch.cat((source, impact, source * impact), dim=-1)
        ).squeeze(-1)
        if cpca is not None and cpca_weight:
            node_logits = node_logits + cpca_weight * torch.log(
                cpca.clamp_min(1e-6)
            )
        node_scores = torch.softmax(node_logits, dim=-1)

        return {
            "failure_logits": system_logits,
            "probability": probability,
            "system_probability": p_system,
            "operation_probability": p_operation,
            "operation_scores": q,
            "node_logits": node_logits,
            "node_scores": node_scores,
            "node_embeddings": node_state,
            "source_embeddings": source,
            "impact_embeddings": impact,
        }
