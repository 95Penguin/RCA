"""OCSID predictor: operation-conditioned next-minute failure prediction.

The model has two cooperating parts:

1. **Service temporal + graph encoder** (reused LTFP building blocks, trained
   from scratch here): encodes each service's five-minute telemetry and lets
   callers/callees exchange context.
2. **Operation risk head**: a shared GRU over each entry operation's trace
   features produces ``q_o``.  Operations are where the label is defined
   (TT: sustained same-operation entry p95 degradation), so keeping them
   explicit makes the prediction interpretable.

Attribution is **not** part of this model.  CPCA (``src/ocsid/cpca.py``) is the
only root-cause module: it works on request span trees and is training-free.
The earlier learned source/impact head was removed because it was redundant
with CPCA and weakened the separation between prediction and attribution.

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
        aux_root: bool = False,
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
        self.aux_root = aux_root

        # Operation risk: one shared encoder, one scalar per operation.
        self.operation_encoder = nn.GRU(operation_dim, operation_hidden, batch_first=True)
        self.operation_norm = nn.LayerNorm(operation_hidden)
        self.operation_head = nn.Linear(operation_hidden, 1)

        # Fuse the system (graph) and operation views in logit space.
        self.system_fuse = nn.Linear(2, 1)

        # Optional training-only auxiliary head.  It regularizes the shared
        # encoder with root labels but is never used for attribution and is
        # absent from the inference model (aux_root=False).
        if aux_root:
            self.source_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.impact_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
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

        # ---- optional training-only root auxiliary ----
        aux_root_logits = None
        if self.aux_root:
            source = self.source_mlp(local_state)
            impact = self.impact_mlp(node_state)
            aux_root_logits = self.root_head(
                torch.cat((source, impact, source * impact), dim=-1)
            ).squeeze(-1)

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

        return {
            "failure_logits": system_logits,
            "probability": probability,
            "system_probability": p_system,
            "operation_probability": p_operation,
            "operation_scores": q,
            "node_embeddings": node_state,
            **({"aux_root_logits": aux_root_logits} if aux_root_logits is not None else {}),
        }
