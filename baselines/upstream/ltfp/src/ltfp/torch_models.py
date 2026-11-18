from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:  # pragma: no cover - optional dependency in this workspace
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for ltfp.torch_models")


@dataclass(frozen=True)
class LTFPConfig:
    input_dim: int
    hidden_dim: int
    n_fault_types: int
    n_horizons: int
    n_tcn_layers: int = 3
    n_graph_layers: int = 2
    dropout: float = 0.1
    modality_dims: tuple[int, ...] = ()
    root_prior_weight: float = 0.0
    edge_weight_gamma: float = 1.0
    gat_negative_slope: float = 0.2
    gat_edge_eps: float = 1e-12
    enable_type_head: bool = False
    enable_auxiliary_heads: bool = False
    enable_root_prior: bool = False
    fusion_mode: str = "gated"
    monotonic_horizons: bool = True

    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.n_horizons <= 0:
            raise ValueError("n_horizons must be positive")
        if self.n_fault_types < 0:
            raise ValueError("n_fault_types must be non-negative")
        if self.gat_edge_eps <= 0:
            raise ValueError("gat_edge_eps must be positive")
        if self.fusion_mode not in {"gated", "concat"}:
            raise ValueError("fusion_mode must be 'gated' or 'concat'")


class TemporalBlock(nn.Module if nn else object):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float):
        require_torch()
        super().__init__()
        padding = dilation
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out[:, :, : x.size(-1)]
        out = self.dropout(F.relu(self.norm(out)))
        return out + residual


class TemporalBranch(nn.Module if nn else object):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        require_torch()
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(hidden_dim, dilation=2**idx, dropout=dropout)
                for idx in range(n_layers)
            ]
        )
        self.time_attn = nn.Linear(hidden_dim, 1)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        seq = x.transpose(1, 2)
        attn = torch.softmax(self.time_attn(seq).squeeze(-1), dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), seq).squeeze(1)
        last = seq[:, -1, :]
        return self.readout(torch.cat([last, pooled], dim=-1))


class TemporalEncoder(nn.Module if nn else object):
    """Modality-aware TCN encoder for per-service metric/log/trace windows.

    Input shape: `[batch, nodes, window, input_dim]`.
    Output shape: `[batch, nodes, hidden_dim]`.
    """

    def __init__(self, config: LTFPConfig):
        require_torch()
        super().__init__()
        modality_dims = tuple(dim for dim in config.modality_dims if dim > 0)
        if not modality_dims or sum(modality_dims) != config.input_dim:
            modality_dims = (config.input_dim,)
        self.modality_dims = modality_dims
        self.branches = nn.ModuleList(
            [
                TemporalBranch(dim, config.hidden_dim, config.n_tcn_layers, config.dropout)
                for dim in modality_dims
            ]
        )
        self.modality_gate = nn.Linear(config.hidden_dim, 1)
        self.fusion_mode = config.fusion_mode
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.concat_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(modality_dims), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )

    def _modality_mask(self, x, modality_mask=None):
        """Return a boolean availability mask with shape ``[B, N, Q]``.

        The compact ``[Q]`` and ``[N, Q]`` forms are accepted for callers that
        use one availability pattern for a whole batch. For a single modality,
        ``[B, N]`` is also accepted.
        """

        batch, nodes = x.shape[:2]
        n_modalities = len(self.modality_dims)
        if modality_mask is None:
            return torch.ones((batch, nodes, n_modalities), dtype=torch.bool, device=x.device)

        mask = torch.as_tensor(modality_mask, dtype=torch.bool, device=x.device)
        if mask.dim() == 1 and mask.shape[0] == n_modalities:
            mask = mask.view(1, 1, n_modalities)
        elif mask.dim() == 2 and mask.shape == (nodes, n_modalities):
            mask = mask.unsqueeze(0)
        elif n_modalities == 1 and mask.dim() == 2 and mask.shape == (batch, nodes):
            mask = mask.unsqueeze(-1)
        elif mask.dim() != 3:
            raise ValueError(
                "modality_mask must have shape [Q], [N,Q], [B,N,Q], "
                "or [B,N] when Q=1"
            )

        try:
            return torch.broadcast_to(mask, (batch, nodes, n_modalities))
        except RuntimeError as exc:
            raise ValueError(
                f"modality_mask shape {tuple(mask.shape)} cannot broadcast to "
                f"{(batch, nodes, n_modalities)}"
            ) from exc

    def feature_availability_mask(self, x, modality_mask=None):
        """Expand a service--modality mask to the concatenated feature axis."""

        mask = self._modality_mask(x, modality_mask)
        feature_masks = [
            mask[..., modality_idx : modality_idx + 1].expand(-1, -1, width)
            for modality_idx, width in enumerate(self.modality_dims)
        ]
        return torch.cat(feature_masks, dim=-1)

    def forward(self, x, modality_mask=None):
        batch, nodes, window, _ = x.shape
        flat = x.reshape(batch * nodes, window, -1)
        chunks = torch.split(flat, self.modality_dims, dim=-1)
        flat_mask = self._modality_mask(x, modality_mask).reshape(batch * nodes, -1)
        states = []
        for modality_idx, (branch, chunk) in enumerate(zip(self.branches, chunks)):
            available = flat_mask[:, modality_idx]
            state = x.new_zeros((batch * nodes, self.fusion[0].out_features))
            if available.any():
                available_idx = torch.nonzero(available, as_tuple=False).squeeze(-1)
                encoded = branch(chunk.index_select(0, available_idx))
                state = state.index_copy(0, available_idx, encoded)
            states.append(state)

        # The same learned projection is used for one or many modalities. This
        # keeps metric-only and multimodal subsets on the same model path.
        stacked = torch.stack(states, dim=1)
        any_available = flat_mask.any(dim=1)
        if self.fusion_mode == "concat":
            concatenated = stacked.masked_fill(~flat_mask.unsqueeze(-1), 0.0).reshape(batch * nodes, -1)
            fused = self.concat_fusion(concatenated)
        else:
            gate_logits = self.modality_gate(stacked).squeeze(-1)
            gate_logits = gate_logits.masked_fill(~flat_mask, float("-inf"))
            safe_gate_logits = torch.where(
                any_available.unsqueeze(-1), gate_logits, torch.zeros_like(gate_logits)
            )
            gate = torch.softmax(safe_gate_logits, dim=-1)
            gate = gate.masked_fill(~flat_mask, 0.0)
            weighted = torch.sum(stacked * gate.unsqueeze(-1), dim=1)

            masked_states = stacked.masked_fill(~flat_mask.unsqueeze(-1), float("-inf"))
            pooled = torch.max(masked_states, dim=1).values
            pooled = torch.where(any_available.unsqueeze(-1), pooled, torch.zeros_like(pooled))
            fused = self.fusion(torch.cat([weighted, pooled], dim=-1))
        fused = torch.where(any_available.unsqueeze(-1), fused, torch.zeros_like(fused))
        return fused.reshape(batch, nodes, -1)


class GraphAttentionLayer(nn.Module if nn else object):
    """Dense weighted graph attention over service nodes.

    Positive adjacency entries are both an edge mask and an attention prior.
    Every node receives a self-loop of weight at least one.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        edge_weight_gamma: float = 1.0,
        negative_slope: float = 0.2,
        edge_eps: float = 1e-12,
    ):
        require_torch()
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.edge_weight_gamma = float(edge_weight_gamma)
        self.negative_slope = float(negative_slope)
        self.edge_eps = float(edge_eps)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.attn_src, std=0.02)
        nn.init.normal_(self.attn_dst, std=0.02)

    def _edge_weights(self, node_state, adjacency):
        batch, nodes = node_state.shape[:2]
        weights = torch.as_tensor(adjacency, dtype=node_state.dtype, device=node_state.device)
        if weights.dim() == 2:
            if weights.shape != (nodes, nodes):
                raise ValueError(f"adjacency must have shape {(nodes, nodes)}, got {tuple(weights.shape)}")
            weights = weights.unsqueeze(0).expand(batch, -1, -1)
        elif weights.dim() == 3:
            if weights.shape[1:] != (nodes, nodes) or weights.shape[0] not in (1, batch):
                raise ValueError(
                    f"batched adjacency must have shape [1|B,{nodes},{nodes}], got {tuple(weights.shape)}"
                )
            weights = weights.expand(batch, -1, -1)
        else:
            raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
        if not torch.isfinite(weights).all():
            raise ValueError("adjacency weights must be finite")
        if (weights < 0).any():
            raise ValueError("adjacency weights must be non-negative")

        eye = torch.eye(nodes, device=node_state.device, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
        weights = torch.where(eye, torch.maximum(weights, torch.ones_like(weights)), weights)
        return weights

    def forward(self, node_state, adjacency, return_attention: bool = False):
        h = self.linear(node_state)
        src_score = torch.einsum("bnh,h->bn", h, self.attn_src)
        dst_score = torch.einsum("bnh,h->bn", h, self.attn_dst)
        edge_weights = self._edge_weights(node_state, adjacency)
        mask = edge_weights > 0
        edge_prior = self.edge_weight_gamma * torch.log(edge_weights + self.edge_eps)
        logits = F.leaky_relu(
            src_score.unsqueeze(2) + dst_score.unsqueeze(1) + edge_prior,
            negative_slope=self.negative_slope,
        )
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out = torch.bmm(attn, h)
        output = F.elu(out)
        if return_attention:
            return output, attn
        return output


class LTFPModel(nn.Module if nn else object):
    """Lead-time-aware temporal graph fault predictor.

    Primary outputs:
    - `fault_logits`: `[batch, n_horizons]`
    - `root_logits`: `[batch, nodes]`

    Legacy type/lead keys remain available for callers that consume the old
    output dictionary. Their heads are disabled by default and must be enabled
    explicitly in :class:`LTFPConfig` to participate in training.
    """

    def __init__(self, config: LTFPConfig):
        require_torch()
        super().__init__()
        self.temporal = TemporalEncoder(config)
        self.graph_layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    config.hidden_dim,
                    dropout=config.dropout,
                    edge_weight_gamma=config.edge_weight_gamma,
                    negative_slope=config.gat_negative_slope,
                    edge_eps=config.gat_edge_eps,
                )
                for _ in range(max(0, config.n_graph_layers))
            ]
        )
        self.graph_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in self.graph_layers])
        self.graph_dropout = nn.Dropout(config.dropout)
        self.pool = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.n_horizons = config.n_horizons
        self.n_fault_types = max(config.n_fault_types, 1)
        self.root_prior_weight = config.root_prior_weight
        self.enable_type_head = bool(config.enable_type_head and config.n_fault_types > 0)
        self.enable_auxiliary_heads = bool(config.enable_auxiliary_heads)
        self.enable_root_prior = bool(config.enable_root_prior and config.root_prior_weight != 0)
        self.monotonic_horizons = bool(config.monotonic_horizons)
        self.fault_base_head = nn.Linear(config.hidden_dim, 1)
        self.fault_delta_head = (
            nn.Linear(config.hidden_dim, config.n_horizons - 1)
            if self.monotonic_horizons and config.n_horizons > 1
            else None
        )
        self.fault_independent_head = (
            nn.Linear(config.hidden_dim, config.n_horizons) if not self.monotonic_horizons else None
        )
        self.type_head = nn.Linear(config.hidden_dim, config.n_fault_types) if self.enable_type_head else None
        self.root_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.lead_base_head = nn.Linear(config.hidden_dim, 1) if self.enable_auxiliary_heads else None
        self.lead_delta_head = (
            nn.Linear(config.hidden_dim, config.n_horizons - 1)
            if self.enable_auxiliary_heads and config.n_horizons > 1
            else None
        )
        self.lead_time_head = nn.Linear(config.hidden_dim, 1) if self.enable_auxiliary_heads else None
        self._init_horizon_heads()

    def _init_horizon_heads(self):
        nn.init.constant_(self.fault_base_head.bias, -1.0)
        if self.fault_delta_head is not None:
            nn.init.constant_(self.fault_delta_head.bias, -2.0)
        if self.fault_independent_head is not None:
            nn.init.constant_(self.fault_independent_head.bias, -1.0)
        if self.enable_auxiliary_heads:
            nn.init.constant_(self.lead_base_head.bias, -1.0)
            nn.init.constant_(self.lead_time_head.bias, 5.7)
            if self.lead_delta_head is not None:
                nn.init.constant_(self.lead_delta_head.bias, -2.0)

    def _monotonic_horizon_logits(self, base_head, delta_head, graph_state):
        base = base_head(graph_state)
        if delta_head is None:
            return base
        deltas = F.softplus(delta_head(graph_state))
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def _recent_magnitude_prior(self, x, modality_mask=None):
        recent = x[:, :, -min(5, x.size(2)) :, :]
        feature_mask = self.temporal.feature_availability_mask(x, modality_mask)
        expanded_mask = feature_mask.unsqueeze(2).expand(-1, -1, recent.size(2), -1)
        denominator = expanded_mask.sum(dim=(2, 3)).clamp_min(1)
        magnitude = (recent.abs() * expanded_mask.to(recent.dtype)).sum(dim=(2, 3)) / denominator

        node_available = feature_mask.any(dim=-1)
        count = node_available.sum(dim=1, keepdim=True).clamp_min(1)
        available_float = node_available.to(magnitude.dtype)
        mean = (magnitude * available_float).sum(dim=1, keepdim=True) / count
        variance = (((magnitude - mean) ** 2) * available_float).sum(dim=1, keepdim=True) / count
        normalized = (magnitude - mean) / torch.sqrt(variance + 1e-6)
        return normalized.masked_fill(~node_available, 0.0)

    def forward(self, x, adjacency, modality_mask=None, *, service_modality_mask=None):
        if modality_mask is not None and service_modality_mask is not None:
            raise ValueError("pass only one of modality_mask and service_modality_mask")
        if service_modality_mask is not None:
            modality_mask = service_modality_mask

        node_state = self.temporal(x, modality_mask=modality_mask)
        for graph_layer, graph_norm in zip(self.graph_layers, self.graph_norms):
            update = graph_layer(node_state, adjacency)
            node_state = graph_norm(node_state + self.graph_dropout(update))
        graph_state = torch.cat([node_state.mean(dim=1), node_state.max(dim=1).values], dim=-1)
        graph_state = self.pool(graph_state)
        graph_context = graph_state.unsqueeze(1).expand(-1, node_state.size(1), -1)
        root_input = torch.cat([node_state, graph_context, node_state * graph_context], dim=-1)
        root_logits = self.root_head(root_input).squeeze(-1)
        if self.enable_root_prior:
            root_logits = root_logits + self.root_prior_weight * self._recent_magnitude_prior(x, modality_mask)

        fault_logits = (
            self._monotonic_horizon_logits(self.fault_base_head, self.fault_delta_head, graph_state)
            if self.monotonic_horizons
            else self.fault_independent_head(graph_state)
        )
        if self.enable_type_head:
            type_logits = self.type_head(graph_state)
        else:
            type_logits = graph_state.new_zeros((graph_state.size(0), self.n_fault_types))
        if self.enable_auxiliary_heads:
            lead_logits = self._monotonic_horizon_logits(
                self.lead_base_head, self.lead_delta_head, graph_state
            )
            lead_time_log = self.lead_time_head(graph_state).squeeze(-1)
        else:
            lead_logits = graph_state.new_zeros((graph_state.size(0), self.n_horizons))
            lead_time_log = graph_state.new_zeros(graph_state.size(0))
        return {
            "fault_logits": fault_logits,
            "type_logits": type_logits,
            "root_logits": root_logits,
            "lead_logits": lead_logits,
            "lead_time_log": lead_time_log,
            "_type_head_enabled": self.enable_type_head,
            "_auxiliary_heads_enabled": self.enable_auxiliary_heads,
            "_root_prior_enabled": self.enable_root_prior,
        }


def lead_time_multitask_loss(
    outputs: dict,
    fault_labels,
    type_labels=None,
    root_labels=None,
    horizons: Sequence[float] | None = None,
    alpha: float = 0.35,
    type_weight: float = 0.0,
    root_weight: float = 0.5,
    lead_weight: float = 0.0,
    lead_reg_weight: float = 0.0,
    horizon_weight_mode: str = "lead",
    focus_horizon: float | None = None,
    focus_weight: float = 1.0,
    lead_times=None,
    pos_weight=None,
):
    require_torch()
    if horizons is None:
        horizon_weight = torch.ones(fault_labels.size(1), device=fault_labels.device)
    else:
        horizon_minutes = torch.as_tensor(horizons, device=fault_labels.device, dtype=torch.float32) / 60.0
        if horizon_weight_mode == "flat":
            horizon_weight = torch.ones_like(horizon_minutes)
        elif horizon_weight_mode == "short":
            max_horizon = torch.clamp(horizon_minutes.max(), min=1e-6)
            horizon_weight = 1.0 + alpha * torch.log1p(max_horizon / torch.clamp(horizon_minutes, min=1e-6))
        else:
            horizon_weight = 1.0 + alpha * torch.log1p(horizon_minutes)
        if focus_horizon is not None and focus_weight > 1.0:
            horizon_seconds = horizon_minutes * 60.0
            focus_idx = torch.argmin(torch.abs(horizon_seconds - float(focus_horizon)))
            horizon_weight = horizon_weight.clone()
            horizon_weight[focus_idx] = horizon_weight[focus_idx] * float(focus_weight)

    fault_loss = F.binary_cross_entropy_with_logits(
        outputs["fault_logits"],
        fault_labels.float(),
        weight=horizon_weight.unsqueeze(0),
        pos_weight=pos_weight,
    )
    total = fault_loss

    positive_mask = fault_labels.max(dim=1).values > 0
    type_head_enabled = bool(outputs.get("_type_head_enabled", "type_logits" in outputs))
    if type_labels is not None and type_weight > 0 and type_head_enabled and outputs.get("type_logits") is not None:
        type_mask = positive_mask & (type_labels >= 0)
        if type_mask.any():
            total = total + type_weight * F.cross_entropy(outputs["type_logits"][type_mask], type_labels[type_mask])
    if root_labels is not None and root_weight > 0:
        root_mask = positive_mask & (root_labels >= 0)
        if root_mask.any():
            total = total + root_weight * F.cross_entropy(outputs["root_logits"][root_mask], root_labels[root_mask])

    auxiliary_heads_enabled = bool(outputs.get("_auxiliary_heads_enabled", "lead_logits" in outputs))
    if lead_weight > 0 and auxiliary_heads_enabled and outputs.get("lead_logits") is not None:
        lead_loss = F.binary_cross_entropy_with_logits(
            outputs["lead_logits"],
            fault_labels.float(),
            weight=horizon_weight.unsqueeze(0),
            pos_weight=pos_weight,
        )
        total = total + lead_weight * lead_loss

    if (
        lead_times is not None
        and lead_reg_weight > 0
        and auxiliary_heads_enabled
        and outputs.get("lead_time_log") is not None
    ):
        finite_mask = torch.isfinite(lead_times) & (lead_times >= 0)
        if horizons is not None:
            max_horizon = max(float(horizon) for horizon in horizons)
            finite_mask = finite_mask & (lead_times <= max_horizon)
            target_lead = torch.clamp(lead_times[finite_mask].float(), min=0.0, max=max_horizon)
        else:
            target_lead = torch.clamp(lead_times[finite_mask].float(), min=0.0)
        if finite_mask.any():
            target_log = torch.log1p(target_lead)
            total = total + lead_reg_weight * F.smooth_l1_loss(outputs["lead_time_log"][finite_mask], target_log)
    return total
