"""Minimal local-to-system interaction under a frozen prediction backbone."""

import torch
from torch import nn


class ServiceRiskInteraction(nn.Module):
    """Auxiliary RCA shapes a local representation used for system prediction.

    The root head is auxiliary in this stage. Deployed RCA scores come from
    the frozen previous-stage checkpoint, not this unscreened auxiliary head.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 16, bound: float = 1.0):
        super().__init__()
        if bound <= 0:
            raise ValueError("residual bound must be positive")
        self.bound = float(bound)
        self.service_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.root_auxiliary = nn.Linear(hidden_dim, 1)
        self.system_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        for head in (self.root_auxiliary, self.system_residual[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features, baseline_failure_logits, baseline_root_logits):
        evidence = self.service_encoder(features.detach())
        base_failure = baseline_failure_logits.detach()
        margin = (base_failure[:, 1] - base_failure[:, 0]).unsqueeze(-1)
        pooled = torch.cat((evidence.mean(1), evidence.max(1).values, margin), dim=-1)
        residual = self.bound * self.system_residual(pooled).squeeze(-1).tanh()
        failure_logits = base_failure + torch.stack((torch.zeros_like(residual), residual), dim=1)
        auxiliary = baseline_root_logits.detach() + self.bound * self.root_auxiliary(evidence).squeeze(-1).tanh()
        return {"failure_logits": failure_logits, "auxiliary_root_logits": auxiliary,
                "system_residual": residual}
