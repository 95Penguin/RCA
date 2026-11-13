"""Shared building blocks for models using the repository tensor contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def validate_graph_inputs(
    node_features: torch.Tensor, adjacency: torch.Tensor
) -> None:
    """Validate the common ``[batch, time, service, ...]`` model inputs."""
    if node_features.ndim != 4:
        raise ValueError("node_features must have shape [B, T, N, F]")
    if adjacency.ndim != 4:
        raise ValueError("adjacency must have shape [B, T, N, N]")
    if node_features.shape[:3] != adjacency.shape[:3]:
        raise ValueError("feature and adjacency batch/time/node dimensions must match")
    if adjacency.shape[-1] != node_features.shape[2]:
        raise ValueError("adjacency must be square over the service dimension")


def normalize_feature_groups(
    feature_dim: int, groups: Sequence[Sequence[int]] | None
) -> tuple[tuple[int, ...], ...]:
    """Return and validate a complete, non-overlapping feature partition."""
    if not groups:
        return (tuple(range(feature_dim)),)
    normalized = tuple(tuple(int(index) for index in group) for group in groups if group)
    flattened = [index for group in normalized for index in group]
    if sorted(flattened) != list(range(feature_dim)):
        raise ValueError("feature groups must partition every feature exactly once")
    return normalized


class CausalConv1d(nn.Module):
    """A length-preserving convolution that only reads current or past steps."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        kernel_size = 3
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))
