"""Backward-compatible imports for the former InstantOps module name."""

from .instantops import GraphConvolution, InstantOpsGNNGRU

__all__ = ["GraphConvolution", "InstantOpsGNNGRU"]
