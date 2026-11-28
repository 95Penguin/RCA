"""OCSID: operation-conditioned prediction + structural counterfactual attribution.

Package layout
--------------
- :mod:`ocsid.features` : panel/operation feature schema and operation-table IO.
- :mod:`ocsid.cpca`     : span-tree decomposition, counterfactual replay, ranking.
- :mod:`ocsid.audit`    : normal-reference collection and audit helpers.
- :mod:`models.ocsid`   : the trainable OCSID model.

Entry points
------------
- ``scripts/build_ocsid_operation_features.py`` : raw traces -> operation table.
- ``scripts/train_ocsid_joint.py``              : five-fold training/evaluation.
- ``scripts/audit_cpca_gate2.py``               : CPCA audit and baselines.
- ``scripts/fuse_ocsid_cpca.py``                : learned-head vs CPCA fusion.
"""

from .cpca import Span, decompose_tree, replay, score_request
from .features import (
    OP_FEATURE_NAMES,
    PANEL_FEATURES,
    load_operation_table,
    operation_feature_vector,
)

__all__ = [
    "OCSIDModel",
    "Span",
    "decompose_tree",
    "replay",
    "score_request",
    "OP_FEATURE_NAMES",
    "PANEL_FEATURES",
    "load_operation_table",
    "operation_feature_vector",
]


def __getattr__(name):
    # Trace-only tools must not require PyTorch or import every neural model.
    if name == "OCSIDModel":
        from models.ocsid import OCSIDModel
        return OCSIDModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
