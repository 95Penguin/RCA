"""Compatibility exports for the former combined paper-baseline module.

New code should import from :mod:`models.ltfp` or :mod:`models.eadro`.
"""

from .eadro import EadroAdapted
from .ltfp import LTFPBaseline

__all__ = ["EadroAdapted", "LTFPBaseline"]
