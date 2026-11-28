#!/usr/bin/env python3
"""Compatibility CLI; implementation lives in :mod:`ocsid.training`.

Historical arguments and imported helper names are preserved.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.training import PANELS, OP_FEATURES, _build_arrays, _standardize, _calibrate, _f1, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
