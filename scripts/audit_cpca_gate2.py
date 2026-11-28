#!/usr/bin/env python3
"""Compatibility CLI; implementation lives in :mod:`ocsid.audit`.

Historical arguments and imported helper names are preserved.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.audit import Span, _tree, _replay, _score_trace, _rank, _metrics, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
