#!/usr/bin/env python3
"""Compatibility CLI; implementation lives in :mod:`ocsid.operation_builder`.

Historical arguments and imported helper names are preserved.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ocsid.operation_builder import ARCHIVES, PREFIX, FIELDS, build_case, _case_members, _case_id, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
