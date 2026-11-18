#!/usr/bin/env python3
"""Run the pinned author-model baseline adapters independently of legacy runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.runner import main

if __name__ == "__main__":
    main()
