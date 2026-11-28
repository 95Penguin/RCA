"""Resolve the source checkout when archived inside its experiments folder."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p/'src/ocsid/cpca.py').is_file()),
            Path(os.environ.get('NEXTMINUTE_RCA_ROOT','/Users/95penguin/project/NextMinuteRCA')))
EXPERIMENT = ROOT/'experiments/cpca_counterfactual_20261126'
