"""Repository-relative paths for the counterfactual experiment snapshot."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VALIDATION = OUT / "inputs"
