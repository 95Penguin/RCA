#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-custom}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/ltfp}"
PYTHON="${PYTHON:-python3}"

if [[ -z "${CASE_ROOT:-}" ]]; then
  echo "Set CASE_ROOT=/path/to/cases before running this script." >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$DATASET}"
EPOCHS="${EPOCHS:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WINDOW_SIZE="${WINDOW_SIZE:-24}"
STRIDE="${STRIDE:-6}"
MAX_FEATURES="${MAX_FEATURES:-180}"
HORIZONS="${HORIZONS:-60,180,300,600,900}"
COMMAND=(
  "$PYTHON" experiments/run_ltfp_model.py
  --dataset rcaeval-cases
  --case-root "$CASE_ROOT"
  --horizons "$HORIZONS"
  --epochs "$EPOCHS"
  --hidden-dim "$HIDDEN_DIM"
  --n-tcn-layers "${TCN_LAYERS:-3}"
  --n-graph-layers "${GRAPH_LAYERS:-2}"
  --dropout "${DROPOUT:-0.12}"
  --batch-size "$BATCH_SIZE"
  --window-size "$WINDOW_SIZE"
  --stride "$STRIDE"
  --max-raw-features "$MAX_FEATURES"
  --feature-selection "${FEATURE_SELECTION:-balanced}"
  --use-pos-weight
  --max-normal-ratio "${MAX_NORMAL_RATIO:-3.0}"
  --hard-negative-seconds "${HARD_NEGATIVE_SECONDS:-1800}"
  --horizon-weight-mode "${HORIZON_WEIGHT_MODE:-short}"
  --type-weight "${TYPE_WEIGHT:-0.8}"
  --root-weight "${ROOT_WEIGHT:-1.2}"
  --root-prior-weight "${ROOT_PRIOR_WEIGHT:-0.5}"
  --lead-weight "${LEAD_WEIGHT:-0.05}"
  --lead-reg-weight "${LEAD_REG_WEIGHT:-0.0}"
  --focus-weight "${FOCUS_WEIGHT:-1.0}"
  --score-fusion "${SCORE_FUSION:-none}"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "${FOCUS_HORIZON:-}" ]]; then
  COMMAND+=(--focus-horizon "$FOCUS_HORIZON")
fi
if [[ -n "${CASE_LIMIT:-}" ]]; then
  COMMAND+=(--case-limit "$CASE_LIMIT")
fi
if [[ "${INCLUDE_DELTA_FEATURES:-0}" == "1" ]]; then
  COMMAND+=(--include-delta-features)
fi

"${COMMAND[@]}"
