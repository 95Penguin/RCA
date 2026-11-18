#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/ltfp_ablation}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
DATASET="${DATASET:-all}"
HORIZON_SECONDS="${HORIZON_SECONDS:-600}"

EPOCHS="${EPOCHS:-20}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WINDOW_SIZE="${WINDOW_SIZE:-24}"
STRIDE="${STRIDE:-6}"
MAX_FEATURES="${MAX_FEATURES:-180}"
DROPOUT="${DROPOUT:-0.12}"
MAX_NORMAL_RATIO="${MAX_NORMAL_RATIO:-3.0}"
HARD_NEGATIVE_SECONDS="${HARD_NEGATIVE_SECONDS:-1800}"

ALL_DATASETS=(re1-ob re1-ss re1-tt re2-ss re3-ob re3-ss re3-tt)
if [[ "$DATASET" == "all" ]]; then
  DATASETS=("${ALL_DATASETS[@]}")
else
  DATASETS=("$DATASET")
fi

case_root_for_dataset() {
  case "$1" in
    re1-ob) echo "$DATA_ROOT/RE1/RE1-OB" ;;
    re1-ss) echo "$DATA_ROOT/RE1/RE1-SS" ;;
    re1-tt) echo "$DATA_ROOT/RE1/RE1-TT" ;;
    re2-ss) echo "$DATA_ROOT/RE2/RE2-SS" ;;
    re3-ob) echo "$DATA_ROOT/RE3/RE3-OB" ;;
    re3-ss) echo "$DATA_ROOT/RE3/RE3-SS" ;;
    re3-tt) echo "$DATA_ROOT/RE3/RE3-TT" ;;
    *) echo "Unknown dataset: $1" >&2; return 1 ;;
  esac
}

run_variant() {
  local dataset="$1"
  local variant="$2"
  local case_root="$3"
  local output_dir="$OUTPUT_ROOT/$dataset/$variant"
  shift 3

  if [[ "${FORCE:-0}" != "1" && -f "$output_dir/rcaeval-cases_ltfp_tcn_gat_summary.json" ]]; then
    echo "[skip] $dataset/$variant already exists. Set FORCE=1 to rerun."
    return
  fi

  echo
  echo "========== LTFP ablation: $dataset / $variant =========="
  local command=(
    "$PYTHON" experiments/run_ltfp_model.py
    --dataset rcaeval-cases \
    --case-root "$case_root" \
    --output-dir "$output_dir" \
    --epochs "$EPOCHS" \
    --hidden-dim "$HIDDEN_DIM" \
    --batch-size "$BATCH_SIZE" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --max-raw-features "$MAX_FEATURES" \
    --dropout "$DROPOUT" \
    --use-pos-weight \
    --max-normal-ratio "$MAX_NORMAL_RATIO" \
    --hard-negative-seconds "$HARD_NEGATIVE_SECONDS" \
    "$@"
  )
  if [[ -n "${CASE_LIMIT:-}" ]]; then
    command+=(--case-limit "$CASE_LIMIT")
  fi
  "${command[@]}"
}

for dataset in "${DATASETS[@]}"; do
  case_root="$(case_root_for_dataset "$dataset")"

  run_variant "$dataset" full "$case_root" \
    --horizons "60,180,300,600,900" \
    --feature-selection balanced \
    --allowed-modalities metric,log,trace \
    --include-delta-features \
    --n-tcn-layers 3 \
    --n-graph-layers 2 \
    --graph-top-k 2 \
    --horizon-weight-mode short \
    --type-weight 0.8 \
    --root-weight 1.2 \
    --root-prior-weight 0.5 \
    --lead-weight 0.05 \
    --lead-reg-weight 0.0

  run_variant "$dataset" no_logs_traces "$case_root" \
    --horizons "60,180,300,600,900" \
    --feature-selection balanced \
    --allowed-modalities metric \
    --include-delta-features \
    --n-tcn-layers 3 \
    --n-graph-layers 2 \
    --graph-top-k 2 \
    --horizon-weight-mode short \
    --type-weight 0.8 \
    --root-weight 1.2 \
    --root-prior-weight 0.5 \
    --lead-weight 0.05 \
    --lead-reg-weight 0.0

  run_variant "$dataset" no_graph "$case_root" \
    --horizons "60,180,300,600,900" \
    --feature-selection balanced \
    --allowed-modalities metric,log,trace \
    --include-delta-features \
    --n-tcn-layers 3 \
    --n-graph-layers 0 \
    --graph-top-k 0 \
    --horizon-weight-mode short \
    --type-weight 0.8 \
    --root-weight 1.2 \
    --root-prior-weight 0.5 \
    --lead-weight 0.05 \
    --lead-reg-weight 0.0

  run_variant "$dataset" no_lead_time "$case_root" \
    --horizons "$HORIZON_SECONDS" \
    --feature-selection balanced \
    --allowed-modalities metric,log,trace \
    --include-delta-features \
    --n-tcn-layers 3 \
    --n-graph-layers 2 \
    --graph-top-k 2 \
    --horizon-weight-mode flat \
    --type-weight 0.8 \
    --root-weight 1.2 \
    --root-prior-weight 0.5 \
    --lead-weight 0.0 \
    --lead-reg-weight 0.0

  run_variant "$dataset" no_aux_tasks "$case_root" \
    --horizons "60,180,300,600,900" \
    --feature-selection balanced \
    --allowed-modalities metric,log,trace \
    --include-delta-features \
    --n-tcn-layers 3 \
    --n-graph-layers 2 \
    --graph-top-k 2 \
    --horizon-weight-mode short \
    --type-weight 0.0 \
    --root-weight 0.0 \
    --root-prior-weight 0.0 \
    --lead-weight 0.05 \
    --lead-reg-weight 0.0

  run_variant "$dataset" no_delta "$case_root" \
    --horizons "60,180,300,600,900" \
    --feature-selection balanced \
    --allowed-modalities metric,log,trace \
    --n-tcn-layers 3 \
    --n-graph-layers 2 \
    --graph-top-k 2 \
    --horizon-weight-mode short \
    --type-weight 0.8 \
    --root-weight 1.2 \
    --root-prior-weight 0.5 \
    --lead-weight 0.05 \
    --lead-reg-weight 0.0
done

"$PYTHON" scripts/summarize_ltfp_ablation.py --root "$OUTPUT_ROOT" --horizon "$HORIZON_SECONDS"
