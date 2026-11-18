#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/r2c2_graph_study}"
DATA_ROOT="${DATA_ROOT:-data}"
TOPOLOGY_DIR="${TOPOLOGY_DIR:-manifests/topology}"
DATASET="${DATASET:-all}"
MODES=(hybrid-max dependency-only correlation-only hybrid-learned hybrid-lagged hybrid-random)
DATASETS=(re1-ob re1-ss re1-tt re2-ss re3-ob re3-ss re3-tt)

if [[ "$DATASET" != "all" ]]; then
  DATASETS=("$DATASET")
fi

case_root() {
  case "$1" in
    re1-ob) echo "$DATA_ROOT/RE1/RE1-OB" ;;
    re1-ss) echo "$DATA_ROOT/RE1/RE1-SS" ;;
    re1-tt) echo "$DATA_ROOT/RE1/RE1-TT" ;;
    re2-ss) echo "$DATA_ROOT/RE2/RE2-SS" ;;
    re3-ob) echo "$DATA_ROOT/RE3/RE3-OB" ;;
    re3-ss) echo "$DATA_ROOT/RE3/RE3-SS" ;;
    re3-tt) echo "$DATA_ROOT/RE3/RE3-TT" ;;
  esac
}

topology_manifest() {
  case "$1" in
    *-ob) echo "$TOPOLOGY_DIR/online-boutique.json" ;;
    *-ss) echo "$TOPOLOGY_DIR/sock-shop.json" ;;
    *-tt) echo "$TOPOLOGY_DIR/train-ticket.json" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    output_dir="$OUTPUT_ROOT/$dataset/$mode"
    summary="$output_dir/rcaeval-cases_ltfp_tcn_gat_summary.json"
    if [[ "${FORCE:-0}" != "1" && -f "$summary" ]]; then
      echo "[skip] $dataset/$mode"
      continue
    fi
    echo "[run] $dataset/$mode"
    OMP_NUM_THREADS=4 "$PYTHON" experiments/run_ltfp_model.py \
      --dataset rcaeval-cases \
      --case-root "$(case_root "$dataset")" \
      --case-limit 30 \
      --output-dir "$output_dir" \
      --dataset-cache "$OUTPUT_ROOT/cache/$dataset.npz" \
      --horizons 60,180,300,600,900 \
      --window-size 24 \
      --stride 6 \
      --seed 7 \
      --epochs 5 \
      --hidden-dim 16 \
      --batch-size 32 \
      --max-raw-features 80 \
      --feature-selection balanced \
      --allowed-modalities metric,log,trace \
      --include-delta-features \
      --n-tcn-layers 3 \
      --n-graph-layers 2 \
      --graph-top-k 2 \
      --graph-lag-steps 3 \
      --graph-mode "$mode" \
      --topology-manifest "$(topology_manifest "$dataset")" \
      --dropout 0.12 \
      --use-pos-weight \
      --max-normal-ratio 3.0 \
      --hard-negative-seconds 1800 \
      --horizon-weight-mode short \
      --type-weight 0.8 \
      --root-weight 1.2 \
      --root-prior-weight 0.5 \
      --lead-weight 0.05 \
      --lead-reg-weight 0.0
  done
done

"$PYTHON" scripts/summarize_r2c2_graph_study.py --root "$OUTPUT_ROOT"
"$PYTHON" scripts/analyze_r2c2_runtime_edges.py --root "$OUTPUT_ROOT"
