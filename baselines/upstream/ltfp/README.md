# LTFP

Official implementation of **LTFP: Lead-Time-Aware Failure Prediction Based on Service GNNs for AIOps**.

LTFP predicts whether a cloud-native service failure will occur within one or more future lead-time windows and ranks the services most likely to be responsible. The implementation combines modality-specific temporal encoders, gated multimodal fusion, a sparse hybrid service graph, edge-weight-aware graph attention, and joint failure-prediction/localization heads.

## What is included

- Causal alignment and window construction for metrics, logs, and traces.
- Service-balanced and modality-balanced feature selection.
- Modality-specific temporal encoders and gated fusion.
- Hybrid graph construction from known dependencies and training-only correlations.
- Edge-weight-aware graph attention.
- Multi-horizon failure prediction and responsible-service ranking.
- Failure-reference-time construction, reviewed-label manifests, and leakage-controlled splits.
- Training, inference, ablation, evaluation, and unit-test code for LTFP.

This repository intentionally does **not** include datasets, pretrained checkpoints, experiment outputs, third-party code, or baseline implementations.

## Repository layout

```text
src/ltfp/                      LTFP model and data-processing library
experiments/run_ltfp_model.py Main paper-oriented training/evaluation entry point
experiments/run_failure_onset_v2.py
                               Strict failure-onset protocol runner
scripts/                       LTFP-only preprocessing and experiment utilities
tests/                         Unit tests using generated temporary data
```

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The implementation runs on CPU by default. A CUDA-enabled PyTorch installation can be used when available.

## Dataset interface

No dataset is bundled. The loaders accept an RCAEval-style directory in which each physical case contains:

- `metrics.json`, `data.csv`, `simple_data.csv`, `simple_metrics.csv`, or `metrics.csv`;
- `inject_time.txt` when the fault-activation time is available;
- optional `logs.csv`/`logts.csv` and `traces.csv`/`tracets_*.csv` files.

For the strict post-activation, pre-failure protocol, provide a finalized failure-onset label manifest, a case-level split manifest, and a topology manifest. Pending or incomplete onset labels are rejected by the loader.

## Quick smoke test without external data

The synthetic generator is code-only and is provided solely to verify that the pipeline runs without downloading a dataset:

```bash
python experiments/run_ltfp_model.py \
  --dataset synthetic \
  --synthetic-cases 24 \
  --epochs 2 \
  --hidden-dim 16 \
  --n-tcn-layers 1 \
  --n-graph-layers 1 \
  --batch-size 16 \
  --output-dir results/smoke
```

## Run LTFP on an external case directory

```bash
python experiments/run_ltfp_model.py \
  --dataset rcaeval-cases \
  --case-root /path/to/cases \
  --horizons 60,180,300,600,900 \
  --window-size 24 \
  --stride 6 \
  --max-raw-features 180 \
  --feature-selection balanced \
  --include-delta-features \
  --hidden-dim 32 \
  --n-tcn-layers 3 \
  --n-graph-layers 2 \
  --graph-top-k 2 \
  --dropout 0.12 \
  --batch-size 32 \
  --epochs 20 \
  --output-dir results/ltfp
```

All output paths are ignored by Git.

## Strict failure-onset workflow

The strict runner separates fault activation from the later failure reference time and requires reviewed labels.

1. Build candidate failure-onset labels:

```bash
python scripts/build_failure_onset_manifest.py \
  --case-root /path/to/cases \
  --output /path/to/labels.candidate.json
```

2. Finalize the label manifest after independent review, then create the case-level split manifest:

```bash
python scripts/build_split_manifest.py \
  --label-manifest /path/to/labels.accepted.json \
  --output /path/to/splits.csv
```

3. Train one fold:

```bash
python experiments/run_failure_onset_v2.py \
  --case-root /path/to/cases \
  --label-manifest /path/to/labels.accepted.json \
  --split-manifest /path/to/splits.csv \
  --topology-manifest /path/to/topology.json \
  --fold 0 \
  --mode train \
  --output-dir results/failure_onset
```

## Tests

```bash
python -m pytest -q
```

The tests create their own temporary inputs and do not require any external dataset.
