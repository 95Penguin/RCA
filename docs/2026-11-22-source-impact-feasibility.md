# Source--impact factorization feasibility audit

Date: 2026-11-22

## Question

The audit tests a narrower claim before changing LTFP:

1. Is the injected fault source identifiable even when the injection causes no
   user-visible business impact?
2. For next-minute failure prediction, is it useful to model source evidence
   and business-impact evidence separately and then combine them?

This is a feasibility probe, not a candidate final model. All normal references,
standardizers, and linear probes are fitted on the training partition of each
deployment fold. The reported results pool the five held-out test folds.

## Probe construction

- **Source evidence:** per-service robust deviations of CPU, memory, disk I/O,
  and socket metrics from a training-only pre-injection median/MAD reference.
- **Impact evidence:** causal five-minute summaries of entry latency and
  degradation, entry volume, trace duration/count/errors, and log count/errors.
- **Source-only:** predicts next-minute impact using only source evidence.
- **Impact-only:** predicts next-minute impact using only impact evidence.
- **Additive:** concatenates source and impact evidence.
- **Factorized:** additionally includes explicit source-strength × impact-feature
  interactions.
- **Comparator:** the existing five-fold LTFP baseline predictions, with the
  same deployment splits and timestamps.

For source localization, `phase=post` is used as the active-injection proxy.
This is suitable for a diagnostic but should not be presented as a new official
benchmark label without documenting the original injection interval for both
datasets.

## Results

### Source localization by injected-case type

| Dataset | Cases | Group | Raw deviation Hit@1 | Raw deviation MRR | Learned ranker Hit@1 | Learned ranker MRR |
|---|---:|---|---:|---:|---:|---:|
| RE2-OB | 89 | all | 0.820 | 0.854 | 0.798 | 0.843 |
| RE2-OB | 51 | impact | 0.745 | 0.794 | 0.725 | 0.780 |
| RE2-OB | 38 | no impact | **0.921** | **0.934** | 0.895 | 0.927 |
| RE2-TT | 89 | all | 0.809 | 0.856 | 0.775 | 0.823 |
| RE2-TT | 53 | impact | **0.868** | **0.908** | 0.811 | 0.859 |
| RE2-TT | 36 | no impact | 0.722 | 0.780 | 0.722 | 0.770 |

The source is therefore not merely a proxy for user-visible impact. It remains
identifiable in most no-impact injections. The learned shared linear ranker does
not improve over the raw normal-state deviation, so a more complex source head
is not justified by this probe.

### Next-minute failure prediction

| Dataset | Method | AP | F1 at 0.5 | FP rate on no-impact post windows |
|---|---|---:|---:|---:|
| RE2-OB | source-only | 0.480 | 0.473 | 0.847 |
| RE2-OB | impact-only | 0.820 | 0.711 | 0.243 |
| RE2-OB | additive | 0.828 | 0.720 | 0.284 |
| RE2-OB | factorized | 0.849 | 0.725 | 0.250 |
| RE2-OB | existing LTFP | **0.881** | **0.833** | **0.097** |
| RE2-TT | source-only | 0.401 | 0.445 | 0.715 |
| RE2-TT | impact-only | 0.705 | 0.684 | 0.247 |
| RE2-TT | additive | 0.693 | 0.688 | 0.288 |
| RE2-TT | factorized | 0.689 | 0.692 | 0.245 |
| RE2-TT | existing LTFP | **0.806** | **0.785** | **0.111** |

The source-only failure probe is expectedly bad: a real injected fault is not
equivalent to a business failure. The explicit factorized interaction helps over
impact-only on OB (+0.029 AP), but hurts on TT (-0.016 AP), and is below LTFP on
both datasets.

Per-fold AP confirms that the aggregate result is not hiding a stable win:

| Dataset | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---:|---:|---:|---:|---:|
| RE2-OB | factorized | 0.950 | 0.915 | 0.871 | 0.815 | 0.913 |
| RE2-OB | LTFP | 0.893 | 0.942 | 0.896 | 0.912 | 0.951 |
| RE2-TT | factorized | 0.420 | 0.754 | 0.793 | 0.775 | 0.935 |
| RE2-TT | LTFP | 0.622 | 0.939 | 0.900 | 0.887 | 0.881 |

## Decision

The semantic hypothesis is supported: **fault source and business impact are
different variables**. The proposed direct product/feature-interaction model is
not supported as a replacement for LTFP.

Do not proceed with a large `q_i × a_i` architecture on the evidence above. A
smaller, defensible next experiment is an **auxiliary source head** trained on
all active injections, while keeping the failure head impact-driven and blocking
the source signal from gating the failure probability. That experiment would be
accepted only if it preserves failure AP/F1 and improves RCA; otherwise the
factorization direction should be closed.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/audit_source_impact_factorization.py \
  --output artifacts/source_impact_feasibility.json
```
