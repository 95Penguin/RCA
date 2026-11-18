# LTFP fault-type auxiliary supervision feasibility

Date: 2026-11-16

## Question

Does an ordinary fault-type auxiliary classifier improve next-minute user-visible
failure prediction and root-cause ranking enough to justify a later prototype or
What--Where matching model?

This is a method feasibility screen, not a claim about final statistical
significance.

## Controlled method

- Backbone, failure head, RCA head, inputs, graph, split, seed, and optimizer are
  unchanged from `ltfp`.
- `ltfp-type-aux` adds one linear classifier on the shared application state.
- Fault-type cross-entropy is applied only when the exact future target minute has
  a positive user-visible failure label. Normal windows are not assigned their
  case's injected fault type.
- Positive-window class weights are fitted on the training split only.
- A type absent from a training fold receives zero auxiliary-loss weight; it is
  still retained in test reporting.
- Seed: 42; epochs: 30; history: 5 minutes; horizon: exactly t+1 minute.

## Fold-0 screening results

### Configuration holdout

| Dataset | Model | Type weight | AP | F1 | Hit@1 | MRR | Type accuracy | Type macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RE2-OB | LTFP | 0 | 0.9470 | 0.8366 | 0.5417 | 0.6968 | - | - |
| RE2-OB | Type auxiliary | 0.1 | 0.9346 | 0.8671 | 0.5694 | 0.7806 | 0.0278 | 0.0381 |
| RE2-OB | Type auxiliary | 0.5 | 0.9461 | 0.8148 | 0.5000 | 0.6759 | 0.0556 | 0.0503 |
| RE2-TT | LTFP | 0 | 0.8167 | 0.7286 | 0.7500 | 0.8362 | - | - |
| RE2-TT | Type auxiliary | 0.1 | 0.8063 | 0.7125 | 0.7955 | 0.8674 | 0.5000 | 0.2534 |
| RE2-TT | Type auxiliary | 0.5 | 0.7781 | 0.7152 | 0.7614 | 0.8551 | 0.3750 | 0.1886 |

Some held-out configurations have no positive user-impact window for particular
fault types. Type macro-F1 therefore averages classes with positive test support;
the per-class support remains available in `metrics.json`.

### Deployment holdout

| Dataset | Model | Type weight | AP | F1 | Hit@1 | MRR | Type accuracy | Type macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RE2-OB | LTFP | 0 | 0.8930 | 0.7368 | 0.9412 | 0.9635 | - | - |
| RE2-OB | Type auxiliary | 0.1 | 0.8542 | 0.6638 | 0.8039 | 0.8698 | 0.3595 | 0.3472 |
| RE2-OB | Type auxiliary | 0.5 | 0.7416 | 0.5395 | 0.6601 | 0.7516 | 0.3595 | 0.3476 |
| RE2-TT | LTFP | 0 | 0.6224 | 0.5616 | 0.6197 | 0.7312 | - | - |
| RE2-TT | Type auxiliary | 0.1 | 0.5974 | 0.5755 | 0.6197 | 0.7207 | 0.2817 | 0.3040 |
| RE2-TT | Type auxiliary | 0.5 | 0.6105 | 0.5693 | 0.6479 | 0.7723 | 0.4930 | 0.4839 |

## Decision

**NO-GO for ordinary fault-type auxiliary supervision as the basis of a
prototype/What--Where paper direction.**

The weaker 0.1 loss reduces the damage but lowers AP in all four dataset/split
settings. Improvements in individual F1 or RCA metrics are inconsistent and are
offset by regressions elsewhere. RE2-OB configuration holdout also shows almost
no transferable type recognition, directly weakening the assumption that a
simple shared fault mechanism can be learned independently of service position.

This screen does not prove that all mechanism-aware models are impossible. It
does show that adding prototypes on top of this failed auxiliary signal would be
poorly justified. A different direction should first target an observed error
mode of the primary tasks rather than add a more elaborate type classifier.

## Reproduction

Example configuration-holdout run:

```bash
.venv/bin/python scripts/run_re2.py train \
  --data-dir artifacts/re2_ob_panel_v1 \
  --output-dir /tmp/re2_ob_ltfp_type_aux_config_fold0 \
  --model ltfp-type-aux \
  --fault-type-loss-weight 0.1 \
  --epochs 30 \
  --device cpu \
  --split-policy configuration \
  --split-fold 0
```

