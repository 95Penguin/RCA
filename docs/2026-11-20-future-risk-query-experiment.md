# Temporal future-risk query: five-fold result

## Decision

Do not promote the current temporal future-risk query to the main model.  It is
a useful capacity-matched negative ablation: directly replacing the RCA head's
graph-level application context with the pre-propagation temporal prediction
state does not improve localization and substantially damages OB prediction.

This result does not reject all prediction-conditioned RCA.  It rejects the
specific hard context replacement used here.  Adding projections, gates, or
contrastive losses now would introduce several simultaneous explanations and is
not justified by this result.

## Controlled change

`ltfp-stage-route-query` has exactly the same parameters and failure forward path
as `ltfp-stage-route`:

- StageRoute RCA context: graph-pooled application state.
- FutureQuery RCA context: pre-propagation temporal application state used by
  the failure head.
- Service embeddings: graph-propagated in both variants.
- Failure path, losses, parameter count, and training protocol: unchanged.

Although the failure forward path is identical under equal weights, the changed
RCA loss path alters the jointly trained temporal backbone.  This explains why
failure AP can change after training.

## Five-fold means

| Dataset | Model | AP | F1@0.5 | Window MRR | Event MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| OB | LTFP | 0.919 | 0.837 | 0.922 | 0.878 |
| OB | StageRoute | 0.909 | 0.811 | 0.932 | 0.933 |
| OB | StageRoute + FutureQuery | 0.882 | 0.730 | 0.907 | 0.907 |
| TT | LTFP | 0.846 | 0.771 | 0.913 | 0.932 |
| TT | StageRoute | 0.828 | 0.739 | 0.907 | 0.936 |
| TT | StageRoute + FutureQuery | 0.828 | 0.740 | 0.903 | 0.924 |

Relative to StageRoute, FutureQuery changes:

- OB: AP `-0.027`, window MRR `-0.024`, event MRR `-0.026`.
- TT: AP approximately unchanged, window MRR `-0.004`, event MRR `-0.012`.

## Fold-level AP and event MRR

| Dataset | Fold | Stage AP | Query AP | Stage event MRR | Query event MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| OB | 0 | 0.866 | 0.837 | 1.000 | 0.806 |
| OB | 1 | 0.965 | 0.966 | 0.962 | 0.923 |
| OB | 2 | 0.853 | 0.734 | 1.000 | 1.000 |
| OB | 3 | 0.897 | 0.908 | 0.848 | 0.857 |
| OB | 4 | 0.963 | 0.962 | 0.853 | 0.950 |
| TT | 0 | 0.617 | 0.603 | 0.730 | 0.687 |
| TT | 1 | 0.940 | 0.901 | 1.000 | 1.000 |
| TT | 2 | 0.899 | 0.864 | 1.000 | 0.962 |
| TT | 3 | 0.885 | 0.860 | 0.948 | 0.969 |
| TT | 4 | 0.800 | 0.914 | 1.000 | 1.000 |

Some folds improve, especially TT fold 4 AP and OB fold 4 event MRR, but the
aggregate deterioration is not caused by requiring every fold to win.

## Interpretation

The temporal application state is optimized to separate future system failure,
not to preserve service identity.  Using it as the sole RCA context can favor
services aligned with system impact or propagation symptoms rather than the
injected root.  The graph-level context appears to be a better match for the
graph-propagated service embeddings used by the root head.

The next graph experiment should keep StageRoute as the candidate architecture
and test direction as a controlled input change: symmetric adjacency, caller-row
adjacency, and its transpose.  It must preserve parameter count and retrain each
variant.  Existing symmetric-topology permutation results warn that using a graph
does not prove that the correct service relationship is responsible for gains.

## Reproduction

- Model: `LTFPStageRouteFutureQuery` in `src/models/ltfp.py`.
- Runner: `scripts/run_stage_route_experiment.py --models ltfp-stage-route-query`.
- Tests: `tests/test_stage_route.py` verifies equal parameter counts and identical
  failure logits under equal weights.
- Current run output:
  `/private/tmp/nextminuterca_stage_route_query_30ep_seed42`.
