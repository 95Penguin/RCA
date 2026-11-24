# DeST-inspired stage routing: five-fold feasibility experiment

## Decision

Keep `ltfp-stage-route` as a Pareto candidate, but do not call it a final win yet.
The model trades a small amount of failure-prediction AP for better OB root-cause
ranking.  TT prediction remains close to the shared LTFP, while its RCA is
effectively tied.  Requiring every fold to improve would incorrectly reject the
OB result; the aggregate result is the relevant screening evidence.

The learned spatial correction gate is almost zero in every selected checkpoint.
The useful structural change is therefore the asymmetric route itself: failure
prediction is learned mainly from pre-propagation temporal states, while RCA keeps
the graph-propagated service states.

## Model

The baseline uses graph-propagated application state for failure prediction and
graph-propagated service/application states for RCA.  Stage routing changes only
the failure path:

1. Encode each service history into pre-propagation `local_state`.
2. Pool `local_state` into a temporal application state and predict failure.
3. Run the unchanged GAT and unchanged RCA head.
4. Let the graph-induced state difference add a bounded binary-logit residual.
5. Initialize the scalar spatial-residual gate at zero.

No new GAT, modality router, auxiliary label, uncertainty loss, or raw-deviation
path is introduced.

## Protocol

- Datasets: RE2-OB and RE2-TT fair panels.
- Task: exact next-minute failure state plus root-service ranking.
- Split: five deployment folds.
- History: five complete minutes.
- Training: 30 epochs, seed 42, batch size 64, learning rate 0.001.
- Root supervision: ranking-loss weight 0.5.
- Baseline: same-protocol integrated LTFP modality-all checkpoints.
- Checkpoint selection: existing validation policy.

## Fold results

The RCA column below is retrospective event-level MRR, matching the existing
integrated experiment summary.  Window-level RCA is reported separately.

| Dataset | Fold | LTFP AP | Stage AP | LTFP event MRR | Stage event MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| OB | 0 | 0.893 | 0.866 | 0.931 | 1.000 |
| OB | 1 | 0.942 | 0.965 | 0.923 | 0.962 |
| OB | 2 | 0.896 | 0.853 | 1.000 | 1.000 |
| OB | 3 | 0.912 | 0.897 | 0.688 | 0.848 |
| OB | 4 | 0.951 | 0.963 | 0.850 | 0.853 |
| TT | 0 | 0.622 | 0.617 | 0.700 | 0.730 |
| TT | 1 | 0.939 | 0.940 | 1.000 | 1.000 |
| TT | 2 | 0.900 | 0.899 | 1.000 | 1.000 |
| TT | 3 | 0.887 | 0.885 | 0.958 | 0.948 |
| TT | 4 | 0.881 | 0.800 | 1.000 | 1.000 |

## Five-fold means

| Dataset | Model | AP | F1@0.5 | Window Hit@1 | Window MRR | Event Hit@1 | Event MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OB | LTFP | 0.919 | 0.837 | 0.872 | 0.922 | 0.807 | 0.878 |
| OB | StageRoute | 0.909 | 0.811 | 0.888 | 0.932 | 0.911 | 0.933 |
| TT | LTFP | 0.846 | 0.771 | 0.865 | 0.913 | 0.894 | 0.932 |
| TT | StageRoute | 0.828 | 0.739 | 0.858 | 0.907 | 0.894 | 0.936 |

Across the two dataset means, StageRoute changes AP by `-0.014`, window MRR by
`+0.002`, and event MRR by `+0.029`.  Thus the strongest gain is event-level OB
localization; the operational window-level aggregate is essentially tied.

## Learned spatial correction

The effective scalar correction coefficient is
`0.5 * tanh(spatial_residual_scale)`.  Its selected-checkpoint values were:

- OB: `+0.0130`, `+0.0095`, `-0.0093`, `-0.0095`, `-0.0028`.
- TT: `-0.0003`, `-0.0069`, `-0.0076`, `-0.0079`, `+0.0013`.

These values are too small to claim that bounded spatial correction caused the
metric change.  The experiment mainly supports stage-asymmetric representation
use, not a learned graph correction to failure logits.

## Baseline context

Against the separate full baseline comparison, StageRoute remains stronger than
the non-LTFP prediction baselines on AP and stronger than the listed localization
baselines on window MRR.  That comparison uses adapted upstream implementations,
so it is contextual evidence only; the integrated LTFP rows above are the direct
controlled ablation.

## Next experiment

The next controlled addition should target per-window RCA without changing the
failure path: replace the RCA head's graph-level context with the temporal future-
risk state, or add a capacity-matched temporal-context control.  This tests the
forecast-specific claim that the root service should explain future system risk.
Do not add DMCN, uncertainty, and raw metric deviation in the same experiment.

Before making final claims, repeat the selected comparison with additional seeds.
The current result is one-seed five-fold evidence and the AP/window-MRR differences
are small enough that seed variance may change the ordering.

## Reproduction

Implementation:

- `src/models/ltfp.py`: `LTFPStageRoute`
- `scripts/run_stage_route_experiment.py`
- `tests/test_stage_route.py`

The current sandbox stored run outputs under
`/private/tmp/nextminuterca_stage_route_30ep_seed42`; the conclusion-level values
are retained in this document.
