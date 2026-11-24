# Direction-aware residual experiment

## Decision

Reject the direction-aware residual as a useful LTFP StageRoute extension.  It
does not improve the two-dataset aggregate, its learned gate remains close to
zero, and a same-checkpoint gate ablation shows that it changes no event-level
ranking and only one window-level fold, where the change is negative.

This result is stronger than a comparison of independently trained models: the
post-hoc ablation holds every learned parameter fixed and changes only the
direction gate.  The apparently strong individual-fold scores therefore come
from the symmetric StageRoute backbone, not from direction-aware propagation.

## Model tested

The experiment supplied the observed caller-row adjacency `D` to the model and
used three shared-weight graph passes:

1. Symmetric backbone: `H_sym = GAT(H, max(D, D^T))`.
2. Callee-to-caller pass: `H_forward = GAT(H, D)`.
3. Caller-to-callee pass: `H_reverse = GAT(H, D^T)`.
4. RCA state: `H_root = H_sym + 0.5 * tanh(gamma) *
   tanh(H_forward - H_reverse)`.

The scalar `gamma` starts at zero.  Only RCA consumes `H_root`; failure
prediction continues to use the temporal StageRoute path and its symmetric
spatial correction.  All graph passes share parameters, so the variant adds
only one trainable scalar to StageRoute.

At initialization, the variant is numerically identical to symmetric
StageRoute.  Unit tests also verify that reversing the input graph cannot change
the failure logits.

## Protocol

- Datasets: RE2-OB and RE2-TT.
- Task: exact next-minute failure state and root-service ranking.
- Split: five deployment folds.
- Training: 30 epochs, seed 42, batch size 64, learning rate 0.001.
- Root supervision: ranking-loss weight 0.5.
- Input direction: caller-row adjacency, which sends callee states to callers.
- Graph audit: `D + D^T` exactly reconstructs the existing symmetric panel.

## Five-fold means

| Dataset | Model | AP | Window MRR | Event MRR |
| --- | --- | ---: | ---: | ---: |
| OB | symmetric StageRoute | **0.909** | **0.932** | **0.933** |
| OB | directional residual | 0.886 | 0.921 | 0.932 |
| TT | symmetric StageRoute | **0.828** | 0.907 | **0.936** |
| TT | directional residual | 0.804 | **0.911** | 0.932 |

Across the two dataset means, the directional variant changes AP from `0.869`
to `0.845`, window MRR from `0.919` to `0.916`, and event MRR from about `0.935`
to `0.932`.  TT window MRR gains only `0.004`, while its AP loses `0.024`.

The failure path does not directly consume the directional residual.  Its AP
difference across independent training runs reflects the changed joint-training
trajectory, including shared temporal parameters and dropout, rather than a
directional inference contribution.

## Learned direction gates

The effective gate is `0.5 * tanh(gamma)`:

- OB: `+0.0017`, `-0.0169`, `-0.0015`, `+0.0121`, `+0.0090`.
- TT: `-0.0122`, `+0.0069`, `-0.0095`, `-0.0258`, `-0.0083`.

The magnitude is small relative to the allowed `0.5`, and the preferred sign
is inconsistent across folds.

## Same-checkpoint gate ablation

Every selected checkpoint was evaluated twice on the identical test split:
once with its learned direction gate and once with `gamma` forced to zero.

- Event MRR changed by exactly `0` in all ten folds.
- Window MRR changed by exactly `0` in nine folds.
- OB fold 4 was the only changed window ranking: the learned direction residual
  reduced MRR by `0.001096`.

Thus the direction branch did not cause any observed ranking improvement.  The
fixed-direction experiment remains useful evidence that TT is direction
sensitive, but this conservative adaptive residual does not exploit that signal.
Do not add node-wise or edge-wise gates to rescue this branch without a new,
independently motivated hypothesis.

## Reproduction

- `src/models/ltfp.py`: `LTFPStageRouteDirectionalResidual`.
- `src/models/factory.py`: `ltfp-stage-route-directional`.
- `scripts/run_directional_residual_experiment.py`: five-fold runner and gate
  extraction.
- `tests/test_stage_route.py`: zero-gate equivalence, prediction invariance,
  and gate-gradient checks.

Current generated outputs are under
`/private/tmp/nextminuterca_directional_residual`; the post-hoc gate ablation is
stored in `directional_gate_ablation.json` there.
