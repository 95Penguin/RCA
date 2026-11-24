# Directed graph StageRoute experiment

## Decision

Do not replace the symmetric graph with one universal directed graph.  Across OB
and TT together, the symmetric StageRoute remains the strongest choice.  However,
direction is not uniformly useless: on TT, callee-to-caller message flow improves
both mean AP and mean window MRR over the symmetric graph.

The result supports retaining caller/callee direction as relational information,
not deleting one direction globally.  A future model may encode incoming and
outgoing calls as separate relations while preserving both.

## Direction semantics

Raw trace extraction stores a caller-row matrix `D[caller, callee]`.  The graph
attention layer aggregates matrix columns into each row, so:

- `symmetric`: use `D + D^T`;
- `callee-to-caller`: use `D`; caller nodes aggregate callee states;
- `caller-to-callee`: use `D^T`; callee nodes aggregate caller states.

All variants use the same StageRoute model, parameter count, telemetry, labels,
deployment folds, optimizer, loss weights, and seed.  Only the selected adjacency
array differs.

## Graph reconstruction audit

The directed arrays were regenerated from the original RE2 trace archives.
Before training, the builder required `D + D^T` to be exactly equal, element by
element, to the existing symmetric panel graph.

| Dataset | Cases | Resolved cross-service calls | Minutes | Services | Exact reconstruction |
| --- | ---: | ---: | ---: | ---: | --- |
| OB | 90 | 14,517,612 | 2,144 | 11 | yes |
| TT | 90 | 11,840,032 | 2,160 | 27 | yes |

The symmetric 2-epoch smoke runs also exactly reproduced the prior StageRoute
smoke AP (`0.780` OB and `0.448` TT), providing an end-to-end control against
panel drift.

## Five-fold means

| Dataset | Message flow | AP | F1@0.5 | Window Hit@1 | Window MRR | Event Hit@1 | Event MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OB | symmetric | **0.909** | 0.811 | **0.888** | **0.932** | **0.911** | **0.933** |
| OB | callee-to-caller | 0.845 | 0.640 | 0.602 | 0.772 | 0.588 | 0.757 |
| OB | caller-to-callee | 0.904 | **0.853** | 0.760 | 0.859 | 0.801 | 0.879 |
| TT | symmetric | 0.828 | 0.739 | 0.858 | 0.907 | 0.894 | 0.936 |
| TT | callee-to-caller | **0.836** | **0.744** | 0.866 | 0.924 | 0.855 | 0.920 |
| TT | caller-to-callee | 0.805 | 0.734 | **0.886** | **0.925** | **0.922** | **0.949** |

For TT, callee-to-caller is the only direction that improves both AP (`+0.008`)
and the primary window MRR (`+0.018`).  Caller-to-callee improves localization
more strongly but loses AP (`-0.024`).

Across the two dataset means:

| Message flow | AP | Window MRR |
| --- | ---: | ---: |
| symmetric | **0.869** | **0.919** |
| callee-to-caller | 0.841 | 0.848 |
| caller-to-callee | 0.854 | 0.892 |

Thus neither single direction is a cross-dataset replacement for the symmetric
graph.

## Interpretation

TT previously showed stronger temporal root-before-neighbor evidence than OB,
and it is also the only dataset where a directed flow improves the aggregate
prediction/window-RCA pair.  OB contains many same-minute root and neighbor
deviations; removing either direction discards useful context and substantially
damages RCA.

The exact service topology should still not be treated as proven causal evidence.
Earlier symmetric-graph permutation experiments found that mismatched topology
could reproduce some gains.  This experiment establishes sensitivity to message
direction under retraining, not causal propagation identification.

## Implementation and reproduction

- `src/data/re2.py` now preserves symmetric, caller-row, and callee-row arrays
  for newly prepared panels and lets `Re2WindowDataset` select message flow.
- `scripts/build_re2_directed_graph_panel.py` reconstructs directed arrays and
  performs the exact symmetric-graph check.
- `scripts/run_re2.py --graph-direction ...` records the selected direction.
- `scripts/run_directed_stage_route_experiment.py` runs the controlled matrix.

Current generated panels:
`/private/tmp/nextminuterca_directed_panels`.

Current run outputs:
`/private/tmp/nextminuterca_directed_stage_route_30ep_seed42`.
