# RE2-OB next-minute prediction audit

This audit uses the same RE2-OB labels, 5-minute windows, repetition split,
and seed for GCN+GRU and GRU. Checkpoints are selected by validation loss.
The audit script evaluates held-out validation and test windows, selects a
classification threshold on validation F1 only, and writes event details to
`artifacts/re2_ob_onset_audit_v1.json` and `artifacts/re2_ob_gru_audit_v1.json`.

| Test measure | GCN+GRU | GRU | Current-minute latency rule |
| --- | ---: | ---: | ---: |
| Average precision | 0.804 | 0.731 | 0.818 |
| F1 at probability 0.5 | 0.603 | 0.365 | 0.883 |
| F1 at validation-selected threshold | 0.739 (0.360) | 0.688 (0.466) | 0.883 |
| First failure minute recall at 0.5 | 1/20 | 1/20 | 0/20 |
| First failure minute recall at selected threshold | 4/20 | 11/20 | 0/20 |

The graph model improves aggregate AP and F1 over this GRU run, but the GRU
has higher onset recall at its own validation-selected threshold. Its broader
alerting also produces more false positives. The persistence rule is strongest
on all positive minutes: it detects 136/136 ongoing test minutes. It cannot
detect the first high-latency minute by construction. Of 156 positive test
windows, 136 represent ongoing degradation and only 20 are event onsets.

## Timing boundary

For 16/20 test event onsets, injection occurs **inside** the final input
minute. For the other four, injection occurs before that minute. No onset
example has an input window wholly before injection. Thus the onset measure
tests prediction of the next complete minute using telemetry from an injection
minute or later. It is **not** evidence of pre-injection forecasting. The
minute containing injection was excluded as a *target* by the label builder,
but can still be included as an *input* for the first post-injection target.

The current repetition split holds each recorded case apart, but the same
service/fault configuration appears in all three repetitions. It measures
generalization across repetitions of known configurations, not unseen fault
configurations. Do not claim unseen-fault transfer from these numbers.

## Configuration holdout

We also ran five configuration-held-out folds. Each fold trains on 18
service/fault configurations, validates on 6, and tests on 6. All three
repetitions of a configuration stay in one partition. Across five folds,
each of the 30 configurations appears in the test partition exactly once.
The following F1 and AP values are **unweighted means across folds**; onset
counts pool the mutually exclusive test events.

| Five-fold test measure | GCN+GRU | GRU | Current-minute latency rule |
| --- | ---: | ---: | ---: |
| Mean AP | 0.799 | 0.765 | 0.832 |
| Mean F1 at 0.5 | 0.678 | 0.623 | 0.892 |
| Mean F1 at validation-selected threshold | 0.608 | 0.602 | 0.892 |
| Onset recall at 0.5 | 8/55 | 6/55 | 0/55 |
| Onset recall at selected threshold | 15/55 | 14/55 | 0/55 |
| Onset root Hit@1 (regardless of alarm) | 20/55 | 21/55 | n/a |
| Joint onset alarm and root Hit@1 at 0.5 | 1/55 | 2/55 | n/a |
| Joint onset alarm and root Hit@1 at selected threshold | 2/55 | 6/55 | n/a |

The mean AP for the binary latency rule is an average of binary-score AP and
is less informative than its F1. GCN+GRU's retrospective event root Hit@1 is
19/55, while GRU's is 30/55. That retrospective metric averages root scores
over the whole observed failure event and does not require an alarm. The
joint onset measure does require an alarm and a correct top-ranked root at
the first failure minute, so it is the stricter task measure. Fold-to-fold
variation is large; for GCN+GRU, test F1 at 0.5 ranges from 0.382 to 0.852.
Across all 55 onsets, 44 input windows include the injection minute and 11
come after it; none is wholly pre-injection.

These results fix the immediate research target: improve joint onset
prediction and ranking, while retaining all-minute state prediction as the
main InstantOps-style protocol. Next add RE2-TT with its own entry telemetry
and label-threshold audit. Architecture changes should be judged against the
latency rule and no-graph GRU under the same splits and labels.

Reproduce the audit after training:

```bash
.venv/bin/python scripts/audit_re2.py \
  --data-dir data/processed/re2_ob_panel_v1 \
  --checkpoint artifacts/re2_ob_gcn_gru_v1/best_model.pt \
  --output artifacts/re2_ob_onset_audit_v1.json
```

For configuration fold `0` through `4`, add
`--split-policy configuration --split-fold N` to `scripts/run_re2.py train`,
using a separate output directory for every fold and model. Then run
`scripts/audit_re2.py` against each checkpoint; it reads the fold from the
saved run configuration.
