# RE2-TT dataset and five-to-one-minute experiment

## Source audit and label

The local `data/RCAEval/RE2-TT.zip` contains 90 numbered cases: five injected
services, six fault types (CPU, delay, disk, loss, memory, socket), three
repetitions each. Every case has injection time, metrics, logs and raw traces.
Across the archive, metrics name 69 resources (including `istio-init`), while
raw traces consistently contain 27 application services. This adapter uses
the 27 trace-observed services as graph nodes; metric-only resources are not
yet modeled. Root HTTP spans have blank `statusCode`, so they do not provide
a usable HTTP 5xx outcome label.

TT has multiple entry operations. A root HTTP span has an empty parent span
ID and an HTTP-method operation. The source audit records each operation's
completed-span p95 per minute. An operation is eligible if at least 20 spans
occur in at least five of the ten complete pre-injection baseline minutes.
The final TT failure label marks a post-injection minute when the **same**
eligible operation has at least two consecutive minutes with p95 at least
`max(2 × baseline p95, baseline p95 + 0.1 seconds)`. The injection minute is
excluded as a target. The injected service is the root label only for an
observed impact event. Isolated high post-injection minutes are excluded from
the supervised target rather than treated as definite negatives.

At a 1.5× threshold, 11 cases have sustained pre-injection high latency
(27 minutes), making that candidate too noisy. At 2× plus 0.1 seconds, the
audited pre-injection period has no sustained high-latency event and the
post-injection period yields 53/90 impacted cases, 64 events and 436 positive
minutes. There are 1,879 valid target minutes. This is an observable business
impact definition, distinct from injection-time labels and from the OB entry
endpoint definition. The archive and label schema contain the exact settings.

## Model-ready panel

The adapter builds a one-minute panel with CPU, memory, disk I/O and socket
means and observation masks, trace counts, nonzero trace status counts, mean
trace duration, log counts and error-message counts. It constructs a dynamic
service graph from parent-child spans after both spans have completed. The
panel has 2,160 minutes, 58,320 service-minute rows, 2,131 minutes with graph
edges, and 1,523 valid five-minute windows. Repetition split sizes are
485/509/529; the configuration-held-out fold-zero sizes are 898/308/317.

## Initial five-fold result

The split holds whole service/fault configurations apart. Within each fold,
18 configurations train, 6 validate and 6 test. Each of the 30 configurations
is tested once across five folds. Checkpoints use five CPU epochs, seed 42,
and the same node-head root supervision. F1/AP values are unweighted fold
means; onset counts pool disjoint test events.

| Test measure | GCN+GRU | GRU | Current-minute latency rule |
| --- | ---: | ---: | ---: |
| Mean next-minute AP | 0.579 | 0.586 | 0.773 |
| Mean next-minute F1 at 0.5 | 0.454 | 0.522 | 0.863 |
| Mean F1 at validation-selected threshold | 0.421 | 0.488 | 0.863 |
| First-failure-minute recall at 0.5 | 16/64 | 31/64 | 33/64 |
| Onset root Hit@1, regardless of alarm | 17/64 | 31/64 | n/a |
| Joint onset alarm and root Hit@1 at 0.5 | 3/64 | 17/64 | n/a |

The GCN+GRU graph implementation underperforms the no-graph GRU on this TT
setup. This is an initial five-epoch comparison, not a tuned architecture
claim. The current-minute latency rule remains much stronger for next-minute
state prediction. For 37/64 onset events, the final input minute contains
the injection; for 27/64 it comes afterward. None of these onsets measures
pre-injection forecasting. The root target is the injected service, so RCA
scores are attribution to a proxy root label. Event-level RCA metrics over
the whole incident are retrospective; the onset joint measure is stricter.

## Reproduction

```bash
.venv/bin/python scripts/audit_re2_tt_source.py \
  data/RCAEval/RE2-TT.zip artifacts/re2_tt_source_audit_v1.json
.venv/bin/python scripts/run_re2.py label-tt \
  artifacts/re2_tt_source_audit_v1.json data/processed/re2_tt_labels_v1
.venv/bin/python scripts/run_re2.py prepare-tt \
  data/RCAEval/RE2-TT.zip \
  data/processed/re2_tt_labels_v1/next_minute_labels.csv \
  data/processed/re2_tt_panel_v1
.venv/bin/python scripts/run_re2.py train \
  --data-dir data/processed/re2_tt_panel_v1 \
  --output-dir artifacts/re2_tt_gcn_gru_config_fold0_v1 \
  --model gcn-gru --split-policy configuration --split-fold 0 \
  --epochs 5 --device cpu --rca-method node-head
```

Run folds 0 through 4 and both `gcn-gru` and `gru` with separate output
directories. Audit each checkpoint with `scripts/audit_re2.py`; its output
includes per-event onset timing, model score, and root rank.
