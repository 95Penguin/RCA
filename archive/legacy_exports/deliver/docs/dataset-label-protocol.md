# Dataset and next-minute label protocol (2026-11-02)

The fixed study systems are MicroSS, RCAEval RE2-OB and, after download,
RCAEval RE2-TT. `A.zip` is not a primary 5-to-1-minute dataset. Each dataset
uses the same task: five completed one-minute telemetry buckets predict the
next minute's **observable degradation state** and rank a service root when
that state has an attributable fault. An injection start is metadata, not a
positive failure label. Results and event counts are reported per dataset.

## MicroSS / GAIA July

The existing `entry_failure_v2.csv` and its schema are the v1 study labels.
The input panel has 440,430 service-minute rows over 44,043 minutes and ten
services. The available graph was built for 42,305 panel minutes. The raw
July run log yielded 667 injection records. The business outcome is based on
the completed spans of `webservice1` and `webservice2` combined:

- At least 20 entry spans in a minute; lower coverage is **unknown**.
- At least 50% of the observed entry spans have a 5xx status.
- The high-error condition lasts at least two consecutive minutes; the
  confirmed event includes its first high-error minute retrospectively.
- Five input minutes predict the following minute, including ongoing events.
- Only one reliable injected service active at event onset yields a candidate
  RCA target. Three run-log records lasting more than 3,600 seconds are
  excluded from root attribution pending verification.

This protocol gives 160 events, 1,403 positive minutes, and 15 events with a
unique candidate root. Event count sensitivity with the same request minimum
and two-minute duration: 0.25 error ratio gives 225 events; 0.50 gives 160;
0.75 gives 10; 0.999 gives 6. The 0.50 cutoff is therefore the fixed severe
degradation setting for this study, not the paper's stated 99/99.9% SLA. The
entry spans are a proxy for user requests; top-level request identity remains
to be verified. The retained failure outcome is usable for prediction; root
scores are reported only on the 15 attributed events.

Generate the same label from the cached panel with a fresh output path:

```bash
.venv/bin/python scripts/run_micross.py label \
  --data-dir data/processed/micross_mps_smoke_01 \
  --output data/processed/micross_mps_smoke_01/entry_failure_v3_study.csv \
  --error-rate-threshold 0.5 --min-entry-spans 20 \
  --consecutive-minutes 2 --max-injection-seconds 3600
```

July is the multi-modal MicroSS subset currently available: the local metric
archive contains August metric files and the business archive contains an
August file, but the local trace archives list only July trace tables.

## RCAEval RE2-OB

The original ZIP has 90 official numbered fault cases: five root services,
six fault types, three repetitions per service/type. A nested
`checkoutservice_cpu/multi-source-data` directory also has an injection-time
file; it is **not** a numbered study case. Every official case has roughly 12
minutes before and 12 minutes after fault injection.

We read raw `traces.csv` frontend spans (`serviceName=frontendservice`,
`operationName=frontend`) and aggregate by **completion** minute. The root
frontend spans have blank status codes, so a 5xx-only outcome cannot be
constructed from those records. The observable outcome is:

- At least 100 completed frontend spans in a minute; lower coverage is
  **unknown**.
- Let `b` be the median of the per-minute frontend P95 latency values in the
  ten complete pre-injection minutes. Each of those minutes must meet the
  request minimum; at least five are required.
- The latency threshold is `max(1.5 * b, b + 0.1 seconds)`.
- Two consecutive complete **post-injection** minutes above that threshold
  form an event. The minute containing injection is excluded as mixed. Event
  onset is the first high-latency minute, assigned retrospectively.
- The injected service named in the case path is a candidate root only for a
  confirmed observable event. Other injected cases remain negative or unknown
  according to their observed request coverage.

The current generator produces 51 cases with observable impact, 55 events,
455 positive minutes, and 1,944 valid target minutes across 90 cases. By fault
type, impact cases are CPU 5/15, delay 13/15, disk 5/15, loss 11/15,
memory 11/15 and socket 6/15. This is a per-case latency outcome, not a claim
that the other 39 injections had no internal effect. The baseline comes only
from that case's earlier, pre-injection telemetry. It is available before any
post-injection target minute; injection metadata is never a predictor feature.

Run on a fresh output directory:

```bash
.venv/bin/python scripts/run_re2.py \
  data/RCAEval/RE2-OB.zip data/processed/re2_ob_labels_v1
```

Outputs are `next_minute_labels.csv`, `events.csv`, `cases.csv`, and
`label_schema.json`. The label CSV is **not yet a model-ready service feature
panel**; an RE2 adapter must build per-service minute features and graphs.
All minutes from a case must remain within one split. Repetitions, acquisition
order and fault-type balance should be considered when fixing that split.

## RE2-TT extension

The official RCAEval RE2-TT benchmark has 90 cases and the same broad data
types: metrics, logs, traces, injection time and root service. Its entrance
operation, request coverage and latency distributions must be audited after
download. The same five-to-one-minute target semantics and observed-impact
principle will apply, with a dataset-specific baseline/threshold recorded
before model training. RE2-OB thresholds are **not** copied numerically to TT
without that audit.

## Evaluation boundary

This is next-minute **state prediction**, including ongoing degradation.
Prediction F1/AP and event-level RCA Hit@K/MRR use only valid targets. Report
the number of positive incidents, attributed incidents and the first-minute
or onset subset separately. A current-state persistence baseline is essential:
scheduled fault injections may have no visible precursor before their onset.
No published InstantOps number is used as a directly comparable result unless
the published and local label/split protocols match.
