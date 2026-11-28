"""Gate 2 prototype: operation-conditioned critical-path counterfactual RCA.

For each positive next-minute window this audit:

1. picks the entry operation whose current p95 is most abnormal relative to its
   pre-injection normal p95 (the same "riskiest operation" policy as Gate A);
2. keeps only completed requests of that entry operation;
3. decomposes every request's span tree into local / pre-gap / post-gap;
4. replaces service ``i``'s local and its incoming gaps with the case normal
   medians, replays the serial tree, and scores
   ``r_i = sum(delta_i) / sum(D)``;
5. compares the root-service rank against global mean duration, inclusive p95
   and exclusive (local) mean duration.

Corrected-v2 assigns each inter-child gap once and rejects unsupported trees.
It exports both hard and soft scores with explicit configuration.
Use --normal-mode first10 for the fixed warm-up experiment; pre uses injection
time and is an oracle-style reference.

Example
-------
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases ts-train-service_delay/1 --max-requests 150 --debug
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


from .paths import ROOT
from .aggregation import aggregate_operations, operation_presence, MODES
from .config import ATTRIBUTION_DEFAULTS as DEFAULTS
ARCHIVES = {
    "ob": ROOT / "data/RCAEval/RE2-OB.zip",
    "tt": ROOT / "data/RCAEval/RE2-TT.zip",
}
PREFIX = {"ob": "RE2-OB", "tt": "RE2-TT"}
PANELS = {name: ROOT / f"artifacts/re2_{name}_panel_fair_v1" for name in ARCHIVES}

from ocsid import cpca  # noqa: E402

# Thin aliases keep the audit readable while the implementation lives in
# ``src/ocsid/cpca.py``.
Span = cpca.Span
_service = cpca.normalize_service
_union_length = cpca.union_length
_is_entry = cpca.is_entry
_tree = cpca.decompose_tree
_replay = cpca.replay
_score_trace = cpca.score_request
_rank = cpca.rank
_metrics = cpca.hit_metrics


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def _median(values: list[float], fallback: float = 0.0) -> float:
    return float(statistics.median(values)) if values else fallback


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)])


def _load_case_labels(dataset: str, horizon_minutes: int = 1) -> dict[str, dict[int, dict]]:
    panel = PANELS[dataset]
    minute_to_case = {
        row["timestamp_utc"]: row["case_id"]
        for row in csv.DictReader((panel / "case_minutes.csv").open(encoding="utf-8-sig"))
    }
    labels: dict[str, dict[int, dict]] = defaultdict(dict)
    with (panel / "next_minute_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = minute_to_case.get(row["input_minute_utc"])
            if case is None:
                continue
            key = _epoch(row["input_minute_utc"]) - (horizon_minutes - 1) * 60
            entry = labels[case].setdefault(
                key,
                {"failure": 0, "root": None, "event": row.get("event_id") or ""},
            )
            if row["failure_next_minute"] == "1":
                entry["failure"] = 1
            if row["root_cause_next_minute"] == "1" and row["root_label_valid"] == "1":
                entry["root"] = row["service"]
    return labels


def _collect_normal(trace_spans: list[Span], buckets, dataset: str,
                    minute: int | None = None, norm_acc=None, gap_policy="post", clock_tolerance_us=0., tree_policy="strict") -> None:
    parsed = _tree(trace_spans, dataset, gap_policy, clock_tolerance_us, tree_policy)
    if parsed is None:
        return
    by_id, _, root, local, pre, post = parsed
    if not _is_entry(root, dataset):
        return
    if _is_entry(root, dataset):
        buckets["entry"][(root.service, root.operation)].append(root.duration)
    if norm_acc is not None and minute is not None:
        agg = norm_acc.setdefault(minute, {})
        for span in trace_spans:
            if span.sid not in by_id:
                continue
            item = agg.setdefault(span.service, {"dur": [], "self": []})
            item["dur"].append(span.duration)
            item["self"].append(local[span.sid])
    for span in trace_spans:
        if span.sid not in by_id:
            continue
        buckets["local"][(span.service, span.operation)].append(local[span.sid])
        parent = by_id.get(span.parent)
        if parent is not None:
            edge = (parent.service, parent.operation, span.service, span.operation)
            buckets["pre"][edge].append(pre[span.sid])
            buckets["post"][edge].append(post[span.sid])


def _robust_stats(series: list[float]) -> tuple[float, float]:
    """Return (median, MAD-based scale) for a per-minute score series."""
    if not series:
        return 0.0, 1.0
    center = statistics.median(series)
    mad = statistics.median(abs(value - center) for value in series) * 1.4826
    scale = mad if mad > 1e-9 else max(abs(center) * 0.1, 1e-6)
    return float(center), float(scale)


def _new_state() -> dict:
    return {
        "D": 0.0,
        "requests": 0,
        "entry_durations": [],
        "cpc_delta": defaultdict(float),
        "cpc_delta_replace": defaultdict(float),
        "component_deltas": defaultdict(lambda: defaultdict(float)),
        "cpc_D": defaultdict(float),
        "cpc_cov": defaultdict(int),
        "dur_sum": defaultdict(float),
        "dur_n": defaultdict(int),
        "dur_vals": defaultdict(list),
        "self_sum": defaultdict(float),
        "self_n": defaultdict(int),
        "self_vals": defaultdict(list),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ARCHIVES), required=True)
    parser.add_argument("--cases", nargs="+", default=["all"])
    parser.add_argument("--max-requests", type=int, default=DEFAULTS["max_requests"])
    parser.add_argument("--min-op-requests", type=int, default=DEFAULTS["min_op_requests"])
    parser.add_argument("--horizon-minutes", type=int, default=1)
    parser.add_argument("--aggregation", choices=MODES, default=DEFAULTS["aggregation"])
    parser.add_argument("--soft-tau", type=float, default=DEFAULTS["temperature"],
                        help="temperature for softmax operation weighting")
    parser.add_argument("--qo-table", type=Path, default=None,
                        help="OCSID JSON with per-window operation_scores (q_o)")
    parser.add_argument("--reference", choices=("fixed", "rolling"), default="fixed",
                        help="normal reference: fixed first10/pre, or causal rolling")
    parser.add_argument("--reference-window", type=int, default=10,
                        help="rolling reference window in minutes")
    parser.add_argument("--shrink-k", type=float, default=5.0,
                        help="shrinkage strength towards the fixed prior")
    parser.add_argument("--reference-guard", type=int, default=0,
                        help="minutes immediately before t excluded from rolling")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--normal-mode", choices=("pre", "first10"), default="first10",
                        help="pre = all minutes before injection (oracle-ish); "
                             "first10 = fixed causal warm-up from the case start")
    parser.add_argument("--normal-minutes", type=int, default=10)
    parser.add_argument("--tree-policy", choices=("strict", "entry-component"), default=DEFAULTS["tree_policy"])
    parser.add_argument("--clock-tolerance-us", type=float, default=DEFAULTS["clock_tolerance_us"],
                        help="bounded clock reconciliation; 0=strict, 1000=1ms sensitivity")
    parser.add_argument("--restoration", choices=("replace", "excess"), default=DEFAULTS["restoration"])
    parser.add_argument("--gap-policy", choices=("pre", "post"), default="post")
    parser.add_argument("--mechanism-ablations", action="store_true",
                        help="same-request local/edge/ownership/operation controls")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new versioned path")
    if args.cases == ["all"]:
        prefix = PREFIX[args.dataset] + "/"
        with zipfile.ZipFile(ARCHIVES[args.dataset]) as archive:
            args.cases = sorted(name[len(prefix):-len("/traces.csv")]
                                for name in archive.namelist()
                                if name.startswith(prefix) and name.endswith("/traces.csv"))

    qo_table: dict[tuple[str, int], dict[tuple[str, str], float]] = {}
    if args.qo_table is not None:
        qo_data = json.loads(args.qo_table.read_text())
        for fold in qo_data.get("folds", []):
            operations = [tuple(op) for op in fold.get("operations", [])]
            for prediction in fold.get("test_predictions", []):
                qo_table[(prediction["case"], int(prediction["minute"]))] = dict(
                    zip(operations, prediction.get("operation_scores", []))
                )

    labels = _load_case_labels(args.dataset, args.horizon_minutes)
    report = {"dataset": args.dataset,
              "protocol": f"operation-conditioned; {args.reference}/{args.normal_mode} reference",
              "implementation_version": "corrected-v4-aggregation",
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
              "cases": {}}

    for case in args.cases:
        member = f"{PREFIX[args.dataset]}/{case.strip('/')}"
        suffix = case.strip("/").replace("/", "_")
        matches = [c for c in labels if c.endswith(suffix)]
        if not matches:
            print(f"[skip] no labels for {case}")
            continue
        case_id = matches[0]
        positive = {m: r for m, r in labels[case_id].items() if r["failure"] == 1}

        with zipfile.ZipFile(ARCHIVES[args.dataset]) as zf:
            inject = int(zf.read(f"{member}/inject_time.txt").decode().strip())
            buckets = {"local": defaultdict(list), "pre": defaultdict(list),
                       "post": defaultdict(list), "entry": defaultdict(list)}
            normal_local: dict[tuple[str, str], float] = {}
            normal_edge: dict[tuple[str, str, str, str], tuple[float, float]] = {}
            # minute -> (service, operation) -> state
            acc: dict[int, dict[tuple[str, str], dict]] = {
                minute: {} for minute in positive
            }
            rolling_cache: dict[int, tuple[dict, dict]] = {}
            tree_diagnostics = {"positive_requests_examined": 0, "invalid_positive_trees": 0, "excluded_non_entry_requests": 0, "component_omitted_spans": 0}

            # A traceID's spans are interleaved with other traces in traces.csv
            # (5,765 traceIDs form 267,041 contiguous blocks), so the whole case
            # must be grouped by traceID before any tree can be reconstructed.
            traces: dict[str, list[tuple]] = defaultdict(list)
            with zf.open(f"{member}/traces.csv") as raw:
                reader = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                )
                for row in reader:
                    try:
                        start = int(row["startTime"])
                        duration = float(row["duration"])
                    except (TypeError, ValueError):
                        continue
                    if duration < 0 or not math.isfinite(duration):
                        continue
                    traces[row["traceID"]].append((
                        row["spanID"], row["parentSpanID"],
                        _service(row["serviceName"]), row["operationName"],
                        float(start), duration,
                    ))

            def as_spans(rows: list[tuple]) -> list[Span]:
                return [Span(*row) for row in rows]

            # Pass 1: normal statistics.  ``pre`` uses everything before the
            # injection (upper bound); ``first10`` uses a fixed causal warm-up
            # from the case start, which is what an online model can actually do.
            completions = [
                int(max(span.end for span in as_spans(rows)) // 60_000_000) * 60
                for rows in traces.values() if rows
            ]
            case_start = min(completions) if completions else inject
            cutoff = (
                inject if args.normal_mode == "pre"
                else case_start + args.normal_minutes * 60
            )
            norm_acc: dict[int, dict[str, dict]] = {}
            for rows in traces.values():
                spans = as_spans(rows)
                if not spans:
                    continue
                entry = next((span for span in spans if _is_entry(span, args.dataset)), None)
                completed = int((entry.end if entry else max(span.end for span in spans)) // 60_000_000) * 60
                if completed >= cutoff:
                    continue
                _collect_normal(spans, buckets, args.dataset, completed, norm_acc, args.gap_policy, args.clock_tolerance_us, args.tree_policy)
            normal_local.update({
                key: _median(values) for key, values in buckets["local"].items()
            })
            normal_edge.update({
                key: (_median(buckets["pre"][key]), _median(buckets["post"][key]))
                for key in buckets["pre"]
            })

            # Per-minute healthy buckets for a causal rolling reference.
            minute_buckets = defaultdict(
                lambda: {"local": defaultdict(list),
                         "pre": defaultdict(list),
                         "post": defaultdict(list)}
            )
            if args.reference == "rolling":
                for rows in traces.values():
                    spans = as_spans(rows)
                    if not spans:
                        continue
                    minute = int(max(span.end for span in spans) // 60_000_000) * 60
                    parsed = _tree(spans, args.dataset, args.gap_policy, args.clock_tolerance_us, args.tree_policy)
                    if parsed is None:
                        continue
                    _, children, root, local, pre, post = parsed
                    bucket = minute_buckets[minute]
                    for span in spans:
                        if span.sid not in local:
                            continue
                        bucket["local"][(span.service, span.operation)].append(local[span.sid])
                    for parent in spans:
                        for child in children.get(parent.sid, []):
                            key = (parent.service, parent.operation,
                                   child.service, child.operation)
                            bucket["pre"][key].append(pre[child.sid])
                            bucket["post"][key].append(post[child.sid])

            # Per-service per-minute normal series -> robust z centre/scale.
            per_service_series: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: {"global": [], "inclusive": [], "exclusive": [],
                         "exclusive_p95": []}
            )
            for minute, services in norm_acc.items():
                for service, values in services.items():
                    per_service_series[service]["global"].append(
                        sum(values["dur"]) / len(values["dur"])
                    )
                    per_service_series[service]["inclusive"].append(_p95(values["dur"]))
                    per_service_series[service]["exclusive"].append(
                        sum(values["self"]) / len(values["self"])
                    )
                    per_service_series[service]["exclusive_p95"].append(
                        _p95(values["self"])
                    )
            normal_z = {
                service: {
                    name: _robust_stats(series)
                    for name, series in parts.items() if series
                }
                for service, parts in per_service_series.items()
            }

            def zscore(service: str, name: str, value: float) -> float:
                stats = normal_z.get(service, {}).get(name)
                if stats is None:
                    return 0.0
                center, scale = stats
                return float(max(min((value - center) / scale, 10.0), -10.0))

            def state_scores(state):
                """Per-service scores for one operation state."""
                total_d = state["D"] or 1.0
                cpca = {s: state["cpc_delta"][s] / total_d for s in state["cpc_delta"]}
                cpca_cov = {
                    s: state["cpc_delta"][s] / max(state["cpc_D"][s], 1e-9)
                    for s in state["cpc_delta"] if state["cpc_cov"][s]
                }
                global_score = {
                    s: state["dur_sum"][s] / state["dur_n"][s]
                    for s in state["dur_sum"] if state["dur_n"][s]
                }
                inclusive = {
                    s: _p95(state["dur_vals"][s])
                    for s in state["dur_vals"] if state["dur_vals"][s]
                }
                exclusive = {
                    s: state["self_sum"][s] / state["self_n"][s]
                    for s in state["self_sum"] if state["self_n"][s]
                }
                exclusive_p95 = {
                    s: _p95(state["self_vals"][s])
                    for s in state["self_vals"] if state["self_vals"][s]
                }
                return {
                    **{name: {service: delta / total_d for service, delta in values.items()}
                       for name, values in state["component_deltas"].items()},
                    "cpca": cpca,
                    "cpca_replace": {s: value / total_d for s, value in state["cpc_delta_replace"].items()},
                    "cpca_cov": cpca_cov,
                    "global_z": {s: zscore(s, "global", v) for s, v in global_score.items()},
                    "inclusive_z": {s: zscore(s, "inclusive", v) for s, v in inclusive.items()},
                    "exclusive_z": {s: zscore(s, "exclusive", v) for s, v in exclusive.items()},
                    "exclusive_p95_z": {
                        s: zscore(s, "exclusive_p95", v) for s, v in exclusive_p95.items()
                    },
                    "global": global_score,
                    "inclusive": inclusive,
                    "exclusive": exclusive,
                }

            def rolling_reference(minute: int, case_labels: dict):
                """Median over the last W healthy minutes, shrunk to the fixed prior."""
                healthy = []
                for offset in range(1 + args.reference_guard,
                                    args.reference_window + 1 + args.reference_guard):
                    candidate_minute = minute - offset * 60
                    row = case_labels.get(candidate_minute)
                    if row is not None and row.get("failure") == 0:
                        healthy.append(candidate_minute)

                ref_local = dict(normal_local)
                local_keys = set(normal_local)
                for healthy_minute in healthy:
                    local_keys.update(minute_buckets[healthy_minute]["local"])
                for key in local_keys:
                    values = []
                    for healthy_minute in healthy:
                        values.extend(minute_buckets[healthy_minute]["local"].get(key, []))
                    if not values:
                        continue
                    center = _median(values)
                    prior = normal_local.get(key)
                    if prior is None:
                        ref_local[key] = center
                    else:
                        weight = len(values) / (len(values) + max(args.shrink_k, 1e-6))
                        ref_local[key] = weight * center + (1.0 - weight) * prior

                ref_edge = dict(normal_edge)
                edge_keys = set(normal_edge)
                for healthy_minute in healthy:
                    edge_keys.update(minute_buckets[healthy_minute]["pre"])
                for key in edge_keys:
                    pre_values, post_values = [], []
                    for healthy_minute in healthy:
                        pre_values.extend(minute_buckets[healthy_minute]["pre"].get(key, []))
                        post_values.extend(minute_buckets[healthy_minute]["post"].get(key, []))
                    if not pre_values and not post_values:
                        continue
                    prior = normal_edge.get(key)
                    pre_center = _median(pre_values)
                    post_center = _median(post_values)
                    if prior is None:
                        ref_edge[key] = (pre_center, post_center)
                    else:
                        count = min(len(pre_values), len(post_values))
                        weight = count / (count + max(args.shrink_k, 1e-6))
                        ref_edge[key] = (
                            weight * pre_center + (1.0 - weight) * prior[0],
                            weight * post_center + (1.0 - weight) * prior[1],
                        )
                return ref_local, ref_edge

            # Pass 2: score completed requests on positive minutes.
            for rows in traces.values():
                spans = as_spans(rows)
                if not spans:
                    continue
                business_entry = next((span for span in spans if _is_entry(span, args.dataset)), None)
                completed = business_entry.end if business_entry else max(span.end for span in spans)
                minute = int(completed // 60_000_000) * 60
                if minute < cutoff or minute not in acc:
                    continue
                entry = next((span for span in spans if _is_entry(span, args.dataset)), None)
                if entry is None:
                    tree_diagnostics["excluded_non_entry_requests"] += 1
                    continue
                existing = acc[minute].get((entry.service, entry.operation))
                if existing is not None and existing["requests"] >= args.max_requests:
                    continue
                if args.reference == "rolling":
                    if minute not in rolling_cache:
                        rolling_cache[minute] = rolling_reference(minute, labels[case_id])
                    ref_local, ref_edge = rolling_cache[minute]
                else:
                    ref_local, ref_edge = normal_local, normal_edge
                tree_diagnostics["positive_requests_examined"] += 1
                parsed = _tree(spans, args.dataset, args.gap_policy,
                               args.clock_tolerance_us, args.tree_policy)
                scored = cpca.score_decomposition(parsed, ref_local, ref_edge,
                                                  args.dataset, args.restoration)
                if scored is None:
                    tree_diagnostics["invalid_positive_trees"] += 1
                    continue
                root, deltas, local = scored
                tree_diagnostics["component_omitted_spans"] += len(spans) - len(local)
                # Same sampled requests and normal references for restoration ablation.
                _, replace_deltas, _ = cpca.score_decomposition(
                    parsed, ref_local, ref_edge, args.dataset, "replace")
                component_scores = {}
                if args.mechanism_ablations:
                    controls = {"cpca_local": ("local", "callee", args.restoration),
                                "cpca_edge": ("edge", "callee", args.restoration),
                                "cpca_caller": ("all", "caller", args.restoration),
                                "cpca_excess": ("all", "callee", "excess")}
                    for name, (components, owner, mode) in controls.items():
                        _, component_scores[name], _ = cpca.score_decomposition(
                            parsed, ref_local, ref_edge, args.dataset, mode,
                            components=components, edge_owner=owner)
                op_key = (root.service, root.operation)
                state = acc[minute].setdefault(op_key, _new_state())
                if state["requests"] >= args.max_requests:
                    continue
                state["D"] += root.duration
                state["requests"] += 1
                state["entry_durations"].append(root.duration)
                for span in spans:
                    if span.sid not in local:
                        continue
                    state["dur_sum"][span.service] += span.duration
                    state["dur_n"][span.service] += 1
                    state["dur_vals"][span.service].append(span.duration)
                    state["self_sum"][span.service] += local[span.sid]
                    state["self_n"][span.service] += 1
                    state["self_vals"][span.service].append(local[span.sid])
                for name, values in component_scores.items():
                    for service, delta in values.items():
                        state["component_deltas"][name][service] += delta
                for service, delta in replace_deltas.items():
                    state["cpc_delta_replace"][service] += delta
                for service, delta in deltas.items():
                    state["cpc_delta"][service] += delta
                    state["cpc_D"][service] += root.duration
                    state["cpc_cov"][service] += 1

        base_names = ("cpca", "cpca_replace", "cpca_cov", "global_z", "inclusive_z",
                      "exclusive_z", "exclusive_p95_z")
        if args.mechanism_ablations:
            base_names += ("cpca_local", "cpca_edge", "cpca_caller", "cpca_excess")
        q_names = ("cpca_q", "cpca_cov_q") if qo_table else ()
        method_names = (tuple(base_names) + tuple(name + "_soft" for name in base_names)
                        + q_names + ("cpca_hard", "cpca_uniform", "cpca_request_weighted", "operation_presence"))
        method_names += tuple(name + "_hard" for name in base_names if name != "cpca")
        windows = []
        for minute in sorted(acc):
            states = acc[minute]
            if not states:
                continue
            candidates = []
            for op_key, state in states.items():
                if state["requests"] < args.min_op_requests:
                    continue
                normal_p95 = _p95(buckets["entry"].get(op_key, []))
                current_p95 = _p95(state["entry_durations"])
                ratio = current_p95 / max(normal_p95, 1e-6) if normal_p95 else 0.0
                candidates.append((ratio, state["requests"], op_key, state))
            if not candidates:
                op_key, state = max(states.items(), key=lambda kv: kv[1]["requests"])
                ratio = 0.0
                candidates = [(0.0, state["requests"], op_key, state)]
            else:
                ratio, _, op_key, state = max(candidates, key=lambda x: (x[0], x[1]))
            scores = state_scores(state)
            all_operation_scores = {key: state_scores(value)["cpca"] for key, value in states.items()}
            all_operation_counts = {key: value["requests"] for key, value in states.items()}
            scores["cpca_uniform"], _ = aggregate_operations(
                all_operation_scores, all_operation_counts, mode="uniform")
            scores["cpca_request_weighted"], _ = aggregate_operations(
                all_operation_scores, all_operation_counts, mode="request-weighted")
            scores["operation_presence"] = operation_presence(all_operation_scores)
            scores["cpca_hard"] = dict(scores["cpca"])

            # Soft operation aggregation: weight each operation's score by a
            # softmax over its entry-p95 ratio, so the root is still scored
            # when it is not on the single riskiest operation.
            if len(candidates) == 1:
                weights = [1.0]
            else:
                scaled = [max(candidate[0], 0.0) / max(args.soft_tau, 1e-6)
                          for candidate in candidates]
                top = max(scaled)
                exp_values = [math.exp(value - top) for value in scaled]
                total = sum(exp_values)
                weights = [value / total for value in exp_values]
            soft = {name: defaultdict(float) for name in base_names}
            for weight, (_, _, _, candidate_state) in zip(weights, candidates):
                candidate_scores = state_scores(candidate_state)
                for name in base_names:
                    for service, value in candidate_scores[name].items():
                        soft[name][service] += weight * value
            for name in base_names:
                scores[name + "_soft"] = dict(soft[name])

            if qo_table:
                q_scores = qo_table.get((case_id, minute), {})
                q_pairs = [
                    (candidate, q_scores.get(candidate[2], 0.0))
                    for candidate in candidates
                ]
                q_pairs = [(candidate, value) for candidate, value in q_pairs if value > 0]
                if q_pairs:
                    scaled = [value / max(args.soft_tau, 1e-6) for _, value in q_pairs]
                    top = max(scaled)
                    exp_values = [math.exp(value - top) for value in scaled]
                    total = sum(exp_values)
                    q_weights = [value / total for value in exp_values]
                    q_candidates = [candidate for candidate, _ in q_pairs]
                else:
                    q_weights = weights
                    q_candidates = candidates
                for name in ("cpca", "cpca_cov"):
                    combined = defaultdict(float)
                    for weight, (_, _, _, candidate_state) in zip(q_weights, q_candidates):
                        for service, value in state_scores(candidate_state)[name].items():
                            combined[service] += weight * value
                    scores[name + "_q"] = dict(combined)

            # The main score follows the declared aggregation; named controls
            # retain their meanings across versions. Keep true emitted weights.
            main_counts = {candidate[2]: candidate[1] for candidate in candidates}
            main_ratios = {candidate[2]: candidate[0] for candidate in candidates}
            soft_weights = weights
            # Match operation aggregation across every mechanism ablation;
            # otherwise uniform-full vs hard-local would change two factors.
            for name in base_names:
                scores[name + "_hard"] = dict(scores[name])
                operation_values = {candidate[2]: state_scores(candidate[3])[name]
                                    for candidate in candidates}
                scores[name], selected_weights = aggregate_operations(
                    operation_values, main_counts, main_ratios,
                    mode=args.aggregation, temperature=args.soft_tau)
                if name == "cpca":
                    main_weights = selected_weights
            row = positive[minute]
            root = row["root"]
            entry = {
                "operation_evidence": [
                    {"operation": list(key), "requests": value["requests"],
                     "p95_ratio": (_p95(value["entry_durations"]) /
                         max(_p95(buckets["entry"].get(key, [])), 1e-6)
                         if buckets["entry"].get(key) else 0.),
                     "scores": state_scores(value)} for key, value in states.items()],
                "minute": minute,
                "event": row["event"],
                "root": root,
                "operation": list(op_key),
                "operations": [list(candidate[2]) for candidate in candidates],
                "operation_weights": [round(main_weights[candidate[2]], 4) for candidate in candidates],
                "soft_operation_weights": [round(weight, 4) for weight in soft_weights],
                "entry_p95_ratio": round(ratio, 3),
                "requests": state["requests"],
                "scores": scores,
                "ranks": {name: _rank(scores[name], root) for name in method_names},
                "coverage": {
                    s: state["cpc_cov"][s] / max(state["requests"], 1)
                    for s in state["cpc_cov"]
                },
            }
            if args.debug and len(windows) == 0:
                print(f"DEBUG {minute} op={op_key} ratio={ratio:.2f} "
                      f"req={state['requests']} root={root}")
                for name in method_names:
                    top = sorted(scores[name].items(), key=lambda kv: -kv[1])[:6]
                    print(f"  {name:10s}", [(k, round(v, 3)) for k, v in top])
            windows.append(entry)

        events: dict[str, list[dict]] = defaultdict(list)
        for window in windows:
            events[window["event"]].append(window)
        event_ranks = {name: [] for name in method_names}
        event_details = []
        for event, rows in events.items():
            if not rows or rows[0]["root"] is None:
                continue
            root = rows[0]["root"]
            detail = {"event": event, "root": root, "windows": len(rows), "ranks": {}}
            for name in method_names:
                merged: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    for service, value in row["scores"][name].items():
                        merged[service].append(value)
                mean_scores = {s: sum(v) / len(v) for s, v in merged.items()}
                rank = _rank(mean_scores, root)
                detail["ranks"][name] = rank
                if rank is not None:
                    event_ranks[name].append(rank)
            event_details.append(detail)

        report["cases"][case] = {
            "case_id": case_id,
            "tree_diagnostics": tree_diagnostics,
            "normal_cutoff_epoch": cutoff,
            "target_events": sorted({row["event"] for row in positive.values() if row["event"]}),
            "inject_epoch": inject,
            "positive_minutes": len(positive),
            "scored_windows": len(windows),
            "window_metrics": {
                name: _metrics([w["ranks"][name] for w in windows if w["ranks"][name]])
                for name in method_names
            },
            "event_metrics": {name: _metrics(ranks) for name, ranks in event_ranks.items()},
            "events": event_details,
            "windows": [
                {**{k: v for k, v in w.items() if k != "scores"},
                 "cpca_scores": w["scores"]["cpca"],
                 "scores_by_method": w["scores"]}
                for w in windows
            ],
        }
        print(f"[{case}] event "
              + " ".join(f"{name}={report['cases'][case]['event_metrics'][name]}"
                         for name in method_names))

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
