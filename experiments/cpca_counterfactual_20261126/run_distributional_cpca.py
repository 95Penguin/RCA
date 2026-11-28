"""Matched empirical counterfactual reference for the existing CPCA audit.

This is a side-by-side experiment runner.  It leaves repository code and the
fixed median-reference audit untouched, and runs the same audit with a
per-case empirical distribution of healthy span costs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from experiment_paths import ROOT, OUT, VALIDATION
sys.path.insert(0, str(ROOT / "src"))
from ocsid import audit, cpca  # noqa: E402
from ocsid.evaluation import evaluate_attribution  # noqa: E402

ORIGINAL_COLLECT_NORMAL = audit._collect_normal
ORIGINAL_SCORE = cpca.score_decomposition
MIN_POOL = 5

# The tuple in each pool is one observed normal span's joint (local, pre, post)
# vector.  The joint key retains both entry-operation and incoming-edge context.
POOLS = {}
SAMPLE_STATS_CACHE = {}
POOL_COUNTS = defaultdict(int)
FALLBACK_COUNTS = defaultdict(int)
NO_POOL_COUNT = 0
VARIANCE_BY_OPERATION = defaultdict(lambda: defaultdict(float))
DENOMINATOR_BY_OPERATION = defaultdict(float)
LAST_PARSED = None


def reset_case_state():
    global POOLS, POOL_COUNTS, FALLBACK_COUNTS, NO_POOL_COUNT
    global SAMPLE_STATS_CACHE, VARIANCE_BY_OPERATION, DENOMINATOR_BY_OPERATION, LAST_PARSED
    POOLS = {
        "exact": defaultdict(list), "entry_operation": defaultdict(list),
        "global_edge": defaultdict(list), "global_operation": defaultdict(list),
    }
    SAMPLE_STATS_CACHE = {}
    POOL_COUNTS = defaultdict(int)
    FALLBACK_COUNTS = defaultdict(int)
    NO_POOL_COUNT = 0
    VARIANCE_BY_OPERATION = defaultdict(lambda: defaultdict(float))
    DENOMINATOR_BY_OPERATION = defaultdict(float)
    LAST_PARSED = None


def edge_context(parent):
    return None if parent is None else (parent.service, parent.operation)


def collect_normal_with_joint_pools(spans, buckets, dataset, minute=None,
                                    norm_acc=None, gap_policy="post",
                                    clock_tolerance_us=0., tree_policy="strict"):
    parsed = cpca.decompose_tree(spans, dataset, gap_policy,
                                 clock_tolerance_us, tree_policy)
    if parsed is not None:
        by_id, _, root, local, pre, post = parsed
        if cpca.is_entry(root, dataset):
            entry_key = (root.service, root.operation)
            for span in by_id.values():
                parent = by_id.get(span.parent)
                parent_key = edge_context(parent)
                has_parent = parent is not None
                vector = (float(local[span.sid]), float(pre.get(span.sid, 0.)),
                          float(post.get(span.sid, 0.)))
                POOLS["exact"][
                    (entry_key, span.service, span.operation, parent_key)
                ].append(vector)
                POOLS["entry_operation"][
                    (entry_key, span.service, span.operation, has_parent)
                ].append(vector)
                POOLS["global_edge"][
                    (span.service, span.operation, parent_key)
                ].append(vector)
                POOLS["global_operation"][
                    (span.service, span.operation, has_parent)
                ].append(vector)
    if parsed is None or not cpca.is_entry(root, dataset):
        return
    buckets["entry"][(root.service, root.operation)].append(root.duration)
    if norm_acc is not None and minute is not None:
        agg = norm_acc.setdefault(minute, {})
        for span in by_id.values():
            item = agg.setdefault(span.service, {"dur": [], "self": []})
            item["dur"].append(span.duration)
            item["self"].append(local[span.sid])
    for span in by_id.values():
        buckets["local"][(span.service, span.operation)].append(local[span.sid])
        parent = by_id.get(span.parent)
        if parent is not None:
            edge = (parent.service, parent.operation, span.service, span.operation)
            buckets["pre"][edge].append(pre[span.sid])
            buckets["post"][edge].append(post[span.sid])


def choose_pool(entry_key, span, parent):
    parent_key = edge_context(parent)
    has_parent = parent is not None
    choices = (
        ("exact", (entry_key, span.service, span.operation, parent_key)),
        ("entry_operation", (entry_key, span.service, span.operation, has_parent)),
        ("global_edge", (span.service, span.operation, parent_key)),
        ("global_operation", (span.service, span.operation, has_parent)),
    )
    for tier, key in choices:
        values = POOLS[tier].get(key, ())
        if len(values) >= MIN_POOL:
            return tier, values
    for tier, key in choices:
        values = POOLS[tier].get(key, ())
        if values:
            return tier, values
    return None, ()


def sample_stats(values, components):
    cache_key = (id(values), tuple(components))
    cached = SAMPLE_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    projected = [sum(row[j] for j in components) for row in values]
    mean = sum(projected) / len(projected)
    if len(projected) > 1:
        variance = sum((x - mean) ** 2 for x in projected) / (len(projected) - 1)
        variance /= len(projected)  # uncertainty of the empirical reference mean
    else:
        variance = 0.0
    result = (mean, variance)
    SAMPLE_STATS_CACHE[cache_key] = result
    return result


def distributional_score(parsed, normal_local, normal_edge, dataset,
                          restoration="replace", *, components="all",
                          edge_owner="callee"):
    global NO_POOL_COUNT, LAST_PARSED
    if parsed is None:
        return None
    if restoration != "replace" or components != "all" or edge_owner != "callee":
        return ORIGINAL_SCORE(parsed, normal_local, normal_edge, dataset,
                              restoration, components=components,
                              edge_owner=edge_owner)

    by_id, _, root, local, pre, post = parsed
    if root.duration <= 0 or not cpca.is_entry(root, dataset):
        return None
    entry_key = (root.service, root.operation)
    deltas = defaultdict(float)
    variance = defaultdict(float)
    used_tiers = defaultdict(int)
    missed = 0
    for span in by_id.values():
        deltas[span.service] += 0.0
        parent = by_id.get(span.parent)
        tier, values = choose_pool(entry_key, span, parent)
        if values:
            mean_cost, var_cost = sample_stats(values, (0, 1, 2))
            observed = local[span.sid] + pre.get(span.sid, 0.) + post.get(span.sid, 0.)
            deltas[span.service] += observed - mean_cost
            variance[span.service] += var_cost
            used_tiers[tier] += 1
        else:
            # Explicit last resort: keep the fixed median CPCA reference and
            # count the fallback so sparse support is visible in the report.
            missed += 1
            observed = local[span.sid]
            reference = normal_local.get((span.service, span.operation), observed)
            if parent is not None:
                edge = (parent.service, parent.operation, span.service, span.operation)
                ref_pre, ref_post = normal_edge.get(
                    edge, (pre.get(span.sid, 0.), post.get(span.sid, 0.)))
                observed += pre.get(span.sid, 0.) + post.get(span.sid, 0.)
                reference += ref_pre + ref_post
            deltas[span.service] += observed - reference
    deltas = {service: max(value, 0.0) for service, value in deltas.items()}

    # The audit calls score_decomposition twice on the same parsed request
    # (configured score and replace control). Record sampling uncertainty once.
    if parsed is not LAST_PARSED:
        LAST_PARSED = parsed
        minute = int(root.end // 60_000_000) * 60
        op_key = (root.service, root.operation)
        DENOMINATOR_BY_OPERATION[(minute, op_key)] += root.duration
        for service, value in variance.items():
            if deltas.get(service, 0.) > 0:
                VARIANCE_BY_OPERATION[(minute, op_key)][service] += value
        for tier, count in used_tiers.items():
            POOL_COUNTS[tier] += count
        NO_POOL_COUNT += missed
        if missed:
            FALLBACK_COUNTS["fixed_median"] += missed
    return root, deltas, local


def flatten_case_name(dataset, case_name):
    prefix = audit.PREFIX[dataset] + "_"
    raw = case_name[len(prefix):] if case_name.startswith(prefix) else case_name
    fault, repetition = raw.rsplit("_", 1)
    if repetition.isdigit():
        return f"{fault}/{repetition}"
    return raw


def annotate_ci(case_report):
    for window in case_report.get("windows", []):
        minute = int(window["minute"])
        op_keys = [tuple(item) for item in window.get("operations", [])]
        weights = window.get("operation_weights", [])
        combined_var = defaultdict(float)
        support_ok = True
        for op_key, weight in zip(op_keys, weights):
            denom = DENOMINATOR_BY_OPERATION.get((minute, op_key), 0.)
            if denom <= 0:
                continue
            for service, value in VARIANCE_BY_OPERATION.get((minute, op_key), {}).items():
                combined_var[service] += float(weight) ** 2 * value / (denom ** 2)
        scores = window.get("scores_by_method", {}).get("cpca_distributional", {})
        ci = {}
        for service, score in scores.items():
            se = math.sqrt(max(combined_var.get(service, 0.), 0.))
            ci[service] = [max(score - 1.96 * se, 0.), score + 1.96 * se]
        window["reference_sampling_ci95"] = ci
        window["reference_sampling_ci_note"] = (
            "Approximate reference-mean uncertainty only; assumes independent matched-span draws."
        )


def rename_cpca_fields(case_doc):
    for case in case_doc.get("cases", {}).values():
        for window in case.get("windows", []):
            if "scores_by_method" in window:
                window["scores_by_method"]["cpca_distributional"] = window["scores_by_method"].pop("cpca", {})
            window["cpca_scores"] = window.get("scores_by_method", {}).get("cpca_distributional", {})
            if "ranks" in window:
                window["ranks"]["cpca_distributional"] = window["ranks"].pop("cpca", None)
            for evidence in window.get("operation_evidence", []):
                scores = evidence.get("scores", {})
                if "cpca" in scores:
                    scores["cpca_distributional"] = scores.pop("cpca")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cohort = json.loads((VALIDATION / f"prediction_{args.dataset}_corrected_v2_seed42.json").read_text())
    positive_cases = sorted({
        row["case"] for fold in cohort["folds"] for row in fold["test_predictions"]
        if row["failure_target"] and row["root"]
    })
    if args.limit_cases:
        positive_cases = positive_cases[:args.limit_cases]
        selected = set(positive_cases)
        cohort = dict(cohort)
        cohort["folds"] = [
            {**fold, "test_predictions": [row for row in fold["test_predictions"]
                                           if row["case"] in selected]}
            for fold in cohort["folds"]
        ]
    output = args.output or OUT / "distributional_cpca" / args.dataset / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    audit._collect_normal = collect_normal_with_joint_pools
    cpca.score_decomposition = distributional_score
    report = {"dataset": args.dataset,
              "implementation_version": "matched-empirical-distributional-cpca-v1",
              "protocol": "all ground-truth positive labeled corrected-v2 forecast windows; not alert-conditioned; first10 healthy reference; joint local/pre/post samples matched by entry operation and incoming edge; mean expected latency reduction; uniform operation aggregation",
              "config": {"min_pool": MIN_POOL, "max_requests": 100,
                         "normal_minutes": 10, "aggregation": "uniform",
                         "positive_cases": len(positive_cases)},
              "cases": {}, "support_diagnostics": {}}
    per_case_dir = output.parent / output.stem / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    for index, case_name in enumerate(positive_cases, 1):
        reset_case_state()
        case_path = flatten_case_name(args.dataset, case_name)
        one_output = per_case_dir / (case_path.replace("/", "_") + ".json")
        if one_output.exists():
            raise FileExistsError(one_output)
        print(f"[{args.dataset} {index}/{len(positive_cases)}] {case_path}", flush=True)
        status = audit.main([
            "--dataset", args.dataset, "--cases", case_path,
            "--max-requests", "100", "--min-op-requests", "1",
            "--aggregation", "uniform", "--normal-mode", "first10",
            "--normal-minutes", "10", "--tree-policy", "entry-component",
            "--clock-tolerance-us", "1000", "--restoration", "replace",
            "--gap-policy", "post", "--output", str(one_output),
        ])
        if status != 0:
            raise RuntimeError(f"CPCA audit failed for {case_path}: {status}")
        case_doc = json.loads(one_output.read_text())
        rename_cpca_fields(case_doc)
        for key, value in case_doc["cases"].items():
            annotate_ci(value)
            report["cases"][key] = value
        report["support_diagnostics"][case_path] = {
            "pool_vectors_by_tier": dict(POOL_COUNTS),
            "median_fallback_span_count": NO_POOL_COUNT,
            "fallback_counts": dict(FALLBACK_COUNTS),
            "pool_sizes": {tier: len(values) for tier, values in POOLS.items()},
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    current = evaluate_attribution(
        report,
        cohort,
        "cpca_distributional",
    )
    report["metrics"] = {"distributional": current}
    if not args.limit_cases:
        median_audit = json.loads((OUT / "attribution" / args.dataset / "audit.json").read_text())
        report["metrics"]["median_cpca"] = evaluate_attribution(median_audit, cohort, "cpca_uniform")
    tier_totals = defaultdict(int)
    for case_diag in report["support_diagnostics"].values():
        for tier, count in case_diag["pool_vectors_by_tier"].items():
            tier_totals[tier] += count
    report["support_summary"] = {
        "pool_vectors_by_tier": dict(tier_totals),
        "median_fallback_span_count": sum(v["median_fallback_span_count"] for v in report["support_diagnostics"].values()),
        "case_count": len(report["support_diagnostics"]),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "distributional_event": current["event"],
        "distributional_window": current["window"],
        "support_summary": report["support_summary"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
