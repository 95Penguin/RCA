"""Read-only coverage audit for the three InstantOps study panels.

Usage: python scripts/audit_dataset_readiness.py /path/to/InstantOps
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def summarize_panel(directory: Path, labels: Path) -> dict:
    schema = json.loads((directory / "feature_schema.json").read_text())
    columns = schema["feature_columns"]
    counts = Counter()
    values = Counter()
    row_count = 0
    with (directory / "service_features_1m.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            row_count += 1
            for column in columns:
                number = float(row[column])
                if number != 0:
                    values[column] += 1
            counts["metric_observed"] += any(
                float(row[column]) > 0 for column in columns
                if column.startswith("observed_") and column != "log_observed"
            )
            counts["trace_observed"] += float(row.get("trace_span_count", 0)) > 0
            counts["log_observed"] += float(row.get("log_observed", 0)) > 0
            counts["log_error_nonzero"] += float(row.get("log_error_count", 0)) > 0
            counts["trace_error_nonzero"] += float(
                row.get("trace_error_count", row.get("trace_5xx_count", 0))
            ) > 0
            counts["three_modalities_observed"] += (
                any(float(row[column]) > 0 for column in columns
                    if column.startswith("observed_") and column != "log_observed")
                and float(row.get("trace_span_count", 0)) > 0
                and float(row.get("log_observed", 0)) > 0
            )

    unique_targets = {}
    roots = set()
    with labels.open(newline="") as source:
        for row in csv.DictReader(source):
            key = row["target_minute_utc"]
            record = unique_targets.setdefault(key, {
                "positive": 0, "observed": 0, "root_valid": 0, "event_id": ""
            })
            record["positive"] = int(row["failure_next_minute"])
            record["observed"] = int(row.get("entry_observed", 1))
            record["root_valid"] = int(row.get("root_label_valid", 0))
            record["event_id"] = row.get("event_id", "")
            if int(row["root_cause_next_minute"]):
                roots.add((key, row["service"]))
    positive_events = {
        record["event_id"] for record in unique_targets.values()
        if record["positive"] and record["event_id"]
    }
    attributed_events = {
        record["event_id"] for record in unique_targets.values()
        if record["positive"] and record["root_valid"] and record["event_id"]
    }
    return {
        "services": len(schema["services"]), "service_minute_rows": row_count,
        "feature_columns": columns, "nonzero_feature_rows": dict(values),
        "coverage_rows": dict(counts),
        "target_minutes": len(unique_targets),
        "observed_targets": sum(r["observed"] for r in unique_targets.values()),
        "positive_target_minutes": sum(r["positive"] for r in unique_targets.values()),
        "root_valid_positive_minutes": sum(
            r["positive"] and r["root_valid"] for r in unique_targets.values()
        ),
        "positive_events": len(positive_events),
        "attributed_events": len(attributed_events),
        "positive_root_service_minutes": len(roots),
        "has_entry_p95_input": any("p95" in name.lower() for name in columns),
    }


def archive_derived_series(path: Path) -> dict:
    wanted = ("logts.csv", "tracets_err.csv", "tracets_lat.csv", "cluster_info.json")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        return {name: sum(path.endswith("/" + name) for path in names)
                for name in wanted}


def adjacent_cross_split_pairs(directory: Path) -> list[int]:
    """Count near-adjacent cases assigned to different configuration folds.

    This is a timing warning, not proof that two cases shared one deployment.
    """
    from datetime import datetime

    bounds = defaultdict(lambda: [float("inf"), float("-inf")])
    with (directory / "case_minutes.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            epoch = datetime.fromisoformat(row["timestamp_utc"]).timestamp()
            pair = bounds[row["case_id"]]
            pair[0] = min(pair[0], epoch)
            pair[1] = max(pair[1], epoch)
    ordered = sorted(bounds.items(), key=lambda item: item[1][0])
    grouped = defaultdict(list)
    for case in bounds:
        config = case.rsplit("_", 1)[0]
        grouped[config.rsplit("_", 1)[-1]].append(config)
    cross = [0] * 5
    nearby = 0
    for (left, a), (right, b) in zip(ordered, ordered[1:]):
        if b[0] - a[1] > 15 * 60:
            continue
        nearby += 1
        for fold in range(5):
            def partition(case: str) -> str:
                config = case.rsplit("_", 1)[0]
                fault = config.rsplit("_", 1)[-1]
                index = sorted(grouped[fault]).index(config)
                fault_index = sorted(grouped).index(fault)
                if index == (fault_index + fold) % 5:
                    return "validation"
                if index == (fault_index + fold + 2) % 5:
                    return "test"
                return "train"
            cross[fold] += partition(left) != partition(right)
    return {"nearby_case_pairs_at_most_15_minutes": nearby,
            "cross_partition_pairs_by_fold": cross}


def main(project: Path) -> None:
    processed = project / "data/processed"
    micro = processed / "micross_mps_smoke_01"
    result = {
        "MicroSS": summarize_panel(micro, micro / "entry_failure_v2.csv"),
    }
    for short, dataset in (("OB", "RE2-OB"), ("TT", "RE2-TT")):
        directory = processed / f"re2_{short.lower()}_panel_v1"
        entry = summarize_panel(directory, directory / "next_minute_labels.csv")
        entry["archive_derived_series"] = archive_derived_series(
            project / "data/RCAEval" / f"{dataset}.zip"
        )
        entry["nearby_split_audit"] = adjacent_cross_split_pairs(directory)
        result[dataset] = entry
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
