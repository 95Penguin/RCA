"""Feature schema shared by OCSID training, prediction and audits.

Two feature spaces are used:

* ``PANEL_FEATURES`` (19) : per-service per-minute panel features.
* ``OP_FEATURE_NAMES`` (17) : per-entry-operation per-minute trace features.

The operation table is produced by ``scripts/build_ocsid_operation_features.py``
and consumed by ``scripts/train_ocsid_joint.py``.  Keeping the vector definition
here avoids the previous duplication between the feature builder and the trainer.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

PANEL_FEATURES = (
    "metric_cpu", "metric_mem", "metric_diskio", "metric_socket",
    "observed_cpu", "observed_mem", "observed_diskio", "observed_socket",
    "trace_span_count", "trace_error_count", "trace_duration_mean_ms",
    "log_count", "log_error_count", "log_observed",
    "entry_span_count", "entry_p95_ms", "observed_entry_p95",
    "observed_entry_baseline", "entry_degradation_score_rolling",
)

OP_FEATURE_NAMES = (
    "log_p50", "log_p95", "log_p99", "log_mean", "log_count", "log_spans",
    "n_services", "error_rate", "log_ratio_p95", "log_ratio_count", "avg_spans",
    "deg_score", "deg_binary", "log_roll_ratio", "roll_deg", "roll_deg_binary",
    "log_roll_count",
)

OP_FEATURE_DIM = len(OP_FEATURE_NAMES)


def _ratio(value: float, baseline: float) -> float:
    return math.log(max(float(value), 1e-6) / max(float(baseline), 1e-6))


def _deg_score(row: dict) -> float:
    p95 = float(row["p95_ms"])
    base = float(row["baseline_p95_ms"])
    return max(p95 / max(1.5 * base, 1e-6), (p95 - base) / 100.0)


def operation_feature_vector(row: dict) -> list[float]:
    """Build the 17-dimensional feature vector for one operation-minute row."""
    roll_p95 = float(row.get("roll_p95_ms") or row["baseline_p95_ms"])
    roll_count = float(row.get("roll_count") or row["baseline_count"])
    p95 = float(row["p95_ms"])
    deg = _deg_score(row)
    roll_deg = max(p95 / max(1.5 * roll_p95, 1e-6), (p95 - roll_p95) / 100.0)
    return [
        math.log1p(float(row["p50_ms"])),
        math.log1p(p95),
        math.log1p(float(row["p99_ms"])),
        math.log1p(float(row["mean_ms"])),
        math.log1p(float(row["count"])),
        math.log1p(float(row["span_count"])),
        float(row["n_services"]) / 10.0,
        float(row["error_count"]) / max(float(row["count"]), 1.0),
        _ratio(p95, row["baseline_p95_ms"]),
        _ratio(row["count"], row["baseline_count"]),
        float(row["span_count"]) / max(float(row["count"]), 1.0),
        deg,
        1.0 if deg >= 1.0 else 0.0,
        _ratio(p95, roll_p95),
        roll_deg,
        1.0 if roll_deg >= 1.0 else 0.0,
        _ratio(row["count"], roll_count),
    ]


def load_operation_table(
    path: str | Path,
) -> tuple[dict[tuple[str, int], dict[tuple[str, str], list[float]]], list[tuple[str, str]]]:
    """Load an operation feature CSV into ``{(case, minute): {operation: vector}}``.

    Returns the table and the sorted operation vocabulary.  The vocabulary is
    global to the dataset so every window shares the same operation axis.
    """
    table: dict[tuple[str, int], dict[tuple[str, str], list[float]]] = defaultdict(dict)
    operations: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["entry_service"], row["entry_operation"])
            operations.add(key)
            table[row["case_id"], int(row["minute_epoch"])][key] = operation_feature_vector(row)
    return table, sorted(operations)
