#!/usr/bin/env python3
"""Gate 0 for multi-horizon OCSID: is the root already anomalous at decision time?

For each task (``state`` / ``first-visible``) and horizon h in {1,3,5}, count
positive windows/events and the fraction where the root service shows a signal
at the decision minute t, using the panel features and a per-case first-10-minute
baseline.  A low fraction means CPCA at that horizon has little to attribute.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2FirstVisibleWindowDataset, Re2WindowDataset  # noqa: E402

PANELS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}
SIGNAL_FEATURES = (
    "metric_cpu", "metric_mem", "metric_diskio", "metric_socket",
    "trace_duration_mean_ms", "entry_p95_ms", "log_error_count",
    "entry_degradation_score_rolling",
)


def case_baseline(dataset, case_id: str):
    indices = [
        i for i in range(len(dataset.timestamps))
        if dataset.case_ids[i] == case_id
    ]
    start = min(dataset.timestamp_epochs[i] for i in indices)
    cutoff = start + 600
    normal = [i for i in indices if dataset.timestamp_epochs[i] < cutoff]
    if len(normal) < 2:
        normal = indices
    values = dataset.features[normal]
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return mean, np.maximum(std, 1e-6)


def root_signal(dataset, end: int, root_index: int, mean, std) -> bool:
    x = dataset.features[end, root_index]
    z = np.abs((x - mean[root_index]) / std[root_index])
    signal = bool(np.nanmax(z) > 3.0)
    if "entry_p95_ms" in dataset.feature_columns:
        column = dataset.feature_columns.index("entry_p95_ms")
        ratio = x[column] / max(float(mean[root_index, column]), 1e-6)
        signal = signal or ratio > 1.5
    return signal


def evaluate(dataset, task: str) -> dict:
    positives = 0
    windows_with_signal = 0
    events: dict[str, bool] = {}
    baselines: dict[str, tuple] = {}
    for end in dataset.sample_indices:
        if dataset.failure_targets[end] != 1:
            continue
        root = int(np.asarray(dataset.node_targets[end]).argmax())
        if dataset.node_targets[end].sum() <= 0:
            continue
        case_id = dataset.case_ids[end]
        if case_id not in baselines:
            baselines[case_id] = case_baseline(dataset, case_id)
        mean, std = baselines[case_id]
        signal = root_signal(dataset, end, root, mean, std)
        positives += 1
        windows_with_signal += int(signal)
        event = dataset.event_ids[end] or f"{case_id}:{end}"
        events[event] = events.get(event, False) or signal
    return {
        "task": task,
        "positive_windows": positives,
        "events": len(events),
        "windows_with_root_signal": windows_with_signal,
        "window_signal_fraction": windows_with_signal / positives if positives else None,
        "events_with_root_signal": sum(events.values()),
        "event_signal_fraction": sum(events.values()) / len(events) if events else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ob", "tt"), required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for horizon in args.horizons:
        state = Re2WindowDataset(
            PANELS[args.dataset], history=5, split_policy="deployment",
            split_fold=0, horizon_minutes=horizon,
        )
        rows.append({"horizon": horizon, **evaluate(state, "state")})
        first = Re2FirstVisibleWindowDataset(
            PANELS[args.dataset], history=5, split_policy="deployment",
            split_fold=0, horizon_minutes=horizon,
        )
        rows.append({"horizon": horizon, **evaluate(first, "first-visible")})

    report = {"dataset": args.dataset, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda value: value.item()))
    print(f"{'h':>2} {'task':14s} {'windows':>8} {'events':>7} "
          f"{'win_sig':>8} {'evt_sig':>8}")
    for row in rows:
        print(f"{row['horizon']:>2} {row['task']:14s} {row['positive_windows']:>8} "
              f"{row['events']:>7} "
              f"{(row['window_signal_fraction'] or 0):>8.3f} "
              f"{(row['event_signal_fraction'] or 0):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
