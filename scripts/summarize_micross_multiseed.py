#!/usr/bin/env python3
"""Summarize full-July MicroSS paired runs and first-minute event outcomes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/micross_july_30ep_multiseed"
PANEL = ROOT / "data/processed/micross_mps_smoke_01"


def main() -> None:
    rows = json.loads((RUNS / "summary.json").read_text())
    schema = json.loads((PANEL / "entry_failure_v2.schema.json").read_text())
    starts = {event["event_id"]: datetime.fromisoformat(event["start_utc"])
              for event in schema["events"]}
    all_predictions = {}
    input_times = set()
    for result in rows:
        key = f"{result['model']}_seed{result['seed']}"
        predictions = [json.loads(line) for line in (RUNS / key / "predictions.jsonl").read_text().splitlines()]
        all_predictions[key] = predictions
        input_times.update(row["input_minute_utc"] for row in predictions if row["event_id"])

    entry = defaultdict(lambda: [0, 0])
    with (PANEL / "service_features_1m.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["timestamp_utc"] in input_times and row["service"] in schema["entry_services"]:
                value = entry[row["timestamp_utc"]]
                value[0] += int(float(row["trace_span_count"]))
                value[1] += int(float(row["trace_5xx_count"]))

    onset_rows = []
    for result in rows:
        key = f"{result['model']}_seed{result['seed']}"
        predictions = all_predictions[key]
        onset = [row for row in predictions
                 if row["event_id"] in starts
                 and row["failure_target"] == 1
                 and datetime.fromisoformat(row["input_minute_utc"]) + timedelta(minutes=1)
                     == starts[row["event_id"]]]
        distinct = {row["event_id"] for row in onset}
        if len(distinct) != len(onset):
            raise ValueError(f"duplicate event onset in {key}")
        attributed = [row for row in onset if row["true_services"]]
        alarm = [row for row in onset if row["failure_probability"] >= 0.5]
        baseline_alarm = [row for row in onset if
                          entry[row["input_minute_utc"]][0] >= schema["min_entry_spans"]
                          and entry[row["input_minute_utc"]][1]
                          / entry[row["input_minute_utc"]][0] >= schema["error_rate_threshold"]]
        root_hit = [row for row in attributed
                    if row["ranking"][0]["service"] in row["true_services"]]
        joint = [row for row in root_hit if row["failure_probability"] >= 0.5]
        onset_rows.append({
            "model": result["model"], "seed": result["seed"],
            "test_onset_events": len(onset), "attributed_onset_events": len(attributed),
            "alarm_at_0_5": len(alarm), "baseline_alarm": len(baseline_alarm),
            "root_top1_on_attributed": len(root_hit), "joint_alarm_root_top1": len(joint),
            "attributed_event_details": [{
                "event_id": row["event_id"], "root": row["true_services"],
                "top1": row["ranking"][0]["service"],
                "alarm": row["failure_probability"] >= 0.5,
            } for row in attributed],
        })

    output = {"dataset": "MicroSS July", "epochs": 30, "seeds": [42, 43, 44],
              "label_schema": "entry_failure_v2", "runs": rows, "onset": onset_rows}
    (RUNS / "analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for model in ("gcn-gru", "gru"):
        group = [row for row in rows if row["model"] == model]
        onsets = [row for row in onset_rows if row["model"] == model]
        def fmt(section: str, metric: str) -> str:
            values = [row[section][metric] for row in group]
            return f"{mean(values):.3f} ± {pstdev(values):.3f}"
        print(model, "F1", fmt("failure_prediction", "f1_at_0_5"),
              "AP", fmt("failure_prediction", "average_precision"),
              "RCA event Hit@1", fmt("root_cause_event_proxy", "hit_at_1"))
        print("onset", [{key: row[key] for key in (
            "seed", "test_onset_events", "attributed_onset_events", "alarm_at_0_5",
            "baseline_alarm", "root_top1_on_attributed", "joint_alarm_root_top1")}
            for row in onsets])


if __name__ == "__main__":
    main()
