#!/usr/bin/env python3
"""Add caller/callee directed graph arrays to an existing audited RE2 panel."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import (  # noqa: E402
    SERVICES,
    TT_SERVICES,
    _aggregate_traces,
    _read_source_labels,
)


DATASETS = {
    "ob": {
        "name": "RE2-OB",
        "services": SERVICES,
        "archive": ROOT / "data" / "RCAEval" / "RE2-OB.zip",
        "labels": ROOT / "data" / "processed" / "re2_ob_labels_v1" / "next_minute_labels.csv",
        "panel": ROOT / "artifacts" / "re2_ob_panel_fair_v1",
    },
    "tt": {
        "name": "RE2-TT",
        "services": TT_SERVICES,
        "archive": ROOT / "data" / "RCAEval" / "RE2-TT.zip",
        "labels": ROOT / "data" / "processed" / "re2_tt_labels_v1" / "next_minute_labels.csv",
        "panel": ROOT / "artifacts" / "re2_tt_panel_fair_v1",
    },
}


def build(dataset: str, output_root: Path) -> dict[str, object]:
    config = DATASETS[dataset]
    archive_path = Path(config["archive"])
    label_path = Path(config["labels"])
    source_panel = Path(config["panel"])
    services = tuple(config["services"])
    dataset_name = str(config["name"])
    destination = output_root / f"re2_{dataset}_panel_directed_v1"
    if destination.exists():
        raise FileExistsError(f"directed panel output exists: {destination}")

    label_schema = json.loads((label_path.parent / "label_schema.json").read_text())
    min_entry_spans = int(
        label_schema.get("min_entry_spans", label_schema.get("min_operation_spans", 0))
    )
    by_case, case_by_minute = _read_source_labels(label_path)
    directed_by_minute: dict[int, np.ndarray] = {}
    resolved_total = 0

    with zipfile.ZipFile(archive_path) as archive:
        for number, (case_id, _) in enumerate(sorted(by_case.items()), 1):
            prefix, repetition = case_id.rsplit("_", 1)
            case = prefix.replace(dataset_name + "_", dataset_name + "/", 1) + "/" + repetition
            minutes = {
                minute for minute, owner in case_by_minute.items() if owner == case_id
            }
            _, edges, _, resolved, _ = _aggregate_traces(
                archive,
                case,
                minutes,
                services,
                dataset_name,
                min_entry_spans,
                float(label_schema["latency_ratio"]),
                1000 * float(label_schema["min_latency_increase_seconds"]),
            )
            directed_by_minute.update(edges)
            resolved_total += resolved
            if number % 10 == 0 or number == len(by_case):
                print(
                    f"directed graph {dataset}: {number}/{len(by_case)} cases",
                    flush=True,
                )

    with np.load(source_panel / "service_adjacency_1m.npz", allow_pickle=False) as old:
        timestamps = old["timestamps"].copy()
        stored_services = old["services"].copy()
        existing_symmetric = old["adjacency"].copy()
    if tuple(stored_services.tolist()) != services:
        raise ValueError("existing panel service order differs from directed extraction")
    if not np.array_equal(timestamps, np.asarray(sorted(case_by_minute), dtype=np.int64)):
        raise ValueError("existing panel timestamps differ from source labels")

    caller_row = np.stack([
        directed_by_minute.get(
            int(minute), np.zeros((len(services), len(services)), dtype=np.float32)
        )
        for minute in timestamps
    ]).astype(np.float32)
    callee_row = caller_row.transpose(0, 2, 1).copy()
    regenerated_symmetric = caller_row + callee_row
    if not np.array_equal(regenerated_symmetric, existing_symmetric):
        difference = np.abs(regenerated_symmetric - existing_symmetric)
        raise ValueError(
            "directed extraction does not reconstruct the existing symmetric graph: "
            f"changed={int((difference > 0).sum())} max_abs={float(difference.max())}"
        )

    destination.mkdir(parents=True)
    for source in source_panel.iterdir():
        if source.name in {"service_adjacency_1m.npz", "feature_schema.json"}:
            continue
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    np.savez_compressed(
        destination / "service_adjacency_1m.npz",
        timestamps=timestamps,
        services=stored_services,
        adjacency=existing_symmetric,
        caller_row_adjacency=caller_row,
        callee_row_adjacency=callee_row,
    )
    schema = json.loads((source_panel / "feature_schema.json").read_text())
    schema["graph_policy"] = (
        "completed parent/child calls with symmetric, caller-row and callee-row arrays"
    )
    (destination / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    report = {
        "dataset": dataset,
        "cases": len(by_case),
        "resolved_cross_service_calls": resolved_total,
        "timestamps": int(len(timestamps)),
        "services": len(services),
        "symmetric_reconstruction_exact": True,
        "caller_row_nonzero": int((caller_row > 0).sum()),
        "callee_row_nonzero": int((callee_row > 0).sum()),
        "output": str(destination),
    }
    (destination / "directed_graph_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/private/tmp/nextminuterca_directed_panels"),
    )
    args = parser.parse_args()
    reports = [build(dataset, args.output_root) for dataset in args.datasets]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
