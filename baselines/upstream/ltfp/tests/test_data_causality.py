from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import (  # noqa: E402
    TIME_COLUMN,
    aggregate_raw_logs,
    clean_metric_frame,
    load_rcaeval_cases,
    merge_causal_time,
    prefix_aux_columns,
    read_interval_aggregate_csv,
    right_closed_resample,
    parse_case_name,
)
from ltfp.features import infer_service_name  # noqa: E402


class DataCausalityTests(unittest.TestCase):
    def test_case_name_parser_handles_inline_numeric_repetition(self) -> None:
        self.assertEqual(
            parse_case_name(Path("ts-route-service_f3_1")),
            ("ts-route-service", "f3"),
        )

    def test_cleaning_sorts_before_forward_fill(self) -> None:
        raw = pd.DataFrame({TIME_COLUMN: [2.0, 1.0], "metric": [2.0, np.nan]})
        clean = clean_metric_frame(raw)
        self.assertEqual(clean[TIME_COLUMN].tolist(), [1.0, 2.0])
        self.assertEqual(clean["metric"].tolist(), [0.0, 2.0])

    def test_cleaning_preserves_latency_quantile_identity(self) -> None:
        raw = pd.DataFrame(
            {
                TIME_COLUMN: [0.0, 1.0],
                "front-end_latency": [1.0, 2.0],
                "front-end_latency-90": [3.0, 4.0],
                "front-end_latency-95": [5.0, 6.0],
            }
        )
        clean = clean_metric_frame(raw)
        self.assertEqual(list(clean.columns), list(raw.columns))

    def test_backward_alignment_never_uses_future_auxiliary_row(self) -> None:
        primary = pd.DataFrame({TIME_COLUMN: [5.0, 10.0, 15.0, 20.0], "metric": [0, 0, 0, 0]})
        auxiliary = pd.DataFrame({TIME_COLUMN: [0.0, 15.0], "log_count": [1.0, 99.0]})
        merged = merge_causal_time(primary, auxiliary)
        self.assertEqual(merged["log_count"].tolist(), [1.0, 1.0, 99.0, 99.0])

    def test_preaggregated_left_edge_is_shifted_to_availability_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logts.csv"
            pd.DataFrame({TIME_COLUMN: [0.0, 15.0], "front-end_1": [3.0, 7.0]}).to_csv(path, index=False)
            completed = read_interval_aggregate_csv(path, interval_seconds=15)
        self.assertEqual(completed[TIME_COLUMN].tolist(), [15.0, 30.0])
        primary = pd.DataFrame({TIME_COLUMN: [14.0, 15.0, 29.0, 30.0], "metric": 0.0})
        merged = merge_causal_time(primary, completed, tolerance_seconds=15)
        self.assertEqual(merged["front-end_1"].tolist(), [0.0, 3.0, 3.0, 7.0])

    def test_right_closed_resampling_assigns_boundary_to_completed_bucket(self) -> None:
        raw = pd.DataFrame({TIME_COLUMN: [0.1, 15.0, 15.1], "value": [1.0, 2.0, 3.0]})
        sampled = right_closed_resample(raw, interval_seconds=15, agg="sum", fill_empty=0)
        self.assertEqual(sampled[TIME_COLUMN].tolist(), [15.0, 30.0])
        self.assertEqual(sampled["value"].tolist(), [3.0, 3.0])

    def test_raw_log_aggregation_is_right_closed_and_regular(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logs.csv"
            pd.DataFrame(
                {
                    "timestamp": [0.1, 15.0, 15.1, 45.0],
                    "container_name": ["front-end"] * 4,
                }
            ).to_csv(path, index=False)
            aggregated = aggregate_raw_logs(path, interval_seconds=15)
        self.assertEqual(aggregated[TIME_COLUMN].tolist(), [15.0, 30.0, 45.0])
        self.assertEqual(aggregated["front-end_1"].tolist(), [2, 1, 1])

    def test_train_ticket_auxiliary_service_identity_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logs.csv"
            pd.DataFrame(
                {"timestamp": [1.0], "serviceName": ["ts-route-service"]}
            ).to_csv(path, index=False)
            aggregated = aggregate_raw_logs(path)
        prefixed = prefix_aux_columns(aggregated, "log")
        feature = next(column for column in prefixed if column != TIME_COLUMN)
        self.assertEqual(feature, "ts-route-service_log_ts-route-service_1")
        self.assertEqual(infer_service_name(feature), "ts-route-service")

    def test_rcaeval_loader_uses_unique_ids_and_topology_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            subset = Path(temporary) / "RE1-SS"
            for repeat in ("1", "2"):
                case_dir = subset / "payment_cpu" / repeat
                case_dir.mkdir(parents=True)
                pd.DataFrame(
                    {TIME_COLUMN: [0.0, 1.0, 2.0], "front-end_error": [0.0, 0.0, 1.0]}
                ).to_csv(case_dir / "data.csv", index=False)
                (case_dir / "inject_time.txt").write_text("2\n", encoding="utf-8")
            (subset / "topology.json").write_text(
                json.dumps({"directed": True, "edges": [["front-end", "payment"]]}),
                encoding="utf-8",
            )

            cases = load_rcaeval_cases(subset, include_auxiliary=False)

        self.assertEqual([case.case_id for case in cases], ["RE1-SS/payment_cpu/1", "RE1-SS/payment_cpu/2"])
        self.assertEqual([case.name for case in cases], [case.case_id for case in cases])
        self.assertEqual({case.scenario_id for case in cases}, {"RE1-SS/payment_cpu"})
        self.assertEqual({case.system for case in cases}, {"sock-shop"})
        self.assertEqual({case.inject_time for case in cases}, {2.0})
        self.assertEqual(cases[0].topology["edges"], [["front-end", "payment"]])
        self.assertEqual(cases[0].metadata["edges"], [["front-end", "payment"]])

    def test_label_manifest_must_cover_every_loaded_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            subset = Path(temporary) / "RE1-SS"
            case_dir = subset / "payment_cpu" / "1"
            case_dir.mkdir(parents=True)
            pd.DataFrame({TIME_COLUMN: [0.0, 15.0], "front-end_error": [0.0, 1.0]}).to_csv(
                case_dir / "data.csv", index=False
            )
            (case_dir / "inject_time.txt").write_text("15\n", encoding="utf-8")
            with self.assertRaisesRegex(KeyError, "refusing to mix"):
                load_rcaeval_cases(subset, label_manifest={"cases": []}, include_auxiliary=False)

    def test_pending_review_candidates_cannot_be_loaded_as_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            subset = Path(temporary) / "RE1-SS"
            case_dir = subset / "payment_cpu" / "1"
            case_dir.mkdir(parents=True)
            pd.DataFrame({TIME_COLUMN: [0.0, 15.0], "front-end_error": [0.0, 1.0]}).to_csv(
                case_dir / "data.csv", index=False
            )
            (case_dir / "inject_time.txt").write_text("15\n", encoding="utf-8")
            candidate = {
                "cases": [
                    {
                        "case_id": "RE1-SS/payment_cpu/1",
                        "inject_time": 15.0,
                        "censor_time": 30.0,
                        "failure_time": None,
                        "status": "pending_review",
                        "audit_status": "candidate_onset_pending_review",
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "pending-review"):
                load_rcaeval_cases(subset, label_manifest=candidate, include_auxiliary=False)


if __name__ == "__main__":
    unittest.main()
