"""Tests for explicit task semantics and exact multi-minute state targets."""

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.re2 import Re2WindowDataset
from models.factory import build_model, training_policy
from training.tasks import ForecastTask


class ForecastTaskTests(unittest.TestCase):
    def test_task_semantics_are_explicit(self):
        state = ForecastTask("state", 3)
        onset = ForecastTask("first-visible", 5)
        self.assertEqual(state.slug, "state_3m")
        self.assertIn("exactly t+3", state.description)
        self.assertEqual(onset.slug, "first_visible_5m")
        self.assertIn("within (t, t+5", onset.description)
        self.assertEqual(
            state.apply_to_label_definition({})["prediction_horizon_seconds"],
            180,
        )

    def test_model_factory_preserves_architecture_policy(self):
        built = build_model(
            "ltfp",
            feature_columns=("metric_cpu", "log_count", "trace_span_count"),
            num_services=2,
        )
        self.assertEqual(built.checkpoint_config["architecture"], "ltfp")
        self.assertEqual(built.checkpoint_config["feature_groups"], [[0], [1], [2]])
        transition = build_model(
            "ltfp-transition",
            feature_columns=("metric_cpu", "log_count", "trace_span_count"),
            num_services=2,
        )
        self.assertEqual(
            transition.checkpoint_config["failure_logit_decomposition"],
            "latest-minute state prior + full-history temporal residual",
        )
        forecast = build_model(
            "ltfp-forecast",
            feature_columns=(
                "metric_cpu",
                "observed_cpu",
                "log_count",
                "log_observed",
                "trace_span_count",
            ),
            num_services=2,
        )
        self.assertEqual(
            forecast.checkpoint_config["telemetry_forecast_columns"],
            ["metric_cpu", "log_count", "trace_span_count"],
        )
        self.assertEqual(forecast.model.forecast_target_indices, (0, 2, 4))
        self.assertEqual(forecast.model.forecast_observation_indices, (1, 3, None))
        rc2 = build_model(
            "ltfp-rc2",
            feature_columns=("metric_cpu", "log_count", "trace_span_count"),
            num_services=2,
        )
        self.assertIn(
            "prediction_localization_link",
            rc2.checkpoint_config,
        )
        policy = training_policy(
            "instantops-adapted",
            node_loss_weight=1.0,
            rank_loss_weight=2.0,
            rca_method="node-head",
        )
        self.assertEqual(policy.node_loss_weight, 0.0)
        self.assertEqual(policy.rank_loss_weight, 0.0)
        self.assertEqual(policy.rca_method, "masking")


class Re2PointHorizonTests(unittest.TestCase):
    def _build_panel(self, root: Path) -> list[list[str]]:
        services = ("frontend", "root")
        all_stamps: list[str] = []
        stamps_by_case: list[list[str]] = []
        origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for repetition in range(1, 4):
            stamps = [
                (origin + timedelta(minutes=(repetition - 1) * 20 + minute)).isoformat()
                for minute in range(8)
            ]
            stamps_by_case.append(stamps)
            all_stamps.extend(stamps)

        with (root / "service_features_1m.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_utc", "service", "telemetry_valid", "metric_cpu"])
            for stamp in all_stamps:
                for service in services:
                    writer.writerow([stamp, service, 1, 0.1])

        with (root / "next_minute_labels.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "input_minute_utc",
                    "target_minute_utc",
                    "service",
                    "failure_next_minute",
                    "root_cause_next_minute",
                    "entry_observed",
                    "event_id",
                    "root_label_valid",
                ]
            )
            for repetition, stamps in enumerate(stamps_by_case, start=1):
                for minute, stamp in enumerate(stamps[:-1]):
                    failure = int(minute == 3)
                    for service in services:
                        writer.writerow(
                            [
                                stamp,
                                stamps[minute + 1],
                                service,
                                failure,
                                int(failure and service == "root"),
                                1,
                                f"event_{repetition}" if failure else "",
                                1,
                            ]
                        )

        with (root / "case_minutes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_utc", "case_id"])
            for repetition, stamps in enumerate(stamps_by_case, start=1):
                for stamp in stamps:
                    writer.writerow([stamp, f"synthetic_cpu_{repetition}"])

        (root / "feature_schema.json").write_text(
            json.dumps({"services": services, "feature_columns": ["metric_cpu"]})
        )
        epochs = [int(datetime.fromisoformat(stamp).timestamp()) for stamp in all_stamps]
        np.savez_compressed(
            root / "service_adjacency_1m.npz",
            timestamps=np.asarray(epochs),
            services=np.asarray(services),
            adjacency=np.zeros((len(epochs), 2, 2), dtype=np.float32),
        )
        return stamps_by_case

    def test_three_minute_state_uses_exact_future_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stamps = self._build_panel(root)
            one_minute = Re2WindowDataset(root, history=2)
            three_minute = Re2WindowDataset(root, history=2, horizon_minutes=3)

            first_train_end = three_minute.predefined_splits[0][0]
            self.assertEqual(three_minute.timestamps[first_train_end], stamps[0][1])
            self.assertEqual(three_minute.failure_targets[first_train_end], 1)
            self.assertEqual(three_minute.node_targets[first_train_end].tolist(), [0.0, 1.0])
            self.assertEqual(
                three_minute.forecast_end_timestamps[first_train_end],
                stamps[0][4],
            )
            self.assertEqual(three_minute.split_report["horizon_minutes"], 3)
            self.assertEqual(len(three_minute.predefined_splits[0]), 4)
            self.assertEqual(one_minute.failure_targets[first_train_end], 0)


if __name__ == "__main__":
    unittest.main()
