"""Regression checks for coverage, archive checks and evaluation correctness."""

import csv
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.preflight import check_split_archive
from data.dataset import GraphWindowDataset, purged_event_split
from rca.ranking import masking_importance
from training.engine import fit_feature_scaler, run_epoch


class SafetyTests(unittest.TestCase):
    def test_calendar_split_purges_history_and_crossing_event(self):
        dataset = SimpleNamespace(
            sample_indices=list(range(1, 19)),
            bucket_seconds=60,
            timestamp_epochs=[minute * 60 for minute in range(20)],
            history=2,
        )
        schema = {"label_lookahead_minutes": 1,
                  "events": [{"event_id": "crossing", "start_epoch": 6 * 60,
                              "end_epoch_exclusive": 9 * 60}]}
        (train, validation, test), report = purged_event_split(
            dataset, schema, (7 * 60, 14 * 60)
        )
        self.assertEqual(report["policy"], "calendar_purged_events")
        self.assertEqual(report["crossing_events_excluded"], ["crossing"])
        self.assertEqual(train, [1, 2, 3, 4])
        self.assertEqual(validation, [10, 11, 12])
        self.assertEqual(test, [15, 16, 17, 18])

    def test_available_policy_keeps_trace_only_inputs_but_excludes_unknown_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            times = [(datetime(2021, 7, 1, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()
                     for i in range(7)]
            with (root / "features.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_utc", "service", "metric_cpu", "observed_cpu",
                                 "telemetry_valid", "trace_span_count"])
                for i, stamp in enumerate(times):
                    for service in ("a", "b"):
                        observed = int(i not in (3, 5))
                        writer.writerow([stamp, service, 10 if observed else 0,
                                         observed, observed, int(i == 3)])
            with (root / "labels.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["input_minute_utc", "target_minute_utc", "service",
                                 "failure_next_minute", "root_cause_next_minute", "entry_observed"])
                for i in range(6):
                    for service in ("a", "b"):
                        writer.writerow([times[i], times[i+1], service, 0, 0,
                                         int(i+1 not in (4, 5))])
            arguments = (root / "features.csv", root / "labels.csv", np.eye(2),
                         ("a", "b"), ("metric_cpu", "observed_cpu", "trace_span_count"))
            complete = GraphWindowDataset(*arguments, history=2)
            available = GraphWindowDataset(*arguments, history=2, coverage_policy="available")
            self.assertEqual(complete.sample_indices, [1])
            self.assertEqual(available.sample_indices, [1, 2])

    def test_scaler_ignores_missing_metric_values_and_preserves_observation_mask(self):
        dataset = SimpleNamespace(
            history=1,
            feature_columns=("metric_cpu", "observed_cpu"),
            features=np.array([[[10., 1.]], [[0., 0.]], [[20., 1.]]], dtype=np.float32),
        )
        mean, scale = fit_feature_scaler(dataset, [0, 1, 2])
        self.assertAlmostEqual(float(mean[0]), 15.)
        self.assertAlmostEqual(float(scale[0]), 5.)
        np.testing.assert_array_equal(dataset.features[:, 0, 0], [-1., 0., 1.])
        np.testing.assert_array_equal(dataset.features[:, 0, 1], [1., 0., 1.])

    def test_missing_minutes_exclude_history_and_target_but_observed_zero_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            times = [(datetime(2021, 7, 1, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()
                     for i in range(8)]
            for explicit in (False, True):
                fields = ["timestamp_utc", "service", "cpu", "observed_cpu"]
                if explicit:
                    fields.append("telemetry_valid")
                with (root / "features.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    for i, stamp in enumerate(times):
                        for service in ("a", "b"):
                            valid = int(not (i == 3 and service == "b"))
                            writer.writerow([stamp, service, 0, valid] + ([valid] if explicit else []))
                with (root / "labels.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["input_minute_utc", "target_minute_utc", "service", "fault_next_minute"])
                    writer.writerows([times[i], times[i + 1], service, 0]
                                     for i in range(7) for service in ("a", "b"))
                dataset = GraphWindowDataset(root / "features.csv", root / "labels.csv",
                                             np.eye(2), ("a", "b"), ("cpu", "observed_cpu"), history=2)
                # End 2 targets missing minute 3; ends 3/4 include it in history.
                self.assertEqual(dataset.sample_indices, [1, 5, 6])
                self.assertEqual(dataset.invalid_minutes, 1)

    def test_common_logit_shift_does_not_create_rca_importance(self):
        class CommonShift(nn.Module):
            def forward(self, features, adjacency):
                shift = features.sum(dim=(1, 2, 3))
                return {"failure_logits": torch.stack((shift, shift + 1), dim=-1)}
        scores = masking_importance(CommonShift(), torch.ones(2, 3, 2, 1), torch.zeros(2, 3, 2, 2))
        torch.testing.assert_close(scores, torch.zeros_like(scores))

    def test_probability_drop_ranks_influential_service_first(self):
        class Signal(nn.Module):
            def forward(self, features, adjacency):
                signal = features[:, :, 1].sum(dim=(1, 2))
                return {"failure_logits": torch.stack((torch.zeros_like(signal), signal), dim=-1)}
        scores = masking_importance(Signal(), torch.ones(1, 2, 2, 1), torch.zeros(1, 2, 2, 2))
        self.assertEqual(scores[0, 0].item(), 0)
        self.assertGreater(scores[0, 1].item(), 0)

    def test_weighted_evaluation_loss_is_batch_invariant(self):
        class Fixed(nn.Module):
            def forward(self, features, adjacency):
                value = features[:, 0, 0, 0]
                node = value[:, None]
                return {"failure_logits": torch.stack((-value, value), dim=-1),
                        "node_logits": node, "node_scores": node.sigmoid()}
        values = [-2., -1., 2., 1., .5]
        labels = [0, 0, 1, 1, 1]
        rows = [{"node_features": torch.tensor([[[value]]]), "adjacency": torch.zeros(1, 1, 1),
                 "failure_target": torch.tensor(label), "node_targets": torch.tensor([float(label)])}
                for value, label in zip(values, labels)]
        ce = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.]))
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
        losses = [run_epoch(Fixed(), DataLoader(rows, batch_size=size), torch.device("cpu"),
                            ce, bce, .7)[0] for size in (1, 2, 5)]
        for loss in losses:
            self.assertAlmostEqual(loss, losses[-1], places=6)

    def test_missing_or_empty_split_volume_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "trace_split.zip"
            archive.write_bytes(b"placeholder")
            listing = "Volumes = 3\nPath = trace/trace_table_a_2021-07.csv\n"
            result = subprocess.CompletedProcess([], 0, listing, "")
            with patch("data.preflight.subprocess.run", return_value=result):
                with self.assertRaises(FileNotFoundError):
                    check_split_archive(archive, "7zz", "trace")
                archive.with_suffix(".z01").write_bytes(b"volume")
                archive.with_suffix(".z02").write_bytes(b"")
                with self.assertRaises(FileNotFoundError):
                    check_split_archive(archive, "7zz", "trace")
                archive.with_suffix(".z02").write_bytes(b"volume")
                self.assertEqual(check_split_archive(archive, "7zz", "trace")["volumes"], 3)


if __name__ == "__main__":
    unittest.main()
