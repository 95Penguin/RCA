"""Check the two important boundaries of RE2 outcome labeling."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.re2_labels import build_re2_ob_labels  # noqa: E402
from data.re2 import _aggregate_traces  # noqa: E402


class Re2LabelTest(unittest.TestCase):
    def test_entry_rolling_feature_uses_only_completed_current_and_prior_minutes(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "traces.zip"
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=[
                "traceID", "spanID", "parentSpanID", "serviceName",
                "operationName", "startTime", "duration", "statusCode",
            ])
            writer.writeheader()
            for minute in range(7):
                for span in range(3):
                    writer.writerow({
                        "traceID": f"{minute}-{span}", "spanID": str(span),
                        "parentSpanID": "", "serviceName": "frontendservice",
                        "operationName": "frontend", "startTime": (minute * 60 + 1) * 1_000_000,
                        "duration": 300_000 if minute == 5 else 100_000,
                        "statusCode": "",
                    })
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("case/traces.csv", buffer.getvalue())
            with zipfile.ZipFile(archive_path) as archive:
                _, _, _, _, entries = _aggregate_traces(
                    archive, "case", {minute * 60 for minute in range(7)},
                    ("frontendservice",), "RE2-OB", 3, 1.5, 100,
                )
            self.assertEqual(entries[0, "frontendservice"]["entry_p95_ms"], 100)
            self.assertNotIn("observed_entry_baseline", entries[0, "frontendservice"])
            self.assertEqual(entries[3 * 60, "frontendservice"]["observed_entry_baseline"], 1)
            self.assertGreaterEqual(
                entries[5 * 60, "frontendservice"]["entry_degradation_score_rolling"], 1
            )

    def test_injection_without_impact_stays_negative_and_impact_starts_after_injection(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "re2.zip"
            injection = 60_030
            injection_minute = injection // 60 * 60
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, impacted in (("foo_cpu", True), ("bar_cpu", False)):
                    prefix = f"RE2-OB/{name}/1"
                    archive.writestr(f"{prefix}/inject_time.txt", str(injection))
                    buffer = io.StringIO()
                    writer = csv.DictWriter(
                        buffer,
                        fieldnames=["serviceName", "operationName", "startTime", "duration"],
                    )
                    writer.writeheader()
                    for minute in range(injection_minute - 720, injection_minute + 720, 60):
                        high = impacted and minute in {
                            injection_minute,
                            injection_minute + 60,
                            injection_minute + 120,
                        }
                        for _ in range(100):
                            writer.writerow({
                                "serviceName": "frontendservice",
                                "operationName": "frontend",
                                "startTime": (minute + 10) * 1_000_000,
                                "duration": 300_000 if high else 100_000,
                            })
                    archive.writestr(f"{prefix}/traces.csv", buffer.getvalue())
            output = Path(temporary) / "labels"
            summary = build_re2_ob_labels(archive_path, output)
            self.assertEqual(summary["case_count"], 2)
            self.assertEqual(summary["cases_with_observed_impact"], 1)
            self.assertEqual(summary["event_count"], 1)
            with (output / "next_minute_labels.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            positives = [row for row in rows if row["failure_next_minute"] == "1"]
            self.assertEqual(len(positives), 2)
            self.assertTrue(all(row["root_cause_service"] == "foo" for row in positives))
            self.assertTrue(all(row["entry_observed"] == "1" for row in positives))
            self.assertTrue(all(row["phase"] == "post" for row in positives))
            self.assertFalse(any(
                row["case_id"].startswith("RE2-OB_bar_cpu")
                and row["failure_next_minute"] == "1"
                for row in rows
            ))
            self.assertTrue(all(row["entry_observed"] == "0" for row in rows
                                if row["phase"] == "mixed"))


if __name__ == "__main__":
    unittest.main()
