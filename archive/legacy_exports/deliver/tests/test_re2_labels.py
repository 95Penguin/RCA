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


class Re2LabelTest(unittest.TestCase):
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
