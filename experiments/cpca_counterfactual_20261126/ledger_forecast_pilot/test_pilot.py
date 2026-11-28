"""Non-regression checks for causal feature availability and the ledger interface."""
import csv
import io
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_features import build_case


class PilotContracts(unittest.TestCase):
    def archive(self, directory, add_late_fragment=False):
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(['traceID', 'spanID', 'parentSpanID', 'serviceName',
                         'operationName', 'startTime', 'duration'])
        for trace, minute, duration, child_duration in (
                ('normal0', 0, 20, 15), ('normal9', 9, 20, 15), ('fault', 11, 200, 195)):
            start = minute * 60_000_000
            writer.writerow([trace, 'root', '', 'frontend', 'frontend', start, duration])
            writer.writerow([trace, 'child', 'root', 'paymentservice', 'charge', start + 2, child_duration])
        if add_late_fragment:
            writer.writerow(['fault', 'orphan', 'missing', 'emailservice', 'send', 12 * 60_000_000, 1])
        path = Path(directory) / 'data.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('RE2-OB/example/1/traces.csv', text.getvalue())
        return path

    def test_recovery_and_no_warmup_features(self):
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(self.archive(directory)) as archive:
                result, diag = build_case(archive, 'ob', 'RE2-OB/example/1', {0, 540, 660, 720})
        self.assertEqual(diag['cutoff'], 600)
        self.assertEqual(set(result), {660})
        self.assertTrue(math.isclose(result[660]['cpca']['paymentservice'], .9))
        self.assertEqual(result[660]['cpca']['frontendservice'], 0.)

    def test_future_raw_fragment_delays_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(self.archive(directory, True)) as archive:
                result, diag = build_case(archive, 'ob', 'RE2-OB/example/1', {660, 720})
        self.assertNotIn(660, result)
        self.assertIn(720, result)
        self.assertEqual(diag['availability_differs_from_entry_minute'], 1)
        self.assertNotIn('emailservice', result[720]['observed'])


if __name__ == '__main__':
    unittest.main()
