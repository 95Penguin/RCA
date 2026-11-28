"""Packaging contracts for the method entry points."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_trace_package_does_not_load_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; sys.path.insert(0, 'src'); from ocsid import Span; "
            "assert 'torch' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_helpers_remain_importable(self):
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        from train_ocsid_joint import _build_arrays
        from ocsid.training import _build_arrays as implementation
        self.assertIs(_build_arrays, implementation)

    def test_lazy_model_export(self):
        from ocsid import OCSIDModel
        from models.ocsid import OCSIDModel as implementation
        self.assertIs(OCSIDModel, implementation)


if __name__ == "__main__":
    unittest.main()
