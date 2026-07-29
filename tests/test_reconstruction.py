import tempfile
import unittest
from pathlib import Path

import f90nml

from karhu_data_handling.convert_sample import convert_sample
from karhu_data_handling.convert_sample import recreate_fort10

import numpy as np


def assert_namelists_equal(testcase, a, b):
    """
    Recursively compare two namelist dictionaries.
    """

    testcase.assertEqual(set(a.keys()), set(b.keys()))

    for key in a:

        va = a[key]
        vb = b[key]

        if isinstance(va, dict):
            assert_namelists_equal(testcase, va, vb)

        elif isinstance(va, (list, tuple)):
            np.testing.assert_allclose(va, vb)

        elif isinstance(va, float):
            testcase.assertAlmostEqual(va, vb)

        else:
            testcase.assertEqual(va, vb)

class RecreateFort10Tests(unittest.TestCase):

    def test_recreate_fort10(self):
        """
        Test that a fort.10 file can be reconstructed from the HDF5 sample.
        """

        repo_root = Path(__file__).resolve().parents[1]
        sample_dir = (
            repo_root
            / "tests"
            / "data"
            / "HelenaRunner-0a8221be-38f7-4679-a937-566e6bf83d5a_scan_2"
        )

        if not sample_dir.exists():
            self.skipTest("No test data found.")

        original_fort10 = sample_dir / "fort.10"

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            sample_file = tmpdir / "sample.h5"
            recreated_fort10 = tmpdir / "fort.10"

            # Create HDF5 sample
            convert_sample(sample_dir, output_file=sample_file)

            # Recreate fort.10
            recreate_fort10(sample_file, recreated_fort10)

            # Read both files
            original = f90nml.read(original_fort10)
            recreated = f90nml.read(recreated_fort10)
            f90nml.write(recreated, "fort.10.recreated", force=True)

            # Compare the parsed namelists
            assert_namelists_equal(self, dict(original), dict(recreated))
