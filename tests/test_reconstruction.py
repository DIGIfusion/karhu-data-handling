import tempfile
import unittest
from pathlib import Path
import numpy as np
import f90nml

from karhu_data_handling.convert_sample import convert_sample
from karhu_data_handling.convert_sample import recreate_fort10

from karhu_data_handling.helena_equilibrium import (
    write_helena_f12_data, read_h5_equilibrium_profiles,
    write_helena_resistivity_file, read_h5_equilibrium_resistivity)


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


class RecreateFort12Tests(unittest.TestCase):

    def test_recreate_fort12(self):
        """
        Test that a fort.12 file is recreated identically from the HDF5 sample.
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

        original_f12 = sample_dir / "fort.12"

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            sample_file = tmpdir / "sample.h5"
            recreated_f12 = tmpdir / "fort.12"

            # Create the sample
            convert_sample(sample_dir, output_file=sample_file)

            # Read equilibrium profiles from the HDF5 sample
            data = read_h5_equilibrium_profiles(sample_file)

            # Recreate fort.12
            write_helena_f12_data(recreated_f12, data)

            # Compare line-by-line
            with open(original_f12, "r") as f:
                original_lines = f.readlines()

            with open(recreated_f12, "r") as f:
                recreated_lines = f.readlines()

            self.assertEqual(
                len(original_lines),
                len(recreated_lines),
                "Number of lines differs.",
            )

            for i, (original, recreated) in enumerate(
                zip(original_lines, recreated_lines),
                start=1,
            ):
                self.assertEqual(
                    original,
                    recreated,
                    f"Difference on line {i}\n"
                    f"Original : {original}"
                    f"Recreated: {recreated}",
                )


class RecreateResistivityTests(unittest.TestCase):

    def test_recreate_resistivity_files(self):
        """
        Test that the CASTOR resistivity and neo files are recreated
        identically from the HDF5 sample.
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

        original_resistivity = sample_dir / "fort.14"

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            sample_file = tmpdir / "sample.h5"
            recreated_resistivity = tmpdir / "fort.14"

            # Create the sample
            convert_sample(sample_dir, output_file=sample_file)

            # Read resistivity profiles from the HDF5 sample
            data = read_h5_equilibrium_resistivity(sample_file)

            # Recreate files
            write_helena_resistivity_file(
                data["s"],
                data["eta_neo"],
                data["deta_e_neo"],
                recreated_resistivity,
            )

            # Compare resistivity
            with open(original_resistivity, "r") as f:
                original_lines = f.readlines()

            with open(recreated_resistivity, "r") as f:
                recreated_lines = f.readlines()

            self.assertEqual(
                len(original_lines),
                len(recreated_lines),
                "resistivity: Number of lines differs.",
            )

            for i, (original, recreated) in enumerate(
                zip(original_lines, recreated_lines),
                start=1,
            ):
                self.assertEqual(
                    original,
                    recreated,
                    f"resistivity differs on line {i}\n"
                    f"Original : {original}"
                    f"Recreated: {recreated}",
                )
