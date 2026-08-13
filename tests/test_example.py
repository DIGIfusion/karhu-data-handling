import tempfile
import unittest
from pathlib import Path

from karhu_data_handling.convert_sample import convert_sample_to_hdf5


class ConvertSampleTests(unittest.TestCase):
    def test_convert_sample_from_data_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "tests" / "data" / "HelenaRunner-0a8221be-38f7-4679-a937-566e6bf83d5a_scan_2"

        if not data_dir.exists():
            self.skipTest("No data/ directory found in the repository")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "sample.h5"
            convert_sample_to_hdf5(data_dir, output_file=output_file)
            self.assertTrue(output_file.exists())
            self.assertGreater(output_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
