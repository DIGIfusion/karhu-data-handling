"""
cd karhu-data-handling
nohup python -u scripts/run.py <run_dir_path> <dst_dir_path> > run.out &
"""

import os
import sys
from datetime import datetime
from karhu_data_handling.convert_sample import convert_sample_to_hdf5
from karhu_data_handling.enchanted_sample import read_es_summary


def main(run_dir_path, dst_dir_path):
    run_dirs = sorted([f.path for f in os.scandir(run_dir_path) if f.is_dir()])
    i = 0
    n = len(run_dirs)

    for run_dir in run_dirs[i:]:
        try:
            # Define sample id
            summary = read_es_summary(sample_dir=run_dir)
            # sample_id = summary["params"]["h_dir_beta_N"][0].split('/')[-1]
            sample_id = summary["h_id"]
            print(f"{datetime.now()} - {i}/{n} - {run_dir} - {sample_id}")

            output_file = os.path.join(dst_dir_path, sample_id + ".h5")
            convert_sample_to_hdf5(sample_dir=run_dir, output_file=output_file)
        except Exception as e:
            print(f"({i}) Exception: {e}. Sample: {run_dir}")
        i += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <run_dir_path> <dst_dir_path>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
