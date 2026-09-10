"""
cd karhu-data-handling
nohup python -u scripts/cleanup.py run_dir_path > cleanup.out &
"""

import os
import sys
from datetime import datetime


def main(run_dir_path):
    run_dirs = sorted([f.path for f in os.scandir(run_dir_path) if f.is_dir()])
    i = 0
    n = len(run_dirs)

    for run_dir in run_dirs[i:]:
        print(f"{datetime.now()} - {i}/{n} - {run_dir}")
        try:
            # Remove all .npy files in run_dir
            removed = 0

            for root, _, files in os.walk(run_dir):
                for filename in files:
                    if filename.endswith(".npy"):
                        filepath = os.path.join(root, filename)
                        os.remove(filepath)
                        removed += 1

            print(f"    Removed {removed} .npy files")

        except Exception as e:
            print(f"({i}) Exception: {e}. Rundir: {run_dir}")
        i += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <run_dir_path>")
        sys.exit(1)

    main(sys.argv[1])