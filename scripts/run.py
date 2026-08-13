import os
from datetime import datetime
from karhu_data_handling.convert_sample import convert_sample_to_hdf5
from karhu_data_handling.enchanted_sample import read_es_summary

run_dir_path = "/scratch/project_2009007/data_JET_pdb/beta_core_slope/rundir_hm/data"
dst_dir_path = "/scratch/project_2009007/data_JET_pdb/beta_core_slope/samples"

run_dirs = sorted([f.path for f in os.scandir(run_dir_path) if f.is_dir()])
i = 0
n = len(run_dirs)

for run_dir in run_dirs[:20]:
    try:
        # Define sample id
        summary = read_es_summary(sample_dir=run_dir)
        sample_id = summary["params"]["h_dir_beta_N"][0].split('/')[-1]
        print(f"{datetime.now()} - {i}/{n} - {run_dir} - {sample_id}")

        output_file = os.path.join(dst_dir_path, sample_id + ".h5")
        convert_sample_to_hdf5(sample_dir=run_dir, output_file=output_file)
    except Exception as e:
        print(f"({i}) Exception: {e}. Sample: {run_dir}")
    i += 1
