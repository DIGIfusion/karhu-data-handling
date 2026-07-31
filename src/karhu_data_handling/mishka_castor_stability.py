from pathlib import Path
import json
import f90nml

from .utils import save_value, save_dict


def write_stability_code(h5, sample_dir, code_name):
    """
    Write MISHKA or CASTOR results.

    Parameters
    ----------
    h5 : h5py.File
    sample_dir : Path
    code_name : {"mishka", "castor"}
    """

    sample_dir = Path(sample_dir)

    code_dir = sample_dir / code_name

    if not code_dir.exists():
        return

    code_group = h5.require_group(code_name)

    # Loop over all mode-number directories
    for mode_dir in sorted(code_dir.iterdir()):

        if not mode_dir.is_dir():
            continue

        summary_file = mode_dir / "summary.json"

        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            summary = json.load(f)

        # Try to determine the toroidal mode number
        ntor = summary.get("ntor")

        if ntor is None:
            try:
                ntor = int(mode_dir.name)
            except ValueError:
                ntor = mode_dir.name

        mode_group = code_group.require_group(f"n{int(ntor):03d}")

        # -------------------------------------------------
        # code input parameters
        # -------------------------------------------------
    
        fort10 = f90nml.read(Path(mode_dir) / "fort.10")
        input_group = mode_group.require_group("inputs")
        save_dict(input_group, fort10["newrun"][0])

        # -------------------------------------------------
        # sampling input parameters
        # -------------------------------------------------

        params = mode_group.require_group("params")
        save_dict(params, summary.get("params", {}))

        # -------------------------------------------------
        # selected outputs
        # -------------------------------------------------

        for key in ("mpol", "ntor", "growthrate", "iterations"):

            if key in summary:
                save_value(mode_group, key, summary[key])


def write_mishka(h5, sample_dir):
    write_stability_code(h5, sample_dir, "mishka")


def write_castor(h5, sample_dir):
    write_stability_code(h5, sample_dir, "castor")
