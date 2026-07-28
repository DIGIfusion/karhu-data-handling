from pathlib import Path

import f90nml
import h5py

from .helena_equilibrium import (
    read_fort10,
    read_summary,
    write_equilibrium_inputs,
    write_metadata,
    write_params,
)
from .mishka_castor_stability import write_mishka, write_castor


def hdf5_group_to_dict(group):
    """
    Recursively convert an HDF5 group into a nested Python dictionary.

    Parameters
    ----------
    group : h5py.Group

    Returns
    -------
    dict
        Nested dictionary suitable for constructing a f90nml.Namelist.
    """

    result = {}

    for key, item in group.items():

        if isinstance(item, h5py.Group):
            result[key] = hdf5_group_to_dict(item)

        else:
            value = item[()]

            # Decode bytes
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            # Convert numpy scalars to Python scalars
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass

            # Convert numpy arrays to lists
            if hasattr(value, "tolist"):
                value = value.tolist()

            result[key] = value

    return result


def recreate_fort10(sample_file, output_file="fort.10"):
    """
    Recreate a HELENA fort.10 file from a sample HDF5 file.

    Parameters
    ----------
    sample_file : str or Path
        Path to the sample HDF5 file.

    output_file : str or Path, optional
        Output fort.10 filename.
    """

    sample_file = Path(sample_file)
    output_file = Path(output_file)

    with h5py.File(sample_file, "r") as h5:

        inputs = hdf5_group_to_dict(h5["equilibrium"]["inputs"])

    nml = f90nml.Namelist(inputs)

    nml.write(output_file, force=True)

    print(f"fort.10 written to {output_file}")


def convert_sample(sample_dir, output_file=None):
    """
    Convert one sample directory into one HDF5 file.

    Parameters
    ----------
    sample_dir : str or Path
    output_file : str or Path, optional

    If output_file is None, creates sample.h5 inside sample_dir.
    """

    sample_dir = Path(sample_dir)

    if output_file is None:
        output_file = sample_dir / "sample.h5"

    summary = read_summary(sample_dir)
    fort10 = read_fort10(sample_dir)

    with h5py.File(output_file, "w") as h5:

        write_metadata(h5, summary, sample_dir)
        write_params(h5, summary)
        write_equilibrium_inputs(h5, fort10)
        write_mishka(h5, sample_dir)
        write_castor(h5, sample_dir)

    print(f"Saved {output_file}")
