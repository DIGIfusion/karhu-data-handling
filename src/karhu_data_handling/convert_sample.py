from pathlib import Path
import io
import f90nml
import h5py

from .helena_equilibrium import (
    write_h5_equilibrium
)
from .mishka_castor_stability import (
    write_mishka,
    write_castor
)
from .enchanted_sample import (
    write_h5_enchanted_metadata,
    write_h5_enchanted_params,
    write_h5_enchanted_datapoint
)


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


def recreate_helena_fort10(sample_file, output_file="fort.10"):
    """
    Recreate a HELENA fort.10 file from a sample HDF5 file.

    Parameters
    ----------
    sample_file : str or Path
        Path to the sample HDF5 file.

    output_file : str or Path, optional
        Output fort.10 filename.
    """

    NAMELIST_ORDER = [
        "shape",
        "profile",
        "phys",
        "num",
        "pri",
        "plot",
        "ball"
    ]

    sample_file = Path(sample_file)
    output_file = Path(output_file)

    with h5py.File(sample_file, "r") as h5:

        inputs = hdf5_group_to_dict(h5["equilibrium"]["input"])

    nml = f90nml.Namelist()

    for section in NAMELIST_ORDER:
        if section in inputs:
            nml[section] = inputs[section]

    nml.write(output_file, force=True)

    print(f"fort.10 written to {output_file}")


def recreate_mishka_castor_fort10(sample_file, ntor, code="mishka", output_file="fort.10"):
    """
    Recreate a MISHKA/CASTOR fort.10 file from a sample HDF5 file.

    The HDF5 input structure is expected to be:

        input/
            000/
                name = "newrun"
                ...
            001/
                name = "newrun"
                ...
            002/
                name = "newlan"
                ...

    The numerical section names determine the order in which the
    namelists are written.

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

        input_group = h5[f"{code}"][f"n{ntor:03d}"]["input"]

        with open(output_file, "w") as f:

            # HDF5 section names are 000, 001, 002, ...
            for section_key in sorted(input_group.keys()):

                section_group = input_group[section_key]

                # Read the namelist name
                name = section_group.attrs["name"]

                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # Convert the HDF5 group back to a dictionary
                section = hdf5_group_to_dict(section_group)

                # Remove the metadata attribute from the dictionary
                section.pop("name", None)

                # Create a temporary f90nml object for this section
                nml = f90nml.Namelist()
                nml[name] = section

                # f90nml writes the namelist to a temporary string
                # so that we can append multiple namelists.
                buffer = io.StringIO()
                nml.write(buffer)
                text = buffer.getvalue()
                f.write(text)

                # Ensure separation between namelists
                if not text.endswith("\n\n"):
                    f.write("\n")

    print(f"fort.10 written to {output_file}")


def convert_sample_to_hdf5(sample_dir, output_file=None):
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

    with h5py.File(output_file, "w") as h5:

        write_h5_equilibrium(h5, sample_dir)
        write_mishka(h5, sample_dir)
        write_castor(h5, sample_dir)
        write_h5_enchanted_metadata(h5, sample_dir)
        write_h5_enchanted_params(h5, sample_dir)
        write_h5_enchanted_datapoint(h5, sample_dir)

    print(f"Saved {output_file}")
