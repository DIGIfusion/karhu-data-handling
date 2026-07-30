from pathlib import Path
import os
import math
import json
import datetime
import h5py
import numpy as np
import f90nml

from .utils import (
    save_value,
    save_dict,
)


def read_summary(sample_dir):
    """
    Read summary.json.
    """

    summary_file = Path(sample_dir) / "summary.json"

    with open(summary_file) as f:
        return json.load(f)


def read_fort10(sample_dir):
    """
    Read HELENA fort.10 namelist.
    """

    fort10 = Path(sample_dir) / "fort.10"

    return f90nml.read(fort10)


def get_creation_date(sample_dir):
    """
    Return modification timestamp of fort.10 as ISO string.

    (Linux filesystems generally do not store true creation time.)
    """

    fort10 = Path(sample_dir) / "fort.10"

    timestamp = fort10.stat().st_mtime

    return datetime.datetime.fromtimestamp(timestamp).isoformat()


def write_metadata(h5, summary, sample_dir):
    """
    Write metadata section.
    """

    meta = h5.create_group("metadata")

    summary_meta = summary.get("metadata", {})

    for key in ("h_id", "run_dir"):
        if key in summary_meta:
            save_value(meta, key, summary_meta[key])

    save_value(meta, "creation_date", get_creation_date(sample_dir))


def write_params(h5, summary):
    """
    Write input parameters from summary.json.
    """

    params = summary.get("params", {})

    group = h5.require_group("params")

    save_dict(group, params)


def write_equilibrium_inputs(h5, fort10):
    """
    Save all fort.10 namelists.
    """

    eq = h5.require_group("equilibrium")
    inputs = eq.require_group("inputs")

    save_dict(inputs, fort10)


def read_lines2(lines, start, end):
    return np.array(
        [float(x) for line in lines[start:end] for x in line.split()],
        dtype=np.float32)


def get_f12_data(filename, variables):
    """
    Read selected variables from HELENA fort.12 file.

    Args:
        filename (str): Path to fort.12 file.
        variables (list[str]): List of variable names to read.

    Supported variables:
        JS0, CS, QS, DQS_1, DQEC, DQS, CURJ, DJ0, DJE,
        NCHI, CHI, GEM11, GEM12, CPSURF, RADIUS, GEM33, RAXIS,
        P0, DP0, DPE, RBPHI, DRBPHI0, DRBPHIE, VX, VY, EPS, XOUT, YOUT
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find {filename}")

    lines = open(filename, "r").readlines()

    def read_n_lines(n, size):
        return math.ceil(size / n)

    data = {}

    JS0 = int(lines[0].split()[0])
    JS0_lines = read_n_lines(4, JS0)
    JS0_1_lines = read_n_lines(4, JS0 + 1)

    print(JS0, JS0_lines, JS0_1_lines)

    i = 1  # line counter

    def maybe_read(name, size):
        nonlocal i
        if name in variables:
            result = read_lines2(lines, i, i + size)
            i += size
            return result
        else:
            i += size
            return None

    if 'JS0' in variables:
        data['JS0'] = np.array(JS0, dtype=np.int32)

    if 'CS' in variables:
        data['CS'] = maybe_read('CS', JS0_1_lines)
    else:
        i += JS0_1_lines

    if 'QS' in variables:
        data['QS'] = maybe_read('QS', JS0_1_lines)
    else:
        i += JS0_1_lines

    if {'DQS_1', 'DQEC'} & set(variables):
        DQS_line = lines[i].split()
        if 'DQS_1' in variables:
            data['DQS_1'] = np.array(float(DQS_line[0]), dtype=np.float32)
        if 'DQEC' in variables:
            data['DQEC'] = np.array(float(DQS_line[1]), dtype=np.float32)
    i += 1

    if 'DQS' in variables:
        data['DQS'] = maybe_read('DQS', JS0_lines)
    else:
        i += JS0_lines

    if 'CURJ' in variables:
        data['CURJ'] = maybe_read('CURJ', JS0_1_lines)
    else:
        i += JS0_1_lines

    if {'DJ0', 'DJE'} & set(variables):
        DJ_line = lines[i].split()
        if 'DJ0' in variables:
            data['DJ0'] = np.array(float(DJ_line[0]), dtype=np.float32)
        if 'DJE' in variables:
            data['DJE'] = np.array(float(DJ_line[1]), dtype=np.float32)
    i += 1

    NCHI = int(lines[i].split()[0])
    NCHI_1_lines = read_n_lines(4, NCHI)
    NCHI_JS0_lines = read_n_lines(4, NCHI * (JS0 + 1) - (NCHI + 1))
    i += 1

    print(NCHI, NCHI_1_lines, NCHI_JS0_lines)

    if 'NCHI' in variables:
        data['NCHI'] = np.array(NCHI, dtype=np.int32)

    for var, lines_needed in [('CHI', NCHI_1_lines), ('GEM11', NCHI_JS0_lines),
                              ('GEM12', NCHI_JS0_lines)]:
        if var in variables:
            data[var] = maybe_read(var, lines_needed)
        else:
            i += lines_needed

    if {'CPSURF', 'RADIUS'} & set(variables):
        line = lines[i].split()
        if 'CPSURF' in variables:
            data['CPSURF'] = np.array(float(line[0]), dtype=np.float32)
        if 'RADIUS' in variables:
            data['RADIUS'] = np.array(float(line[1]), dtype=np.float32)
    i += 1

    if 'GEM33' in variables:
        data['GEM33'] = maybe_read('GEM33', NCHI_JS0_lines)
    else:
        i += NCHI_JS0_lines

    if 'RAXIS' in variables:
        data['RAXIS'] = np.array(float(lines[i].split()[0]), dtype=np.float32)
    i += 1

    for var in ['P0', 'RBPHI']:
        if var in variables:
            data[var] = maybe_read(var, JS0_1_lines)
        else:
            i += JS0_1_lines

        if var == 'P0' and {'DP0', 'DPE'} & set(variables):
            line = lines[i].split()
            if 'DP0' in variables:
                data['DP0'] = np.array(float(line[0]), dtype=np.float32)
            if 'DPE' in variables:
                data['DPE'] = np.array(float(line[1]), dtype=np.float32)
        if var == 'RBPHI' and {'DRBPHI0', 'DRBPHIE'} & set(variables):
            line = lines[i].split()
            if 'DRBPHI0' in variables:
                data['DRBPHI0'] = np.array(float(line[0]), dtype=np.float32)
            if 'DRBPHIE' in variables:
                data['DRBPHIE'] = np.array(float(line[1]), dtype=np.float32)
        i += 1

    for var in ['VX', 'VY']:
        if var in variables:
            data[var] = maybe_read(var, NCHI_1_lines)
        else:
            i += NCHI_1_lines

    if 'EPS' in variables:
        data['EPS'] = np.array(float(lines[i].split()[0]), dtype=np.float32)
    i += 1

    for var in ['XOUT', 'YOUT']:
        if var in variables:
            data[var] = maybe_read(var, NCHI_JS0_lines)
        else:
            i += NCHI_JS0_lines

    return data


# Variables to extract from fort.12
FORT12_VARIABLES = [
    "JS0",
    "CS",
    "QS",
    "DQS_1",
    "DQEC",
    "DQS",
    "CURJ",
    "DJ0",
    "DJE",
    "NCHI",
    "CHI",
    "GEM11",
    "GEM12",
    "CPSURF",
    "RADIUS",
    "GEM33",
    "RAXIS",
    "P0",
    "DP0",
    "DPE",
    "RBPHI",
    "DRBPHI0",
    "DRBPHIE",
    "VX",
    "VY",
    "EPS",
    "XOUT",
    "YOUT",
]
import math
import numpy as np


FORT12_LAYOUT = [
    ("JS0", "scalar_int"),
    ("CS", "array", "JS0+1"),
    ("QS", "array", "JS0+1"),
    (("DQS_1", "DQEC"), "pair"),
    ("DQS", "array", "JS0"),
    ("CURJ", "array", "JS0+1"),
    (("DJ0", "DJE"), "pair"),
    ("NCHI", "scalar_int"),
    ("CHI", "array", "NCHI"),
    ("GEM11", "array", "NCHI_JS0"),
    ("GEM12", "array", "NCHI_JS0"),
    (("CPSURF", "RADIUS"), "pair"),
    ("GEM33", "array", "NCHI_JS0"),
    ("RAXIS", "scalar"),
    ("P0", "array", "JS0+1"),
    (("DP0", "DPE"), "pair"),
    ("RBPHI", "array", "JS0+1"),
    (("DRBPHI0", "DRBPHIE"), "pair"),
    ("VX", "array", "NCHI"),
    ("VY", "array", "NCHI"),
    ("EPS", "scalar"),
    ("XOUT", "array", "NCHI_JS0"),
    ("YOUT", "array", "NCHI_JS0"),
]


def _write_array(f, array, values_per_line=4):
    """Write a 1D array with four values per line."""
    array = np.asarray(array).ravel()

    for i in range(0, len(array), values_per_line):
        values = array[i:i + values_per_line]
        line = "".join(f"{v:16.8E}" for v in values)
        f.write(line + "\n")


def _array_size(size_name, JS0, NCHI):
    """Return the expected array length."""

    if size_name == "JS0":
        return JS0

    if size_name == "JS0+1":
        return JS0 + 1

    if size_name == "NCHI":
        return NCHI

    if size_name == "NCHI+1":
        return NCHI + 1

    if size_name == "NCHI_JS0":
        return NCHI * (JS0 + 1) - (NCHI)

    raise ValueError(f"Unknown array size '{size_name}'")


def read_equilibrium_profiles(h5_or_filename):
    """
    Read equilibrium profiles from a sample HDF5 file.

    Parameters
    ----------
    h5_or_filename : h5py.File, h5py.Group, str, or Path
        Either an open HDF5 file (or group) or the path to a sample HDF5 file.

    Returns
    -------
    dict
        Dictionary in the same format expected by ``write_f12_data()``.
    """

    def _read_profiles(profiles):
        data = {}

        for key, dataset in profiles.items():
            value = dataset[()]

            # Convert NumPy scalars to Python scalars
            if isinstance(value, np.generic):
                value = value.item()

            data[key] = value

        return data

    if isinstance(h5_or_filename, (str, bytes)) or hasattr(h5_or_filename, "__fspath__"):
        with h5py.File(h5_or_filename, "r") as h5:
            return _read_profiles(h5["equilibrium"]["profiles"])

    return _read_profiles(h5_or_filename["equilibrium"]["profiles"])


def write_f12_data(filename, data):
    """
    Write a HELENA fort.12 file from the dictionary returned by
    ``get_f12_data()``.
    """

    JS0 = int(data["JS0"])
    NCHI = int(data["NCHI"])

    with open(filename, "w") as f:

        for field, kind, *extra in FORT12_LAYOUT:
            print(f"Writing {field} ({kind}) {extra} {'' if not extra else _array_size(extra[0], JS0, NCHI)}")

            if kind == "scalar_int":
                f.write(f"  {int(data[field])}\n")

            elif kind == "scalar":
                f.write(f"  {float(data[field]):.8e}\n")

            elif kind == "pair":
                a, b = field
                f.write(
                    f"  {float(data[a]):.8e} "
                    f"{float(data[b]):.8e}\n"
                )

            elif kind == "array":

                size_name = extra[0]
                expected = _array_size(size_name, JS0, NCHI)

                array = np.asarray(data[field]).ravel()

                if len(array) != expected:
                    raise ValueError(
                        f"{field}: expected length {expected}, "
                        f"got {len(array)}"
                    )

                _write_array(f, array)

            else:
                raise ValueError(f"Unknown field type '{kind}'")


def write_equilibrium(h5, sample_dir):
    """
    Save all equilibrium information.

    Structure
    ---------
    equilibrium/
        inputs/
            ... fort.10 namelists ...
        profiles/
            ... fort.12 profiles ...
    """

    sample_dir = Path(sample_dir)

    eq = h5.require_group("equilibrium")

    # --------------------------------------------------------
    # fort.10 inputs
    # --------------------------------------------------------

    fort10 = read_fort10(sample_dir)

    inputs = eq.require_group("inputs")
    save_dict(inputs, fort10)

    # --------------------------------------------------------
    # fort.12 profiles
    # --------------------------------------------------------

    fort12 = sample_dir / "fort.12"

    if fort12.exists():

        profiles = eq.require_group("profiles")

        data = get_f12_data(fort12, FORT12_VARIABLES)

        for key, value in data.items():
            save_value(profiles, key, value)

    else:
        print(f"Warning: {fort12} not found.")
