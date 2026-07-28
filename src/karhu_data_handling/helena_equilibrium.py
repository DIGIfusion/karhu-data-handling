from pathlib import Path
import os
import math
import json
import datetime
import h5py
import numpy as np
import f90nml


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def save_value(group, key, value):
    """
    Save a Python value into an HDF5 group.

    Supports:
        - int
        - float
        - bool
        - str
        - list
        - tuple
        - numpy arrays

    Falls back to string representation for unsupported types.
    """

    if value is None:
        group.attrs[key] = "__NONE__"
        return

    if isinstance(value, (int, float, bool, np.integer, np.floating)):
        group.create_dataset(key, data=value)
        return

    if isinstance(value, str):
        dt = h5py.string_dtype("utf-8")
        group.create_dataset(key, data=value, dtype=dt)
        return

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)

        if arr.dtype.kind in ("U", "O"):
            dt = h5py.string_dtype("utf-8")
            group.create_dataset(key, data=arr.astype(str), dtype=dt)
        else:
            group.create_dataset(key, data=arr)

        return

    # fallback
    dt = h5py.string_dtype("utf-8")
    group.create_dataset(key, data=str(value), dtype=dt)


def save_dict(group, dictionary):
    """
    Recursively save a dictionary into an HDF5 group.
    """

    for key, value in dictionary.items():

        if isinstance(value, dict):
            sub = group.create_group(key)
            save_dict(sub, value)

        else:
            save_value(group, key, value)


# -----------------------------------------------------------------------------
# Reading functions
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Writing functions
# -----------------------------------------------------------------------------

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

    group = h5.create_group("params")

    save_dict(group, params)


def write_equilibrium_inputs(h5, fort10):
    """
    Save all fort.10 namelists.
    """

    eq = h5.create_group("equilibrium")
    inputs = eq.create_group("inputs")

    save_dict(inputs, fort10)


# -----------------------------------------------------------------------------
# Main conversion
# -----------------------------------------------------------------------------

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

    print(f"Saved {output_file}")


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

    eq = h5.create_group("equilibrium")

    # --------------------------------------------------------
    # fort.10 inputs
    # --------------------------------------------------------

    fort10 = read_fort10(sample_dir)

    inputs = eq.create_group("inputs")
    save_dict(inputs, fort10)

    # --------------------------------------------------------
    # fort.12 profiles
    # --------------------------------------------------------

    fort12 = sample_dir / "fort.12"

    if fort12.exists():

        profiles = eq.create_group("profiles")

        data = get_f12_data(fort12, FORT12_VARIABLES)

        for key, value in data.items():
            save_value(profiles, key, value)

    else:
        print(f"Warning: {fort12} not found.")
