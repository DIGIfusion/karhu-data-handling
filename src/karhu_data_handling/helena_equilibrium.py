
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


def read_helena_fort10(sample_dir):
    """
    Read HELENA fort.10 namelist.
    """

    fort10 = Path(sample_dir) / "fort.10"

    return f90nml.read(fort10)


def write_h5_scalars(h5, summary):
    """
    Write scalar values section.
    """

    scalars = h5.create_group("scalars")

    for key in ("alpha_max_helena", "b0", "ballooning_stable",
                "betan", "betap", "bmag", "bt", "bvac", "d_ped_ne",
                "d_ped_te", "ip", "jphi_max",
                "mercier_stable", "n_eped", "psi_maxalpha_helena",
                "q_at_boundary", "q_on_axis", "radius", "rmag",
                "run_dir", "rvac", "shear_min_helena",
                "t_eped", "total_area", "total_current",
                "total_volume"):
        if key in summary:
            save_value(scalars, key, summary[key])


def _read_lines(lines, start, end):
    return np.array(
        [float(x) for line in lines[start:end] for x in line.split()],
        dtype=np.float32)


def read_helena_f12_data(filename, variables):
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

    # print(JS0, JS0_lines, JS0_1_lines)

    i = 1  # line counter

    def maybe_read(name, size):
        nonlocal i
        if name in variables:
            result = _read_lines(lines, i, i + size)
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

    # print(NCHI, NCHI_1_lines, NCHI_JS0_lines)

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


def read_h5_equilibrium_profiles(h5_or_filename):
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

    if (isinstance(h5_or_filename, (str, bytes)) or
            hasattr(h5_or_filename, "__fspath__")):
        with h5py.File(h5_or_filename, "r") as h5:
            return _read_profiles(h5["equilibrium"]["profiles"])

    return _read_profiles(h5_or_filename["equilibrium"]["profiles"])


def read_h5_equilibrium_resistivity(h5_or_filename):
    """
    Read equilibrium resistivity data from a sample HDF5 file.

    Parameters
    ----------
    h5_or_filename : h5py.File, h5py.Group, str, or Path
        Either an open HDF5 file (or group) or the path to a sample HDF5 file.

    Returns
    -------
    dict
        Dictionary containing resistivity data.
    """

    def _read_resistivity(resistivity):
        data = {}

        for key, dataset in resistivity.items():
            value = dataset[()]

            # Convert NumPy scalars to Python scalars
            if isinstance(value, np.generic):
                value = value.item()

            data[key] = value

        return data

    if (isinstance(h5_or_filename, (str, bytes)) or
            hasattr(h5_or_filename, "__fspath__")):
        with h5py.File(h5_or_filename, "r") as h5:
            return _read_resistivity(h5["equilibrium"]["resistivity"])

    return _read_resistivity(h5_or_filename["equilibrium"]["resistivity"])


def write_helena_f12_data(filename, data):
    """
    Write a HELENA fort.12 file from the dictionary returned by
    ``read_helena_f12_data()``.
    """

    JS0 = int(data["JS0"])
    NCHI = int(data["NCHI"])

    with open(filename, "w") as f:

        for field, kind, *extra in FORT12_LAYOUT:
            print(
                f"Writing {field} ({kind}) {extra} "
                f"{'' if not extra else _array_size(extra[0], JS0, NCHI)}")

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


def check_if_successful(fort20):
    """
    Check whether a HELENA run was successful.
    HELENA is considered unsuccessful if the second-to-last line
    of fort.20 contains the string "REST".

    Parameters
    ----------
    fort20 : str or Path
        Path to the fort.20 file.

    Returns
    -------
    bool
        True if the run was successful, False otherwise.
    """

    fort20 = Path(fort20)

    if not fort20.exists():
        return False

    with fort20.open("r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return False

    # HELENA has failed if the second-last row contains "REST"
    if "REST" in lines[-2]:
        return False

    return True


def write_h5_equilibrium(h5, sample_dir):
    """
    Save all equilibrium information.

    Structure
    ---------
    equilibrium/
        input/
            ... fort.10 namelists ...
        profiles/
            ... fort.12 profiles ...
    """

    sample_dir = Path(sample_dir)

    eq = h5.require_group("equilibrium")

    # --------------------------------------------------------
    # fort.10 input
    # --------------------------------------------------------

    fort10 = read_helena_fort10(sample_dir)

    input_group = eq.require_group("input")
    save_dict(input_group, fort10)

    # --------------------------------------------------------
    # fort.12 profiles
    # --------------------------------------------------------

    fort12 = sample_dir / "fort.12"

    if fort12.exists():

        profiles = eq.require_group("profiles")

        data = read_helena_f12_data(fort12, FORT12_VARIABLES)

        for key, value in data.items():
            save_value(profiles, key, value)

    else:
        print(f"Warning: {fort12} not found.")

    # --------------------------------------------------------
    # fort.20 profiles and scalars
    # --------------------------------------------------------

    fort20 = sample_dir / "fort.20"
    # Check if successful
    success = check_if_successful(fort20)
    save_value(eq, "success", success)

    if not success:
        return

    if fort20.exists():

        # Beta section
        data = read_helena_fort20_beta_section(fort20)
        scalars_group = eq.require_group("scalars")
        for key, value in data.items():
            save_value(scalars_group, key, value)

        # Real world table
        nrmap = fort10["num"]["nrmap"]
        data = read_helena_realworld_table(fort20, NRMAP=nrmap)

        realworld_group = eq.require_group("realworld")
        for key, value in data.items():
            save_value(realworld_group, key, value)

        # Read table
        data = read_helena_current_volume_area_table(fort20)
        current_volume_area_group = eq.require_group("current_volume_area")

        for key, value in data.items():
            save_value(current_volume_area_group, key, value)

        # Resistivity profiles
        (s,
         eta_neo,
         eta_spitzer,
         deta_e_neo,
         deta_e_spitzer) = extract_helena_resistivity(run_dir=sample_dir)

        resistivity_group = eq.require_group("resistivity")

        save_value(resistivity_group, "s", s)
        save_value(resistivity_group, "eta_neo", eta_neo)
        save_value(resistivity_group, "deta_e_neo", deta_e_neo)
        save_value(resistivity_group, "eta_spitzer", eta_spitzer)
        save_value(resistivity_group, "deta_e_spitzer", deta_e_spitzer)

        # Ballooning table
        data = read_helena_ballooning_table(fort20)
        ballooning_group = eq.require_group("ballooning")

        save_value(ballooning_group, "psi", data["flux"])
        save_value(ballooning_group, "rho", data["rho"])
        save_value(ballooning_group, "q", data["q"])
        save_value(ballooning_group, "shear", data["shear"])
        save_value(ballooning_group, "alpha", data["alpha"])
        save_value(ballooning_group, "fmarg", data["fmarg"])

        # Current density table
        (psi_list,
         jphi_list,
         jphi_average,
         jphi_max,
         jphi_last) = read_helena_jphi_table(fort20, fort10)
        current_density_group = eq.require_group("current_density")
        save_value(current_density_group, "psi", psi_list)
        save_value(current_density_group, "jphi", jphi_list)
        save_value(current_density_group, "jphi_average", jphi_average)
        save_value(current_density_group, "jphi_max", jphi_max)
        save_value(current_density_group, "jphi_last", jphi_last)
    else:
        print(f"Warning: {fort20} not found.")


def read_helena_fort20_beta_section(filename):
    """
    ***************************************
    MAGNETIC AXIS :   0.01908  0.00000
    POLOIDAL BETA :   0.1198E+00
    TOROIDAL BETA :   0.3802E-02
    BETA STAR     :   0.4250E-02
    NORM. BETA    :   0.00335
    TOTAL CURRENT :   0.1428E+01
    TOTAL AREA    :   0.5115E+01
    TOTAL VOLUME  :   0.3110E+02
    INT. INDUCTANCE :  0.685990E+00
    POL. FLUX     :   0.2130E+01
    A,B,C         :   0.4176E+01  0.1522E-01  0.1000E+01
    ***************************************
    """
    data = {}
    file = open(filename, "r")
    lines = file.readlines()
    for line in lines:
        # line = file.readline()
        if line.find("NORM. BETA") > -1:
            spl = line.split(":")
            data["betan"] = float(spl[1]) * 100
        if line.find("POLOIDAL BETA") > -1:
            spl = line.split(":")
            data["betap"] = float(spl[1])
        if line.find("TOTAL CURRENT") > -1:
            spl = line.split(":")
            data["total_current"] = float(spl[1])
        if line.find("TOTAL AREA") > -1:
            spl = line.split(":")
            data["total_area"] = float(spl[1])
        if line.find("TOTAL VOLUME") > -1:
            spl = line.split(":")
            data["total_volume"] = float(spl[1])
        if line.find("TOROIDAL BETA") > -1:
            spl = line.split(":")
            data["beta_tor"] = float(spl[1])
        if line.find("BETA STAR") > -1:
            spl = line.split(":")
            data["beta_star"] = float(spl[1])
        if line.find("PED. BETAPOL") > -1:
            spl = line.split(":")
            data["helena_betap"] = float(spl[1])
        if line.find("A,B,C") > -1:
            spl = line.split(":")
            sp2 = spl[1].split()
            data["b_last_round"] = float(sp2[1])
        if line.find("RADIUS") > -1:
            spl = line.split(":")
            data["radius"] = float(spl[1])
        if line.find("B0") > -1:
            spl = line.split(":")
            data["b0"] = float(spl[1])
            break
    file.close()
    return data


def extract_helena_resistivity(run_dir: str):
    """
    __author__ = "Hampus Nyström"

    Args:
        run_dir (str): Path to the HELENA output directory containing fort.20.

    Returns:
        s (np.ndarray): Normalized flux surface label.
        eta_neo (np.ndarray): Neoclassical resistivity profile.
        eta_spitzer (np.ndarray): Spitzer resistivity profile.
        deta_e_neo (float): Derivative of neoclassical resistivity at the edge.
        deta_e_spitzer (float): Derivative of Spitzer resistivity at the edge.

    """
    # Initializing arrays for storing data
    s = []
    spitzer = []
    neo = []

    filepath = os.path.join(run_dir, "fort.20")
    with open(filepath, "r", encoding="utf-8") as f:
        # Find major radius
        for line in f:
            if "MAJOR RADIUS" in line:
                break
        else:
            raise ValueError("Could not find 'MAJOR RADIUS' in fort.20")

        r = float(line.split()[-2])

        # Magnetic field is on the next line
        line = next(f, None)
        if line is None:
            raise ValueError("Unexpected end of file after 'MAJOR RADIUS'")
        b0 = float(line.split()[-2])

        # Find conductivity table
        for line in f:
            if "SIG(Spitz)" in line:
                break
        else:
            raise ValueError("Could not find 'SIG(Spitz)' in fort.20")

        # Skip header line
        next(f, None)

        # Read conductivity data
        first = True
        for line in f:
            spl = line.split()
            if len(spl) != 7:
                break

            if first:
                rho0 = float(spl[2]) * 1e19
                first = False

            s.append(float(spl[0]))
            spitzer.append(float(spl[5]))
            neo.append(float(spl[6]))

        if first:
            raise ValueError("No conductivity data found in fort.20")

    # Calculating resistivity data and normalizing to CASTOR standard
    eta_neo = np.array(neo)
    eta_spitzer = np.array(spitzer)

    mu0 = 4e-7 * np.pi
    amu = 1.672623e-27
    mdeut = 2.01400 * amu
    rho0 = mdeut * rho0
    norm_constant = np.sqrt(rho0 / mu0) / (r * b0)

    eta_neo = norm_constant / eta_neo
    eta_spitzer = norm_constant / eta_spitzer
    deta_e_neo = (eta_neo[-1] - eta_neo[-2]) / (s[-1] - s[-2])
    deta_e_spitzer = (eta_spitzer[-1] - eta_spitzer[-2]) / (s[-1] - s[-2])

    # adding point in s = 1
    s = np.append(s, 1.0)
    eta_neo = np.append(eta_neo, eta_neo[-1] + deta_e_neo * (s[-1] - s[-2]))
    eta_spitzer = np.append(
        eta_spitzer, eta_spitzer[-1] + deta_e_spitzer * (s[-1] - s[-2]))

    return s, eta_neo, eta_spitzer, deta_e_neo, deta_e_spitzer


def write_helena_resistivity_file(s, eta, deta_e, outputpath):
    """
    Write resistivity data to a file in the specified format as
    taken as input by CASTOR.
    """

    # Writing resistivity data to output
    with open(outputpath, "w", encoding="utf-8") as f:
        # number of grid points
        f.write(f"  {len(s) - 1}")

        # write s array (4 values per row)
        for i, val in enumerate(s):
            if i % 4 == 0:
                f.write("\n")
            f.write(f"  {val:.8e}")

        # write eta array (4 values per row)
        for i, val in enumerate(eta):
            if i % 4 == 0:
                f.write("\n")
            f.write(f"  {val:.8e}")

        # final line
        f.write("\n")
        f.write(f"  {0:.8e}  {deta_e:.8e}")

    return


def read_helena_resistivity_file(filepath):
    """
    Read resistivity data from a file in the specified format as
    taken as input by CASTOR.

    Returns:
        s (np.ndarray): Normalized flux surface label.
        eta (np.ndarray): Resistivity profile.
        deta_e (float): Derivative of resistivity at the edge.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # number of grid points
    n_points = int(lines[0].strip())

    # read s array
    s = np.array(
        [float(x) for line in lines[1:1 + (n_points + 3) // 4]
         for x in line.split()])

    # read eta array
    eta_start = 1 + (n_points + 3) // 4
    eta_end = eta_start + (n_points + 3) // 4
    eta = np.array(
        [float(x) for line in lines[eta_start:eta_end]
         for x in line.split()])

    # final line contains deta_e
    deta_e_line = lines[eta_end].split()
    deta_e = float(deta_e_line[1])

    return s, eta, deta_e


def read_helena_alphamax_from_fort20(filepath: str, profile_delta: float):
    data = read_helena_ballooning_table(filepath)

    mask = data["flux"] > (1 - profile_delta)

    alpha = data["alpha"][mask]
    flux = data["flux"][mask]
    shear = data["shear1"][mask]

    alpha_max = alpha.max()
    psi_maxalpha = flux[np.argmax(alpha)]
    shear_min = shear.min()

    return alpha_max, psi_maxalpha, shear_min


def read_helena_ballooning_table(filepath: str):
    """
    Read the ballooning stability table from a HELENA fort.20 file.

    Returns
    -------
    dict
        Dictionary containing one NumPy array per column.
    """

    data = {
        "i": [],
        "flux": [],
        "rho": [],
        "q": [],
        "shear": [],
        "shear1": [],
        "alpha": [],
        "alpha1": [],
        "fmarg": [],
        "ballooning": [],
    }

    with open(filepath, "r") as f:

        # Find table header
        for line in f:
            if "I, FLUX" in line:
                break
        else:
            raise ValueError("Ballooning table not found.")

        # Skip separator line
        next(f)

        # Read rows
        for line in f:
            if "*****" in line or not line.strip():
                break

            spl = line.split()

            if len(spl) < 10:
                continue

            data["i"].append(int(spl[0]))
            data["flux"].append(float(spl[1]))
            data["rho"].append(float(spl[2]))
            data["q"].append(float(spl[3]))
            data["shear"].append(float(spl[4]))
            data["shear1"].append(float(spl[5]))
            data["alpha"].append(float(spl[6]))
            data["alpha1"].append(float(spl[7]))
            data["fmarg"].append(float(spl[8]))
            data["ballooning"].append(spl[9])

    # Convert numeric columns to arrays
    for key in data:
        if key != "ballooning":
            data[key] = np.asarray(data[key])

    return data


def read_helena_jphi_table(filepath_f20: str, f10_namelist):
    """Find max J_phi, J_phi of the last radial point, and <J_phi> from
    HELENA output file.

    Attributes Used
    ---------------
    self.directory : dict
        Dictionary containing all directories for input and output files
    self.inputfile_name : str
        Name of HELENA input file, also used as base for other file names
    self.crashed : bool
        Whether a problem has arised yet

    Attributes Defined
    ------------------
    self.max_psi : float
        Highest value of psi (pressure)
    self.jpi_last : float
        Outermost value of j_phi
    self.jphi_max : float
        Highest value of j_phi
    self.jphi_average : float
        Average value of j_phi
    """

    mu0 = 4e-7 * np.pi
    btvac = f10_namelist['phys']['bvac']
    # rvac = f10_namelist['phys']['rvac']
    a = f10_namelist['phys']['rvac'] * f10_namelist['phys']['eps']
    # eps = a / rvac
    xiab = f10_namelist['phys']['xiab']
    ip = xiab / mu0 * a * btvac
    s_list = []
    psi_list = []
    jphi_list = []
    max_psi = -1
    jphi_max = -1

    with open(filepath_f20) as outputfile:
        for line in outputfile:
            if 'JPHI' in line:  # Not J_phi
                break
        outputfile.readline()

        for line in outputfile:
            if ('*****' in line) or line.isspace():
                break
            spl = line.split()

            # Name of parameter in outputfile: S
            try:
                s_list.append(float(spl[0]))
                psi_list.append(float(spl[0])**2)
            except ValueError:
                print("Failed to extract PSI, S and <J> from " + filepath_f20)
            # Name of parameter in outputfile: JPHI
            try:
                jphi_list.append(float(spl[1]))
            except ValueError:
                print("Failed to extract PSI, S and <J> from " + filepath_f20)

    # >0.92psi is where ELITE defines the pedestal
    pedestalindex = np.argmax(psi_list > (np.float64(0.92)))
    psi_list = psi_list[pedestalindex:]
    for psi in psi_list:
        if psi > max_psi:
            max_psi = psi
    jphi_list = jphi_list[pedestalindex:]
    i = 0
    # print(psi_list)
    for jphi in jphi_list:
        if jphi > jphi_max:
            jphi_max = jphi
            # print(psi_list[i])
        i += 1
    jphi_last = jphi_list[-1]

    # Normalise j_phi
    with open(filepath_f20) as outputfile:
        for line in outputfile:
            if 'TOTAL AREA' in line:
                spl = line.split()
                try:
                    totalarea = float(spl[3])
                except (ValueError, IndexError):
                    print("Total Area could not be found")
                    jphi_average = np.nan
                    jphi_max = np.nan
                    jphi_last = np.nan
                else:
                    jphi_average = ip / (totalarea * a**2)
                    jphi_max = jphi_max / jphi_average
                    jphi_last = jphi_last / jphi_average
                    break
    return psi_list, jphi_list, jphi_average, jphi_max, jphi_last


def get_index_next_empty_line(lines):
    for _i, line in enumerate(lines):
        if len(line) == 0 or line == '\n' or line == ' \n':
            return _i
    return -1


def get_index_next_line_containing_str(lines, text: str):
    for _i, line in enumerate(lines):
        if text in line:
            return _i
    return -1


def read_helena_realworld_table(filename, NRMAP: int = 301):
    """
    **************************************************
        S,   P [Pa], Ne [10^19m^-3], Te [eV],  Ti [eV],
    **************************************************
    """
    npts = NRMAP - 1
    with open(filename, "r") as file:
        lines = file.readlines()

    i_table_start = get_index_next_line_containing_str(
        lines, "S,   P [Pa], Ne [10^19m^-3], Te [eV],  Ti [eV]")
    numerical_lines = lines[i_table_start + 2:i_table_start + 1 + npts]

    # Convert the data to numpy arrays
    data_array = np.array(
        [list(map(float, line.split())) for line in numerical_lines]
    )

    # Split into columns
    data = {
        "S": data_array[:, 0],
        "p": data_array[:, 1],
        "ne": data_array[:, 2],
        "Te": data_array[:, 3],
        "Ti": data_array[:, 4]
    }

    return data


def read_helena_current_volume_area_table(filepath: str):
    """
    Read the HELENA table from fort.20.

    Returns
    -------
    dict
        Dictionary containing one NumPy array per column.
    """

    data = {
        # "i": [],
        "psi": [],
        # "s": [],
        "jphi": [],
        # "error": [],
        # "length": [],
        # "bussac": [],
        "vol": [],
        "volp": [],
        "area": [],
    }

    with open(filepath, "r", encoding="utf-8") as f:

        # Find table header
        for line in f:
            if "I   PSI" in line and "<J>" in line:
                break
        else:
            raise ValueError("Current density table not found.")

        # Skip separator line
        next(f)

        # Read table
        for line in f:
            if "*****" in line or not line.strip():
                break

            spl = line.split()

            if len(spl) != 10:
                break

            # data["i"].append(int(spl[0]))
            data["psi"].append(float(spl[1]))
            # data["s"].append(float(spl[2]))
            data["jphi"].append(float(spl[3]))
            # data["error"].append(float(spl[4]))
            # data["length"].append(float(spl[5]))
            # data["bussac"].append(float(spl[6]))
            data["vol"].append(float(spl[7]))
            data["volp"].append(float(spl[8]))
            data["area"].append(float(spl[9]))

    # Convert to NumPy arrays
    for key in data:
        data[key] = np.asarray(data[key])

    return data
