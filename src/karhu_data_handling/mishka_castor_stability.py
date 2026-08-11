from pathlib import Path
import json
import f90nml
import numpy as np
import h5py
from .utils import save_value, save_dict, normalize_f90nml_value


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
    string_dtype = h5py.string_dtype(encoding="utf-8")

    # Loop over all mode-number directories
    for mode_dir in sorted(code_dir.iterdir()):

        if not mode_dir.is_dir():
            continue

        fort10_path = mode_dir / "fort.10"
        fort10 = f90nml.read(fort10_path)

        # -------------------------------------------------
        # Determine ntor from the first newrun section
        # -------------------------------------------------

        newrun = fort10["newrun"]

        if isinstance(newrun, list):
            first_newrun = newrun[0]
        else:
            first_newrun = newrun

        ntor = first_newrun.get("ntor")

        if ntor is None:
            raise ValueError(
                f"Could not find ntor in {fort10_path}"
            )

        ntor = -int(ntor)
        mode_group = code_group.require_group(f"n{ntor:03d}")

        # -------------------------------------------------
        # Code input parameters
        # -------------------------------------------------

        input_group = mode_group.require_group("input")

        # Remove an existing input group so that old sections
        # cannot remain when rewriting a sample.
        for key in list(input_group.keys()):
            del input_group[key]

        # f90nml returns repeated namelists as a list.
        # Iterate over the original order and preserve it.
        section_index = 0

        for name, sections in fort10.items():

            if not isinstance(sections, list):
                sections = [sections]

            for section in sections:

                section = {
                    key: normalize_f90nml_value(value)
                    for key, value in section.items()
                }

                section_group = input_group.require_group(
                    f"{section_index:03d}"
                )
                section_group.attrs.create(
                    "name",
                    name,
                    dtype=string_dtype,
                )

                save_dict(section_group, section)

                section_index += 1

        # -------------------------------------------------
        # code output
        # -------------------------------------------------

        output_group = mode_group.require_group("output")
        fort20_file = mode_dir / "fort.20"
        if fort20_file.exists():
            niter, ew, ewc = read_output_f20(fort20_file)
            save_value(output_group, "niter", niter)
            # save_value(output_group, "ew", ew)    # from fort22
            # save_value(output_group, "ewc", ewc)  # from fort22

        fort22_file = mode_dir / "fort.22"
        if fort22_file.exists():
            fort22_data = read_fort22(fort22_file)
            save_dict(output_group, fort22_data)

        # -------------------------------------------------
        # enchanted-surrogates sampling input parameters
        # -------------------------------------------------

        summary_file = mode_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)

            es_params = mode_group.require_group("enchanted_surrogates_params")
            save_dict(es_params, summary.get("params", {}))


def write_mishka(h5, sample_dir):
    write_stability_code(h5, sample_dir, "mishka")


def write_castor(h5, sample_dir):
    write_stability_code(h5, sample_dir, "castor")


def read_output_f20(filepath):
    niter, ew, ewc = None, None, None

    try:
        with open(filepath, "r") as outfile:
            lines = outfile.readlines()

        for line in lines:
            if line.find("INSTABILITY") > -1:
                ew = float(line.split()[5])
                ewc = float(line.split()[6])
                niter = int(line.split()[4])
                # print('niterations: %i, ew=(%f,%f)'%(niter,ew,ewc))
                break
    except FileNotFoundError:
        print(f"FileNotFoundError: {filepath}")

    return niter, ew, ewc


def read_fort22(filename):
    """
    Read the output file fort.22

    Parameters
    ----------
    ew: real part of growth rate
    ewc: complex part of growth rate
    ng: number of radial grid points
    manz: number of poloidal harmonics
    ngl: ?
    rfour: ?
    sgrid: grid points in the radial direction (s=sqrt(psi))
    ev_list: Every second value is including the complex part

    (rfour(m),m=1,manz)
    write(22,11) (sgrid(i),i=1,ng)
    do 5 i=1,ng
    do 5 j=1,nbg
        rev = real(ev(j,i))
        zev = imag(ev(j,i))
        if (abs(rev).lt.1.e-20) rev=0.
        if (abs(zev).lt.1.e-20) zev=0.
        ev(j,i) = rev + (0.,1.)*zev
    5 continue
    do 10 i=1,ng
        write(22,11) (ev(j,i),j=1,nbg)

    where NGL=2, NBG=2*NGL*MANZ
    """

    data = {}
    with open(filename) as f:
        tok = f.read().split()

    p = 0
    data["ew"] = float(tok[p])
    p += 1
    data["ewc"] = float(tok[p])
    p += 1
    data["ng"] = int(tok[p])
    p += 1  # Grid points
    data["manz"] = int(tok[p])
    p += 1  # Number of poloidal harmonics
    data["ngl"] = int(tok[p])
    p += 1  # Number of variables (0: V1,V2. 1: dV1/dS, v2 midnode)
    data["nbg"] = 2 * data["ngl"] * data["manz"]
    data["neq"] = data["ngl"]

    data["rfour"] = np.asarray(tok[p:p + data["manz"]], float)
    p += data["manz"]

    data["sgrid"] = np.asarray(tok[p:p + data["ng"]], float)
    p += data["ng"]
    data["psi"] = data["sgrid"]**2

    # Reading eigenvectors and saving as flat array
    data["ev"] = np.asarray(tok[p:], float)

    return data
