
import h5py
import numpy as np

DATASET_KWARGS = dict(
    compression="gzip",
    compression_opts=4,
    shuffle=True,
    fletcher32=True,
)


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

        # if arr.dtype.kind in ("U", "O"):
        #     dt = h5py.string_dtype("utf-8")
        #     group.create_dataset(key, data=arr.astype(str), dtype=dt)
        if arr.dtype.kind in ("U", "O"):
            dt = h5py.string_dtype("utf-8")
            group.create_dataset(
                key,
                data=np.asarray(arr, dtype=object),
                dtype=dt,
            )
        else:
            kwargs = DATASET_KWARGS if arr.ndim > 0 else {}
            group.create_dataset(key, data=arr, **kwargs)

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
            sub = group.require_group(key)
            save_dict(sub, value)

        else:
            save_value(group, key, value)


def normalize_f90nml_value(value):
    """
    Convert f90nml values into consistent Python/NumPy types.

    In particular, convert Fortran complex values represented as
    strings such as '(0.009979, -7.447e-08)' into Python complex numbers.
    The f90nml package reads the complex value as a string if it is negative.
    """

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, np.ndarray):

        if value.dtype.kind in "biufc":
            return value

        values = [
            normalize_f90nml_value(x)
            for x in value.flat
        ]

        try:
            return np.asarray(values).reshape(value.shape)
        except (ValueError, TypeError):
            return values

    if isinstance(value, (list, tuple)):

        values = [
            normalize_f90nml_value(x)
            for x in value
        ]

        try:
            return np.asarray(values)
        except (ValueError, TypeError):
            return values

    if isinstance(value, str):

        value = value.strip()

        # Fortran complex notation:
        # (real, imaginary)
        if value.startswith("(") and value.endswith(")"):
            content = value[1:-1]

            if "," in content:
                real, imaginary = content.split(",", 1)

                try:
                    return complex(
                        float(real.strip()),
                        float(imaginary.strip()),
                    )
                except ValueError:
                    pass

        return value

    return value
