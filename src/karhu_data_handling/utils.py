
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

        if arr.dtype.kind in ("U", "O"):
            dt = h5py.string_dtype("utf-8")
            group.create_dataset(key, data=arr.astype(str), dtype=dt)
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
