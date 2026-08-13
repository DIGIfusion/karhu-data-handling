import os
import json
from pathlib import Path

import pandas as pd
import numpy as np

from .utils import (
    save_value,
    save_dict,
    get_creation_date
)


def write_h5_enchanted_datapoint(h5, sample_dir):
    """
    Save enchanted_datapoint.csv to the HDF5 sample.
    The enchanted_datapoint only has one row.

    Parameters
    ----------
    h5 : h5py.File or h5py.Group
        Open HDF5 file/group.
    sample_dir : str or Path
        Sample directory containing enchanted_datapoint.csv.
    """

    enchanted_datapoint_path = os.path.join(
        sample_dir,
        "enchanted_datapoint.csv",
    )

    if not os.path.exists(enchanted_datapoint_path):
        return

    df = pd.read_csv(enchanted_datapoint_path)
    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one row in {enchanted_datapoint_path}, "
            f"got {len(df)}"
        )
    row = df.iloc[0]

    enchanted_group = h5.require_group("enchanted_datapoint")

    # Store each column as a dataset
    for key, value in row.items():
        save_value(enchanted_group, key, value)


def write_h5_enchanted_params(h5, sample_dir):
    """
    Write input parameters from summary.json.
    """

    summary = read_es_summary(sample_dir)
    es_params = summary.get("params", {})

    group = h5.require_group("enchanted_surrogates_params")

    save_dict(group, es_params)


def write_h5_enchanted_metadata(h5, sample_dir):
    """
    Write metadata section.
    """

    summary = read_es_summary(sample_dir)

    meta = h5.create_group("enchanted_metadata")

    for key in ("h_id", "run_dir", "success", "error"):
        if key in summary:
            save_value(meta, key, summary[key])

    save_value(
        meta, "creation_date",
        get_creation_date(Path(sample_dir) / "summary.json")
    )


def read_es_summary(sample_dir):
    """
    Read summary.json.
    """

    summary_file = Path(sample_dir) / "summary.json"

    with open(summary_file) as f:
        return json.load(f)
