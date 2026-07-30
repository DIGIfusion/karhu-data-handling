# karhu-data-handling

A minimal Python package scaffold for converting HELENA-style sample directories into a single HDF5 sample file.

## Project structure

- `src/karhu_data_handling/` - package source
- `examples/` - usage examples and notebooks
- `tests/` - package tests
- `tests/data/` - sample HELENA output data for tests
- `pyproject.toml` - project metadata and dependencies
- `LICENSE` - open source license

## HDF5 sample file structure

The conversion pipeline writes a single HDF5 file with the following layout:

```text
sample.h5
├── metadata
│   ├── h_id
│   ├── run_dir
│   └── creation_date
├── params
│   └── ... sampled input parameters from summary.json
├── equilibrium
│   ├── inputs
│   │   └── ... namelist entries from fort.10
│   ├── profiles
│   │   └── ... equilibrium profile arrays from fort.12
│   └── resistivity
│       ├── s
│       ├── eta_neo
│       ├── eta_spitzer
│       ├── deta_e_neo
│       └── deta_e_spitzer
├── mishka
│   └── nXXX
│       ├── params
│       ├── mpol
│       ├── ntor
│       ├── growthrate
│       └── iterations
└── castor
    └── nXXX
        ├── params
        ├── mpol
        ├── ntor
        ├── growthrate
        └── iterations
```

Each leaf value is stored as an HDF5 dataset, and dictionary-like structures are recursively written as nested groups.
