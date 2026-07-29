# karhu-data-handling

A minimal Python package scaffold for converting HELENA-style sample directories into a single HDF5 sample file.

## Project structure

- src/karhu_data_handling/ - package source
- tests/ - package tests

## HDF5 sample file structure

The conversion pipeline writes a single HDF5 file with the following layout:

```text
sample.h5
├── metadata
│   ├── h_id
│   ├── run_dir
│   └── creation_date
├── params
│   └── ... input parameters from summary.json
├── equilibrium
│   └── inputs
│       └── ... namelist entries from fort.10
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
