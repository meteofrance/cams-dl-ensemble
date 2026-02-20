"""Script that opens sample data and opens pdb to allow you to inspect it.
The sample data opened it one raw input, target, processed input and target
files.
"""

import earthkit.data as ekd
from random import choice
from pathlib import Path
import xarray as xr
from cams.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

# Build file paths
raw_input_path: Path = choice(list(RAW_DATA_DIR.glob("**/*.grib")))
raw_target_path: Path = choice(list(RAW_DATA_DIR.glob("**/*.netcdf")))
processed_input_path: Path = choice(list(PROCESSED_DATA_DIR.glob("input/*.netcdf")))
processed_target_path: Path = choice(list(PROCESSED_DATA_DIR.glob("target/*.netcdf")))

# Open sample files
raw_input = xr.open_dataarray(raw_input_path)
raw_target: xr.DataArray = (
    ekd.from_source("file", raw_target_path)
    .to_xarray()
    .to_dataarray()[0]
)
processed_input = xr.open_dataarray(processed_input_path)
processed_target = xr.open_dataarray(processed_target_path)

# Open pdb
breakpoint()