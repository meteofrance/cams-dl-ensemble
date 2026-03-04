import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cams.settings import MODEL_NAMES, SIZE_LAT, SIZE_LON


@pytest.fixture(scope="module")
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for tests and clean it up after."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def tmp_dataset_dir(temp_dir: Path) -> Iterator[Path]:
    """Set up input and target directories in the temporary directory."""
    input_dir = temp_dir / "input"
    target_dir = temp_dir / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


def create_dummy_input_netcdf(path: Path, size_lat: int = SIZE_LAT, size_lon: int = SIZE_LON):
    """Create a dummy NetCDF file filled with zeros."""
    # Data shape = (model, species, level, leadtime, latitude, longitude)
    data_shape = (len(MODEL_NAMES), 1, 1, 1, size_lat, size_lon)
    lats = np.linspace(71.95, 30.05, data_shape[-2])
    lons = np.linspace(-24.95, 44.95, data_shape[-1])
    data = np.zeros(data_shape)
    ds = xr.Dataset(
        {
            "data": (
                ["model", "species", "level", "leadtime", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "model": MODEL_NAMES,
            "species": ["O3"],
            "level": [0],
            "leadtime": [15],
            "latitude": lats,
            "longitude": lons,
        },
    )
    try:
        ds.to_netcdf(path)
    except Exception:
        # If error, tries with a temporary buffer.
        # Caused by concurential access from pytest
        import io

        buffer = io.BytesIO()
        ds.to_netcdf(buffer)
        buffer.seek(0)
        with open(path, "wb") as f:
            f.write(buffer.getvalue())


def create_dummy_target_netcdf(path: Path, size_lat: int = SIZE_LAT, size_lon: int = SIZE_LON):
    """Create a dummy NetCDF file filled with zeros."""
    data_shape = (1, 1, size_lat, size_lon)  # (species, level, latitude, longitude)
    lats = np.linspace(71.95, 30.05, data_shape[-2])
    lons = np.linspace(-24.95, 44.95, data_shape[-1])
    data = np.zeros(data_shape)
    ds = xr.Dataset(
        {"data": (["species", "level", "latitude", "longitude"], data)},
        coords={
            "species": ["O3"],
            "level": [0],
            "latitude": lats,
            "longitude": lons,
        },
    )
    try:
        ds.to_netcdf(path)
    except Exception:
        # If error, tries with a temporary buffer.
        # Caused by concurential access from pytest
        import io

        buffer = io.BytesIO()
        ds.to_netcdf(buffer)
        buffer.seek(0)
        with open(path, "wb") as f:
            f.write(buffer.getvalue())
