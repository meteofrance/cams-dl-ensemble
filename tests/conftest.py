import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
import datetime as dt

import numpy as np
import pytest
import xarray as xr
from cams.settings import SIZE_LAT, SIZE_LON

INPUT_SPECIES = (
    "NH3_CO_HCHO_CHOCHO_NO2_NO_NMVOC_O3_PM25_EC_RES_SIA_EC_TOT_PM25_OM_PM10_"
    "DUST_DYNSAL_PM_WF_PANS_SO2"
)


def dummy_input_name(day: str | int) -> str:
    """Build the input NetCDF filename for the given run day."""
    return f"2022_07_{day:02}-{INPUT_SPECIES}-0m-0-96h.netcdf"


@pytest.fixture(scope="module")
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for tests and clean it up after."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def tmp_dataset_dir(temp_dir: Path) -> Iterator[Path]:
    """Set up input and target directories in the temporary directory."""
    input_dir = temp_dir / "mocage"
    target_dir = temp_dir / "reanalysis"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


def create_dummy_input_netcdf(
    path: Path, size_lat: int = SIZE_LAT, size_lon: int = SIZE_LON
):
    """Create a dummy NetCDF file filled with zeros."""
    # Data shape = (level, leadtime, latitude, longitude)
    data_shape = (1, 1, size_lat, size_lon)
    lats = np.linspace(71.95, 30.05, data_shape[-2])
    lons = np.linspace(-24.95, 44.95, data_shape[-1])
    data = np.zeros(data_shape)
    ds = xr.Dataset(
        {
            "o3_conc": (
                ["level", "time", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "level": [0],
            "time": [15],
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


def create_dummy_target_netcdf(
    path: Path, size_lat: int = SIZE_LAT, size_lon: int = SIZE_LON
):
    """Create a dummy NetCDF file filled with zeros."""
    data_shape = (31, size_lat, size_lon)  # (time, latitude, longitude)
    lats = np.linspace(71.95, 30.05, data_shape[-2])
    lons = np.linspace(-24.95, 44.95, data_shape[-1])
    data = np.zeros(data_shape)
    ds = xr.Dataset(
        {"o3": (["time", "lat", "lon"], data)},
        coords={
            "time": [dt.datetime(2022, 7, i, 15) for i in range(1, 32)],
            "lat": lats,
            "lon": lons,
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
