import datetime as dt
import tempfile
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import pytest
import torch
from collections.abc import Iterator
from mfai.pytorch.namedtensor import NamedTensor
from typing import Literal

from cams.sample import Sample
from cams.settings import MODEL_NAMES


@pytest.fixture(scope="module")
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for tests and clean it up after."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(scope="function")
def setup_cams_directories(temp_dir: Path) -> Iterator[Path]:
    """Set up input and target directories in the temporary directory."""
    input_dir = temp_dir / "input"
    target_dir = temp_dir / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


def create_dummy_input_netcdf(path: Path):
    """Create a dummy NetCDF file filled with zeros."""
    data_shape = (len(MODEL_NAMES), 420, 700)
    lats = np.linspace(71.95, 30.05, data_shape[1])
    lons = np.linspace(-24.95, 44.95, data_shape[2])
    data = np.zeros(data_shape)
    ds = xr.Dataset(
        {
            "O3": (["model", "latitude", "longitude"], data),
        },
        coords={
            "model": MODEL_NAMES,
            "latitude": lats,
            "longitude": lons,
        }
    )
    ds.to_netcdf(path)

def create_dummy_target_netcdf(path: Path):
    """Create a dummy NetCDF file filled with zeros."""
    data_shape = (420, 700)
    lats = np.linspace(71.95, 30.05, data_shape[0])
    lons = np.linspace(-24.95, 44.95, data_shape[1])
    data = np.zeros(data_shape)
    ds = xr.Dataset(
        {
            "O3": (["latitude", "longitude"], data),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        }
    )
    ds.to_netcdf(path)


@pytest.mark.parametrize("run_date, lead_time", [
    (dt.datetime(2022, 7, 22), 15),
    (dt.datetime(2023, 1, 1), 3),
    (dt.datetime(2023, 12, 31), 96),
])
def test_sample_creation(run_date: dt.datetime, lead_time: int):
    sample = Sample(run_date, lead_time)
    assert sample.date_run == run_date
    assert sample.lead_time == lead_time
    expected_valid_time = run_date + dt.timedelta(hours=lead_time)
    assert sample.valid_time == expected_valid_time


def test_sample_str():
    sample = Sample(dt.datetime(2022, 7, 22, 12), 15)
    result = str(sample)
    assert "2022-07-22 12:00" in result
    assert "+15h" in result


def test_sample_paths():
    sample = Sample(dt.datetime(2022, 7, 22), 15)
    assert sample.input_path.name == "2022_07_22.netcdf"
    assert sample.target_path.name == "2022_07_22_15.netcdf"


def test_sample_is_valid_false():
    sample = Sample(dt.datetime(2022, 7, 22, 12), 15)
    assert not sample.is_valid


def test_sample_is_valid_true(setup_cams_directories: Path):
    # Create dummy files in directories
    input_path = setup_cams_directories / "input/2022_07_22.netcdf"
    target_path = setup_cams_directories / "target/2022_07_22_15.netcdf"

    create_dummy_input_netcdf(input_path)
    create_dummy_target_netcdf(target_path)

    sample = Sample(dt.datetime(2022, 7, 22), 15, setup_cams_directories)
    assert sample.is_valid


def test_sample_input_data(setup_cams_directories: Path):
    input_path = setup_cams_directories / "input/2022_07_22.netcdf"
    create_dummy_input_netcdf(input_path)

    sample = Sample(dt.datetime(2022, 7, 22), 15, setup_cams_directories)
    data = sample.input_data

    assert isinstance(data, NamedTensor)
    assert data.tensor.shape == (11, 420, 700)
    assert data.feature_names == MODEL_NAMES


def test_sample_target_data(setup_cams_directories: Path):
    target_path = setup_cams_directories / "target/2022_07_22_15.netcdf"
    create_dummy_target_netcdf(target_path)

    sample = Sample(dt.datetime(2022, 7, 22), 15, setup_cams_directories)
    data = sample.target_data

    assert isinstance(data, NamedTensor)
    assert data.tensor.shape == (1, 420, 700)
    assert list(data.feature_names) == ["Analysis"]