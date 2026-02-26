import datetime as dt
from functools import cached_property
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from typing_extensions import override

from cams.datamodule import CAMSDataModule
from cams.dataset import CAMSDataset
from cams.sample import NamedTensor, Sample
from cams.settings import MODEL_NAMES, SIZE_LAT, SIZE_LON


@pytest.fixture(scope="module")
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for tests and clean it up after."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


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
    # Data shape = (model, species, level, leadtime, latitude, longitude)
    data_shape = (len(MODEL_NAMES), 1, 1, 1, SIZE_LAT, SIZE_LON)
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
        import io

        buffer = io.BytesIO()
        ds.to_netcdf(buffer)
        buffer.seek(0)
        with open(path, "wb") as f:
            f.write(buffer.getvalue())


def create_dummy_target_netcdf(path: Path):
    """Create a dummy NetCDF file filled with zeros."""
    data_shape = (1, 1, SIZE_LAT, SIZE_LON)  # (species, level, latitude, longitude)
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
        # Si erreur, essaye avec un buffer temporaire
        import io

        buffer = io.BytesIO()
        ds.to_netcdf(buffer)
        buffer.seek(0)
        with open(path, "wb") as f:
            f.write(buffer.getvalue())


class SampleTest(Sample):
    @property
    @override
    def is_valid(self) -> bool:
        """Always returns True, because we use fake data."""
        return True

    @property
    @override
    def input_data(self) -> NamedTensor:
        """Returns fake input ensemble data as a NamedTensor."""
        num_models = 11
        tensor = torch.zeros((num_models, 128, 128))
        names = [str(i) for i in range(num_models)]
        nt = NamedTensor(tensor, ["features", "lat", "lon"], names)
        return nt

    @property
    @override
    def target_data(self) -> NamedTensor:
        """Returns fake target analysis data as a NamedTensor."""
        tensor = torch.zeros((1, 128, 128))
        nt = NamedTensor(tensor, ["features", "lat", "lon"], ["Analysis"])
        return nt


class CAMSDatasetTest(CAMSDataset):
    @property
    @override
    def run_dates(self) -> list[dt.datetime]:
        """Returns a fake list of available run dates for CAMS models"""
        run_dates = [dt.datetime(2000, 1, i) for i in range(1, 32)]
        return run_dates

    @override
    def create_sample(
        self, date_run: dt.datetime, lead_time: int, path: Path
    ) -> Sample:
        return SampleTest(date_run, lead_time, path)


class CAMSDataModuleTest(CAMSDataModule):
    @cached_property
    @override
    def run_dates(self) -> list[dt.datetime]:
        """Returns a fake list of available run dates for CAMS models"""
        return [dt.datetime(2000, 1, i) for i in range(1, 32)]

    @override
    def create_dataset(
        self, start: dt.datetime, end: dt.datetime, dir: Path
    ) -> CAMSDataset:
        return CAMSDatasetTest(start, end, dir)
