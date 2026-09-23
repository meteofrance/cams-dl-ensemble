import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cams.sample import Sample
from cams.types import Leadtimes
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


@pytest.mark.parametrize(
    "run_date, lead_time",
    [
        (dt.date(2022, 7, 22), 15),
        (dt.date(2023, 1, 1), 3),
        (dt.date(2023, 12, 31), 96),
    ],
)
def test_sample_creation(run_date: dt.date, lead_time: Leadtimes):
    sample = Sample(
        run_date,
        models=["CHIMERE", "MOCAGE"],
        lead_times=[lead_time],
        species=["O3"],
        levels=[0],
    )
    assert sample.date_run == run_date
    assert sample.lead_times == [lead_time]
    expected_valid_times = [
        (
            dt.datetime(run_date.year, run_date.month, run_date.day, 0, 0, 0)
            + dt.timedelta(hours=lead_time)
        )
    ]
    assert sample.valid_times == expected_valid_times


def test_sample_str():
    sample = Sample(
        dt.date(2022, 7, 22),
        models=["CHIMERE", "MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
    )
    result = str(sample)
    assert "2022-07-22" in result
    assert "15" in result


def test_sample_paths():
    sample = Sample(
        dt.date(2022, 7, 22),
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=Path("."),
    )
    assert sample.input_paths == [
        Path("mocage/2022_07_22-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf")
    ]
    assert sample.target_paths == [
        Path("reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc")
    ]


def test_sample_is_valid_false():
    sample = Sample(
        dt.date(2022, 7, 22),
        models=["CHIMERE", "MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
    )
    assert not sample.is_valid


def test_sample_is_valid_true(tmp_dataset_dir: Path):
    # Create dummy files in directories
    input_path = (
        tmp_dataset_dir / "mocage/2022_07_22-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    )
    target_path = tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"

    create_dummy_input_netcdf(input_path)
    create_dummy_target_netcdf(target_path)

    sample = Sample(
        dt.datetime(2022, 7, 22),
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )
    assert sample.is_valid


def test_sample_data(tmp_dataset_dir: Path):
    input_path = (
        tmp_dataset_dir / "mocage/2022_07_22-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    )
    create_dummy_input_netcdf(input_path)
    target_path = tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"
    create_dummy_target_netcdf(target_path)

    sample = Sample(
        dt.datetime(2022, 7, 22),
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )
    data = sample.data

    assert isinstance(data, xr.Dataset)
    assert list(data.data_vars) == ["MOCAGE", "TARGET"]
    assert data["MOCAGE"].values.shape == (1, 1, 1, 420, 700)


def _create_cross_month_target_netcdf(
    path: Path, times: list[dt.datetime], size_lat: int = 420, size_lon: int = 700
) -> None:
    """Create a dummy reanalysis NetCDF with the given valid times."""
    data_shape = (len(times), size_lat, size_lon)
    lats = np.linspace(71.95, 30.05, size_lat)
    lons = np.linspace(-24.95, 44.95, size_lon)
    ds = xr.Dataset(
        {"o3": (["time", "lat", "lon"], np.zeros(data_shape))},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)


def _create_input_netcdf_with_leadtimes(
    path: Path, leadtimes: list[int], size_lat: int = 420, size_lon: int = 700
) -> None:
    """Create a dummy model input NetCDF with the given leadtime hours."""
    data_shape = (len(leadtimes), 1, size_lat, size_lon)
    lats = np.linspace(71.95, 30.05, size_lat)
    lons = np.linspace(-24.95, 44.95, size_lon)
    ds = xr.Dataset(
        {"o3_conc": (["time", "level", "latitude", "longitude"], np.zeros(data_shape))},
        coords={
            "time": leadtimes,
            "level": [0],
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path)


def test_load_input_data_overlapping_two_months(tmp_dataset_dir: Path):
    input_path = (
        tmp_dataset_dir
        / "mocage/2022_07_31-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    )
    _create_input_netcdf_with_leadtimes(input_path, [12, 36])

    sample = Sample(
        dt.datetime(2022, 7, 31),
        models=["MOCAGE"],
        lead_times=[12, 36],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )
    data = sample.load_input_data_for_one_model("MOCAGE")

    assert list(data.time.values) == [
        np.datetime64("2022-07-31T12:00:00"),
        np.datetime64("2022-08-01T12:00:00"),
    ]
    assert data["o3_conc"].values.shape == (2, 1, 420, 700)


def test_sample_data_overlapping_two_months(tmp_dataset_dir: Path):
    target_path_july = tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"
    target_path_august = (
        tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-08.nc"
    )
    _create_cross_month_target_netcdf(
        target_path_july,
        [dt.datetime(2022, 7, 31, 12), dt.datetime(2022, 7, 31, 15)],
    )
    _create_cross_month_target_netcdf(
        target_path_august,
        [dt.datetime(2022, 8, 1, 12)],
    )

    sample = Sample(
        dt.datetime(2022, 7, 31),
        models=["MOCAGE"],
        lead_times=[12, 15, 36],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )

    assert sample.target_paths == [target_path_july, target_path_august]
    target = sample.load_target_data()

    assert list(target.time.values) == [
        np.datetime64("2022-07-31T12:00:00"),
        np.datetime64("2022-07-31T15:00:00"),
        np.datetime64("2022-08-01T12:00:00"),
    ]
    assert target["O3"].values.shape == (3, 1, 420, 700)


def test_sample_is_valid_overlapping_two_months(tmp_dataset_dir: Path):
    input_path = (
        tmp_dataset_dir
        / "mocage/2022_07_31-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    )
    create_dummy_input_netcdf(input_path)
    _create_cross_month_target_netcdf(
        tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc",
        [dt.datetime(2022, 7, 31, 12)],
    )
    _create_cross_month_target_netcdf(
        tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-08.nc",
        [dt.datetime(2022, 8, 1, 12)],
    )

    sample = Sample(
        dt.datetime(2022, 7, 31),
        models=["MOCAGE"],
        lead_times=[12, 36],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )

    assert sample.is_valid
