import datetime as dt
from pathlib import Path

import pytest
import xarray as xr

from cams.sample import Sample
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


@pytest.mark.parametrize(
    "run_date, lead_time",
    [
        (dt.date(2022, 7, 22), 15),
        (dt.date(2023, 1, 1), 3),
        (dt.date(2023, 12, 31), 96),
    ],
)
def test_sample_creation(run_date: dt.date, lead_time: int):
    sample = Sample(
        run_date,
        models=["CHIMERE", "MOCAGE"],
        lead_times=[lead_time],
        species=["O3"],
        levels=[0],
    )
    assert sample.date_run == run_date
    assert sample.lead_times == [lead_time]
    expected_valid_times = [run_date + dt.timedelta(hours=lead_time)]
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


def test_sample_wrong_species():
    with pytest.raises(NotImplementedError):
        Sample(
            dt.date(2022, 7, 22),
            models=["CHIMERE", "MOCAGE"],
            lead_times=[15],
            species=["O67"],  # pyright: ignore[reportArgumentType]
            levels=[0],
        )


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
