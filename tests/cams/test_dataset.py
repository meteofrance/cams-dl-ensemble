import datetime as dt
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor

from cams.dataset import CAMSDataset, get_run_dates
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_cams_dataset_creation(tmp_dataset_dir: Path):
    """Test dataset creation with valid samples."""
    # Create dummy files
    for i in range(1, 32):
        create_dummy_input_netcdf(
            tmp_dataset_dir
            / f"mocage/2022_07_{i:02}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        )
    create_dummy_target_netcdf(
        tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"
    )

    dates = [dt.datetime(2022, 7, i) for i in range(1, 32)]

    dataset = CAMSDataset(
        dates,
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )
    assert len(dataset) == 31
    assert len(dataset.samples) == 31
    assert len(dataset.run_dates) == 31

    # Get first item
    x, y = dataset[0]

    # Check types
    assert isinstance(x, NamedTensor)
    assert isinstance(y, NamedTensor)

    # Check shapes
    assert x.tensor.shape == (1, 420, 700)
    assert y.tensor.shape == (1, 420, 700)

    # Check names
    assert list(x.feature_names) == ["MOCAGE - O3 - +15h - 0m"]
    assert list(y.feature_names) == ["TARGET - O3 - +15h - 0m"]


def test_cams_dataset_no_valid_samples(tmp_dataset_dir: Path):
    """Test dataset with no valid samples."""
    # Create input files but no corresponding target files
    for i in range(1, 32):
        create_dummy_input_netcdf(
            tmp_dataset_dir
            / f"mocage/2022_07_{i:02}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        )

    # Add
    create_dummy_input_netcdf(
        tmp_dataset_dir / "mocage/CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    )

    dates = [dt.datetime(2023, 1, i) for i in range(1, 32)]

    dataset = CAMSDataset(
        dates,
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )

    assert len(dataset) == 0
    assert len(dataset.samples) == 0


def test_get_run_dates(tmp_dataset_dir: Path):
    for i in range(1, 32):
        create_dummy_input_netcdf(
            tmp_dataset_dir
            / f"mocage/2022_07_{i:02}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        )

    # Add file without valid name
    create_dummy_input_netcdf(
        tmp_dataset_dir / "mocage/CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    )

    run_dates = get_run_dates(tmp_dataset_dir)
    assert run_dates == [dt.datetime(2022, 7, i) for i in range(1, 32)]


def test_cams_dataset_empty_dates(tmp_dataset_dir: Path):
    """Test dataset with empty list of date."""

    dataset = CAMSDataset(run_dates=[], models=["foo"], processed_dir=tmp_dataset_dir) # pyright: ignore[reportArgumentType]

    assert len(dataset) == 0
    assert len(dataset.samples) == 0
