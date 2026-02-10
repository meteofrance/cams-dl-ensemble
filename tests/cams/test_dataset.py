import datetime as dt
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor
from test_sample import (
    create_dummy_input_netcdf,
    create_dummy_target_netcdf,
)

from cams.dataset import CamsDataset
from cams.settings import MODEL_NAMES


def test_cams_dataset_creation(setup_cams_directories: Path):
    """Test dataset creation with valid samples."""
    # Create dummy files
    input_path1 = setup_cams_directories / "input/2022_01_01.netcdf"
    input_path2 = setup_cams_directories / "input/2022_01_02.netcdf"
    target_path1 = setup_cams_directories / "target/2022_01_01_15.netcdf"
    target_path2 = setup_cams_directories / "target/2022_01_02_15.netcdf"

    create_dummy_input_netcdf(input_path1)
    create_dummy_input_netcdf(input_path2)
    create_dummy_target_netcdf(target_path1)
    create_dummy_target_netcdf(target_path2)

    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime(2022, 1, 3)
    dataset = CamsDataset(start_date, end_date, setup_cams_directories)

    assert len(dataset) == 2
    assert len(dataset.samples) == 2

    # Get first item
    x, y = dataset[0]

    # Check types
    assert isinstance(x, NamedTensor)
    assert isinstance(y, NamedTensor)

    # Check shapes
    assert x.tensor.shape == (11, 420, 700)
    assert y.tensor.shape == (1, 420, 700)

    # Check names
    assert list(x.feature_names) == MODEL_NAMES
    assert list(y.feature_names) == ["Analysis"]


def test_cams_dataset_no_valid_samples(setup_cams_directories: Path):
    """Test dataset with no valid samples."""
    # Create input files but no corresponding target files
    input_path = setup_cams_directories / "input/2022_01_03.netcdf"
    create_dummy_input_netcdf(input_path)

    start_date = dt.datetime(2022, 1, 3)
    end_date = dt.datetime(2022, 1, 4)
    dataset = CamsDataset(start_date, end_date, setup_cams_directories)

    assert len(dataset) == 0
    assert len(dataset.samples) == 0


def test_cams_dataset_filtering_by_date(setup_cams_directories: Path):
    """Test that dataset filters samples by date range."""
    # Create multiple date files
    dates = [
        dt.datetime(2021, 12, 31),  # Should be excluded (before start)
        dt.datetime(2022, 1, 1),  # Should be included
        dt.datetime(2022, 1, 2),  # Should be included
        dt.datetime(2022, 3, 20),  # Should be excluded (after end)
    ]

    for date in dates:
        input_path = (
            setup_cams_directories / f"input/{date.strftime('%Y_%m_%d')}.netcdf"
        )
        target_path = (
            setup_cams_directories / f"target/{date.strftime('%Y_%m_%d_15')}.netcdf"
        )
        create_dummy_input_netcdf(input_path)
        create_dummy_target_netcdf(target_path)

    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime(2022, 3, 19)
    dataset = CamsDataset(start_date, end_date, setup_cams_directories)

    # Only 2 dates should be included (2022-01-01 and 2022-01-02)
    assert len(dataset) == 2
    assert len(dataset.samples) == 2


def test_cams_dataset_empty_range(setup_cams_directories: Path):
    """Test dataset with empty date range."""
    # Create some files but outside the date range
    input_path = setup_cams_directories / "input/2022_01_01.netcdf"
    target_path = setup_cams_directories / "target/2022_01_01_15.netcdf"
    create_dummy_input_netcdf(input_path)
    create_dummy_target_netcdf(target_path)

    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2023, 12, 31)
    dataset = CamsDataset(start_date, end_date, setup_cams_directories)

    assert len(dataset) == 0
    assert len(dataset.samples) == 0
