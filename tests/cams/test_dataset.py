import datetime as dt
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor

from cams.dataset import CAMSDataset
from cams.settings import MODEL_NAMES
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_cams_dataset_creation(tmp_dataset_dir: Path):
    """Test dataset creation with valid samples."""
    # Create dummy files
    for i in range(1,32):
        create_dummy_input_netcdf(tmp_dataset_dir / f"input/2022_01_{i:02}.netcdf")
        create_dummy_target_netcdf(tmp_dataset_dir / f"target/2022_01_{i:02}_15.netcdf")
    
    dates = [dt.datetime(2022, 1, i) for i in range(1, 32)]

    dataset = CAMSDataset(dates, tmp_dataset_dir)
    assert len(dataset) == 31
    assert len(dataset.samples) == 31
    assert len(dataset.dates_run) == 31

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
    assert list(y.feature_names) == ["O3"]


def test_cams_dataset_no_valid_samples(tmp_dataset_dir: Path):
    """Test dataset with no valid samples."""
    # Create input files but no corresponding target files
    for i in range(1,32):
        create_dummy_input_netcdf(tmp_dataset_dir / f"input/2023_01_{i:02}.netcdf")
    
    dates = [dt.datetime(2023, 1, i) for i in range(1, 32)]

    dataset = CAMSDataset(dates, tmp_dataset_dir)

    assert len(dataset) == 0
    assert len(dataset.samples) == 0


def test_cams_dataset_empty_dates(tmp_dataset_dir: Path):
    """Test dataset with empty list of date."""

    dataset = CAMSDataset([], tmp_dataset_dir)

    assert len(dataset) == 0
    assert len(dataset.samples) == 0
