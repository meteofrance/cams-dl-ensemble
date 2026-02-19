import datetime as dt
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor

from cams.dataset import CAMSDataset
from cams.settings import MODEL_NAMES

PROCESSED_DIR = Path("tests/data/")


def test_cams_dataset_creation():
    """Test dataset creation with valid samples."""

    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2023, 1, 3)
    dataset = CAMSDataset(start_date, end_date, PROCESSED_DIR)

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
    assert set(x.feature_names) == set(MODEL_NAMES)
    assert list(y.feature_names) == ["Analysis"]


def test_cams_dataset_no_valid_samples():
    """Test dataset with no valid samples."""
    start_date = dt.datetime(2022, 1, 3)
    end_date = dt.datetime(2022, 1, 4)
    dataset = CAMSDataset(start_date, end_date, PROCESSED_DIR)

    assert len(dataset) == 0
    assert len(dataset.samples) == 0


def test_cams_dataset_filtering_by_date():
    """Test that dataset filters samples by date range."""
    start_date = dt.datetime(2023, 1, 2)
    end_date = dt.datetime(2023, 1, 19)
    dataset = CAMSDataset(start_date, end_date, PROCESSED_DIR)

    # Only 1 date should be included
    assert len(dataset) == 1
    assert len(dataset.samples) == 1


def test_cams_dataset_empty_range():
    """Test dataset with empty date range."""
    start_date = dt.datetime(2023, 2, 1)
    end_date = dt.datetime(2023, 12, 31)
    dataset = CAMSDataset(start_date, end_date, PROCESSED_DIR)

    assert len(dataset) == 0
    assert len(dataset.samples) == 0
