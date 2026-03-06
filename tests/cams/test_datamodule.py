import datetime as dt
from pathlib import Path

import pytest
from mfai.pytorch.namedtensor import NamedTensor

from cams.datamodule import CAMSDataModule
from cams.settings import MODEL_NAMES
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_CAMSDatamodule(tmp_dataset_dir: Path):
    """Test CAMSDataModule initialization."""
    # Create some dummy files to simulate a real dataset
    dates = [dt.datetime(2022, 1, i) for i in range(1, 11)]

    for date in dates:
        input_path = tmp_dataset_dir / f"input/{date.strftime('%Y_%m_%d')}.netcdf"
        target_path = tmp_dataset_dir / f"target/{date.strftime('%Y_%m_%d_15')}.netcdf"
        create_dummy_input_netcdf(input_path)
        create_dummy_target_netcdf(target_path)

    dm = CAMSDataModule(
        batch_size=4, num_days_in_val_set=2, processed_dir=tmp_dataset_dir
    )

    # Check default values
    assert dm.batch_size == 4
    assert dm.num_workers == 1
    assert dm.prefetch_factor == 2
    assert dm.train_dataset is None
    assert dm.val_dataset is None

    # Check date calculations
    assert dm.train_start == dt.datetime(2022, 1, 1)
    assert dm.train_end == dt.datetime(2022, 1, 5)  # val_start - 4 days
    assert dm.val_start == dt.datetime(2022, 1, 9)  # val_end - 2 days
    assert dm.val_end == dt.datetime(2022, 1, 10)

    # This should raise an error
    with pytest.raises(ValueError, match="should be either 'fit', 'val', 'validate'"):
        dm.setup("invalid_stage")  # type: ignore[reportArgumentType]

    # Setup for fit stage
    dm.setup("fit")

    # Check that train dataset was created
    assert dm.train_dataset is not None
    assert len(dm.train_dataset) == 5  # 2022-01-01 to 2022-01-02 (minus 4 days overlap)

    # Check that val dataset was also created (because of the condition)
    assert dm.val_dataset is not None
    assert len(dm.val_dataset) == 2  # 2022-01-3 to 2022-01-05

    # Get train dataloader
    train_loader = dm.train_dataloader()
    assert hasattr(train_loader, "__iter__")
    assert train_loader.batch_size == 4

    # Get validation dataloader
    val_loader = dm.val_dataloader()
    assert hasattr(val_loader, "__iter__")
    assert val_loader.batch_size == 4

    # Create a mock batch
    batch = []
    for i in range(2):
        sample = dm.train_dataset.samples[i]
        input_data = sample.input_data
        target_data = sample.target_data
        batch.append((input_data, target_data))

    # Test collate function
    inputs, targets = dm.collate_batch(batch)

    # Check types
    assert isinstance(inputs, NamedTensor)
    assert isinstance(targets, NamedTensor)

    # Check shapes
    assert inputs.tensor.shape == (2, 11, 420, 700)
    assert targets.tensor.shape == (2, 1, 420, 700)

    # Check names
    assert list(inputs.feature_names) == MODEL_NAMES
    assert list(targets.feature_names) == ["O3"]
