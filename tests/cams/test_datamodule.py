import datetime as dt
from pathlib import Path

import pytest
from mfai.pytorch.namedtensor import NamedTensor

from cams.datamodule import CAMSDataModule
from cams.settings import MODEL_NAMES


PROCESSED_DIR = Path("tests/data/")

def test_CAMSDatamodule():
    """Test CAMSDataModule initialization."""
    dm = CAMSDataModule(
        batch_size=1, num_days_in_val_set=2, processed_dir=PROCESSED_DIR
    )

    # Check default values
    assert dm.batch_size == 1
    assert dm.num_workers == 1
    assert dm.prefetch_factor == 2
    assert dm.train_dataset is None
    assert dm.val_dataset is None

    # Check date calculations
    assert dm.train_start == dt.datetime(2023, 1, 1)
    assert dm.train_end == dt.datetime(2022, 12, 28)  # val_start - 4 days
    # train_start > train_end, because we don't have enough data for tests

    assert dm.val_start == dt.datetime(2023, 1, 1)  # val_end - 2 days
    assert dm.val_end == dt.datetime(2023, 1, 2)

    # This should raise an error
    with pytest.raises(ValueError, match="should be either 'fit', 'val', 'validate'"):
        dm.setup("invalid_stage")  # type: ignore[reportArgumentType]

    # Setup for fit stage
    dm.setup("fit")

    # Check that train dataset was created
    assert dm.train_dataset is not None
    assert len(dm.train_dataset) == 0

    # Check that val dataset was also created (because of the condition)
    assert dm.val_dataset is not None
    assert len(dm.val_dataset) == 2

    # Get validation dataloader
    val_loader = dm.val_dataloader()
    assert hasattr(val_loader, "__iter__")
    assert val_loader.batch_size == 1

    # Create a mock batch
    batch = []
    for i in range(2):
        sample = dm.val_dataset.samples[i]
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
    assert set(inputs.feature_names) == set(MODEL_NAMES)
    assert list(targets.feature_names) == ["Analysis"]
