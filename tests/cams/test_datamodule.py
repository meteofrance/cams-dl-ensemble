import datetime as dt
from pathlib import Path

import pytest
from mfai.pytorch.namedtensor import NamedTensor

from cams.datamodule import CAMSDataModule
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_CAMSDatamodule(tmp_dataset_dir: Path):
    """Test CAMSDataModule initialization."""
    # Create some dummy files to simulate a real dataset
    for day in range(1, 32):
        create_dummy_input_netcdf(
            tmp_dataset_dir
            / f"mocage/2022_07_{day:02}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        )
    create_dummy_target_netcdf(
        tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"
    )

    dates = [dt.datetime(2022, 7, day) for day in range(1, 32)]

    dm = CAMSDataModule(
        batch_size=4,
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )

    # Check default values
    assert dm.batch_size == 4
    assert dm.num_workers == 1
    assert dm.prefetch_factor == 2
    assert dm.train_dataset is None
    assert dm.val_dataset is None

    val_split_size = 5  # Default
    train_split_size = len(dates) - 4 - val_split_size

    # Check date calculations
    assert len(dm.train_dates) == train_split_size
    assert len(dm.val_dates) == val_split_size

    # This should raise an error
    with pytest.raises(ValueError, match="should be either 'fit', 'val', 'validate'"):
        dm.setup("invalid_stage")

    # Setup for fit stage
    dm.setup("fit")

    # Check that train dataset was created
    assert dm.train_dataset is not None
    assert len(dm.train_dataset) == train_split_size  # Should be 22 (31 - 4 - 5)
    assert (
        sorted(dm.train_dataset.run_dates) == dates[:train_split_size]
    )  # 2022-01-01 to 2022-01-22

    # Check that val dataset was also created
    assert dm.val_dataset is not None
    assert len(dm.val_dataset) == val_split_size  # Default 5
    assert (
        sorted(dm.val_dataset.run_dates) == dates[-val_split_size:]
    )  # 2022-01-27 to 2022-01-31

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
        input_data, target_data = sample.get_input_and_target()  # pyright: ignore[reportGeneralTypeIssues]
        batch.append((input_data, target_data))

    # Test collate function
    inputs, targets = dm.collate_batch(batch)

    # Check types
    assert isinstance(inputs, NamedTensor)
    assert isinstance(targets, NamedTensor)

    # Check shapes
    assert inputs.tensor.shape == (2, 1, 420, 700)
    assert targets.tensor.shape == (2, 1, 420, 700)

    # Check names
    assert list(inputs.feature_names) == ["MOCAGE - O3 - +15h - 0m"]
    assert list(targets.feature_names) == ["TARGET - O3 - +15h - 0m"]

    # Check custom start and end dates
    val_split_size = 4
    train_split_size = len(dates) - 4 - val_split_size

    dm = CAMSDataModule(
        processed_dir=tmp_dataset_dir,
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        start_date=dt.datetime(2022, 1, 2),
        val_days=4,
        end_date=dt.datetime(2022, 1, 31),
    )
    assert len(dm.train_dates) == len(dates) - 4 - 4
    assert len(dm.val_dates) == 4
    dm.setup("fit")

    assert dm.train_dataset is not None
    assert len(dm.train_dataset) == len(dm.train_dates)  # Should be 23 (31 - 4 - 4)
    assert (
        sorted(dm.train_dataset.run_dates) == dates[:train_split_size]
    )  # 2022-01-01 to 2022-01-23

    assert dm.val_dataset is not None
    assert len(dm.val_dataset) == len(dm.val_dates)  # Should be 4 (val_days)
    assert (
        sorted(dm.val_dataset.run_dates) == dates[-val_split_size:]
    )  # 2022-01-28 to 2022-01-31


def test_dataloader(tmp_dataset_dir: Path):
    """Test CAMSDataModule initialization."""

    dm = CAMSDataModule(
        batch_size=4,
        models=["MOCAGE"],
        lead_times=[15],
        species=["O3"],
        levels=[0],
        processed_dir=tmp_dataset_dir,
    )
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
