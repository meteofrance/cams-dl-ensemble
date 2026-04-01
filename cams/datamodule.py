import datetime as dt
from pathlib import Path
from typing import Literal

from lightning.pytorch.core import LightningDataModule
from mfai.pytorch.namedtensor import NamedTensor
from torch import nn
from torch.utils.data import DataLoader
from typing_extensions import override

from cams.dataset import CAMSDataset, get_run_dates
from cams.settings import PROCESSED_DATA_DIR
from cams.transforms import ReversibleTransformMixin


class CAMSDataModule(LightningDataModule):
    """
    A Lightning DataModule wrapping the dataset.
    It defines the train/valid/test datasets and their dataloaders.
    """

    train_dataset: CAMSDataset | None = None  # Set at setup
    val_dataset: CAMSDataset | None = None

    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        start_date: dt.datetime | None = None,
        val_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
        processed_dir: Path = PROCESSED_DATA_DIR,
        transforms: list[nn.Module] = [],
    ) -> None:
        """
        Args:
            batch_size: The batch size. Defaults to 2.
            num_workers: Num of processes to load data from disk. Defaults to 1.
            prefetch_factor: Num of batches loaded in advance by each worker.
                Defaults to 2.
            start_date: Dataset start date, inclusive. If None, earliest date
                is selected. Defaults to None.
            val_date: Date after which the data is reserved for validation,
                inclusive. If None, is defined to be 365 days before the end
                date or the date after which there are 30% of the available data
                if there are less than 365 days of data available.
                Defaults to None.
            end_date: Dataset end date, inclusive. If None, latest date is
                selected. Defaults to None.
            processed_dir: Path to the CAMS processed dataset.
            transforms: list of transforms to apply to the data after loading it.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.processed_dir = processed_dir

        # Define a transform and reverse transform sequences
        self.transform_sequence = nn.Sequential(*transforms)
        self.reverse_transform_sequence = nn.Sequential(
            *[
                transform.reverse_transform()
                for transform in reversed(transforms)
                if isinstance(transform, ReversibleTransformMixin)
            ]
        )

        # Define kwargs given to the dataloader class for all datasets
        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": self.collate_batch,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "prefetch_factor": self.prefetch_factor,
        }

        # Gather run dates available
        run_dates: list[dt.datetime] = get_run_dates(self.processed_dir)
        if len(run_dates) == 0:
            raise FileNotFoundError(
                f"CAMS dataset empty: {self.processed_dir / 'input'}"
            )

        # Set dates if they are not defined
        start_date = start_date if start_date else run_dates[0]
        end_date = end_date if end_date else run_dates[-1]
        if val_date is None and end_date - dt.timedelta(days=365) <= start_date:
            val_date = start_date + dt.timedelta(
                days=int((end_date - start_date).days * 0.7)
            )
        elif val_date is None:
            val_date = end_date - dt.timedelta(days=365)
        if not (start_date < val_date < end_date):
            raise ValueError(
                "Given start, val or end values are invalid:\n"
                f"start {start_date} - val {val_date} - end {end_date}"
            )

        # Define training set start and end date
        # To avoid overlap in train/val sets, we remove the last 4 days from train set
        self.train_start = start_date
        self.train_end = val_date - dt.timedelta(days=4)

        # Define validation set start and end date
        self.val_start = val_date
        self.val_end = end_date

        # Sanity checks on the train and validation dates
        train_dates = [
            date for date in run_dates if self.train_start <= date <= self.train_end
        ]
        val_dates = [
            date for date in run_dates if self.val_start <= date <= self.val_end
        ]
        if (
            any(date in val_dates for date in train_dates)
            or any(date in train_dates for date in val_dates)
            or len(train_dates) == 0
            or len(val_dates) == 0
        ):
            raise ValueError(
                "Start and end dates given for CAMS "
                f"dataset {self.processed_dir.parent} are invalid and "
                "produce an empty dataset."
            )

        # Display reports
        print(f"--> {len(run_dates)} runs available in whole dataset.")
        print(f"--> {len(run_dates)} runs available in selected dataset.")
        print(
            f"--> {len(train_dates)} train dates: "
            f"from {self.train_start} to {self.train_end}"
        )
        print(
            f"--> {len(val_dates)} val dates: from {self.val_start} to {self.val_end}"
        )

        self.save_hyperparameters()

    @override
    def setup(self, stage: Literal["fit", "val", "validate"] | str) -> None:
        """Called by lightning, at the start of a stage.

        Args:
            stage: Selects which dataset is loaded,
                either 'fit', 'val', 'validate' or 'test'.
        """
        dataset_kwargs = {
            "processed_dir": self.processed_dir,
            "transform_sequence": self.transform_sequence,
        }
        if stage == "fit":
            self.train_dataset = (
                CAMSDataset(self.train_start, self.train_end, **dataset_kwargs)
                if self.train_dataset is None
                else self.train_dataset
            )
            print("--> Train dataset length: ", len(self.train_dataset))
        if stage in ["fit", "val", "validate"]:
            self.val_dataset = (
                CAMSDataset(self.val_start, self.val_end, **dataset_kwargs)
                if self.val_dataset is None
                else self.val_dataset
            )
            print("--> Val dataset length: ", len(self.val_dataset))
        else:
            raise ValueError(
                "CAMSDatamodule.setup():\n\tparameter stage should be either 'fit', "
                + f"'val', 'validate', got '{stage}'."
            )

    @override
    def train_dataloader(self) -> DataLoader[CAMSDataset]:
        """Returns the train dataloader"""
        if self.train_dataset is None:
            self.setup("fit")
        if self.train_dataset is None:
            raise RuntimeError(
                "Datamodule setup function failed to instantiate a train dataset"
            )

        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    @override
    def val_dataloader(self) -> DataLoader[CAMSDataset]:
        """Returns the validation dataloader"""
        if self.val_dataset is None:
            self.setup("val")
        if self.val_dataset is None:
            raise RuntimeError(
                "Datamodule setup function failed to instantiate a validation dataset"
            )

        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def collate_batch(
        self,
        batch: list[tuple[NamedTensor, NamedTensor]],
    ) -> tuple[NamedTensor, NamedTensor]:
        """Collates a batch of NamedTensor data."""

        inputs = NamedTensor.collate_fn([item[0] for item in batch])
        targets = NamedTensor.collate_fn([item[1] for item in batch])

        return inputs, targets

    def undo_transforms(self, x: NamedTensor, y: NamedTensor):
        """Applies the reverse transforms on the given data."""
        return self.reverse_transform_sequence((x, y))


if __name__ == "__main__":
    dm = CAMSDataModule()
    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))
    print(x, y)
