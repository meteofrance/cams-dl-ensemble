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
        num_days_in_val_set: int = 365,
        processed_dir: Path = PROCESSED_DATA_DIR,
        transform_list: list[nn.Module] = [],
    ) -> None:
        """_summary_

        Args:
            batch_size: The batch size. Defaults to 2.
            num_workers: Num of processes to load data from disk. Defaults to 1.
            prefetch_factor: Num of batches loaded in advance by each worker.
                Defaults to 2.
            num_days_in_val_set: The number of days of data from the end of the dataset
                reserved for validation. Defaults to 365 days.
            processed_dir: Path to the CAMS processed dataset.
            transform_list: list of transforms to apply to the data after loading it.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.processed_dir = processed_dir
        self.transform_list = transform_list

        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": self.collate_batch,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "prefetch_factor": self.prefetch_factor,
        }

        run_dates = get_run_dates(self.processed_dir)
        if len(run_dates) == 0:
            raise FileNotFoundError(
                f"CAMS dataset empty: {self.processed_dir / 'input'}"
            )
        print(f"--> {len(run_dates)} runs available in whole dataset.")

        # The val dataset spans 'num_days_in_val_set' days a the end of the period
        self.val_end = run_dates[-1]
        # The last date is included, so the first date is `num_days - 1` days ago:
        self.val_start = self.val_end - dt.timedelta(days=num_days_in_val_set - 1)

        # The train dataset spans the rest of the period, starting at the begining
        self.train_start = run_dates[0]
        # To avoid overlap in train/val sets, we remove the last 4 days from train set
        self.train_end = self.val_start - dt.timedelta(days=4)

        print(f"--> Train dataset: from {self.train_start} to {self.train_end}")
        print(f"--> Val dataset: from {self.val_start} to {self.val_end}")

        self.save_hyperparameters()

    @override
    def setup(self, stage: Literal["fit", "val", "validate"]) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        """Called by lightning, at the start of a stage.

        Args:
            stage: Selects which dataset is loaded,
                either 'fit', 'val', 'validate' or 'test'.
        """
        if stage == "fit":
            self.train_dataset = (
                CAMSDataset(
                    self.train_start,
                    self.train_end,
                    self.processed_dir,
                    self.transform_list,
                )
                if self.train_dataset is None
                else self.train_dataset
            )
            print("--> Train dataset length: ", len(self.train_dataset))
        if stage in ["fit", "val", "validate"]:
            self.val_dataset = (
                CAMSDataset(
                    self.val_start,
                    self.val_end,
                    self.processed_dir,
                    self.transform_list,
                )
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
    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader"""
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)  # type: ignore[reportArgumentType]

    @override
    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader"""
        if self.val_dataset is None:
            self.setup("val")
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)  # type: ignore[reportArgumentType]

    def collate_batch(
        self,
        batch: list[tuple[NamedTensor, NamedTensor]],
    ) -> tuple[NamedTensor, NamedTensor]:
        """Collates a batch of NamedTensor data."""

        inputs = NamedTensor.collate_fn([item[0] for item in batch])
        targets = NamedTensor.collate_fn([item[1] for item in batch])

        return inputs, targets


if __name__ == "__main__":
    dm = CAMSDataModule()
    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))
    print(x, y)
