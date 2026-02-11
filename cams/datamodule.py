import datetime as dt
from pathlib import Path
from typing import Literal

from lightning.pytorch.core import LightningDataModule
from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import DataLoader
from typing_extensions import override

from cams.dataset import CamsDataset
from cams.settings import CAMS_DATASET_DIR


class CamsDataModule(LightningDataModule):
    """
    A Lightning DataModule wrapping the dataset.
    It defines the train/valid/test datasets and their dataloaders.
    """

    train_dataset: CamsDataset | None = None  # Set at setup
    val_dataset: CamsDataset | None = None

    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        num_days_in_val_set: int = 365,
        data_dir: Path = CAMS_DATASET_DIR,
    ) -> None:
        """_summary_

        Args:
            batch_size (int, optional): The batch size. Defaults to 2.
            num_workers (int, optional): Num of processes to load data from disk.
                Defaults to 1.
            prefetch_factor (int, optional): Num of batches loaded in advance by
                each worker. Defaults to 2.
            num_days_in_val_set (int, optional): Num of days of data included in the
                validation set. Defaults to 12.
            data_dir (Path, optional): Path to the Cams dataset.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.data_dir = data_dir

        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": self.collate_batch,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "prefetch_factor": self.prefetch_factor,
        }

        list_runs = sorted(list(data_dir.glob("input/*.netcdf")))
        if len(list_runs) == 0:
            raise FileNotFoundError(f"Cams dataset empty: {data_dir / 'input'}")

        print(f"{len(list_runs)} elements in whole dataset.")
        list_dates = [dt.datetime.strptime(path.stem, "%Y_%m_%d") for path in list_runs]
        self.val_end = list_dates[-1]
        self.val_start = self.val_end - dt.timedelta(days=num_days_in_val_set)
        self.train_start = list_dates[0]
        # To avoid data overlap in train/val sets, remove the last 4 days from train set
        self.train_end = self.val_start - dt.timedelta(days=4)
        print(f"Train dataset: from {self.train_start} to {self.train_end}")
        print(f"Val dataset: from {self.val_start} to {self.val_end}")

    def setup(self, stage: Literal["fit", "val", "validate"]) -> None:  # type: ignore
        """Called by lightning, at the start of a stage.

        Args:
            stage: either 'fit', 'val', 'validate' or 'test'.
        """
        if stage == "fit":
            self.train_dataset = (
                CamsDataset(self.train_start, self.train_end, self.data_dir)
                if self.train_dataset is None
                else self.train_dataset
            )
        if stage in ["fit", "val", "validate"]:
            self.val_dataset = (
                CamsDataset(self.val_start, self.val_end, self.data_dir)
                if self.val_dataset is None
                else self.val_dataset
            )
        else:
            raise ValueError(
                "BMRDatamodule.setup():\n\tparameter stage should be either 'fit', "
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
        """Collate a batch of NamedTensor data."""

        inputs = NamedTensor.collate_fn([item[0] for item in batch])
        targets = NamedTensor.collate_fn([item[1] for item in batch])

        return inputs, targets


if __name__ == "__main__":
    dm = CamsDataModule()
    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))
    print(x, y)
