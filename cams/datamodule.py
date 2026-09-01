import calendar
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
from cams.types import Leadtimes, Levels, SpeciesNames


class CAMSDataModule(LightningDataModule):
    """
    A Lightning DataModule wrapping the dataset.
    It defines the train/valid/test datasets and their dataloaders.
    """

    train_dataset: CAMSDataset | None = None  # Set at setup
    val_dataset: CAMSDataset | None = None

    def __init__(
        self,
        # We don't use ModelsNames type because of jsonargparse error:
        # "Parser key 'data.models': Cannot take a Union of no types".
        models: list[str],
        lead_times: list[Leadtimes],
        species: list[SpeciesNames],
        levels: list[Levels],
        batch_size: int = 2,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        start_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
        val_days: int = 5,
        train_val_separation: int = 4,
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
            end_date: Dataset end date, inclusive. If None, latest date is
                selected. Defaults to None.
            val_days: Number of days reserved for validation at the
                end of each month, inclusive. Defaults to 5.
            train_val_separation: Number of days between train and validation
                datasets. Defaults to 4.
            models: Models to load in the dataset.
            lead_times: Leadtimes to load in the dataset.
            species: Species to load in the dataset.
            levels: Levels to load in the dataset.
                '0' corresponds to 'ground' level.
            processed_dir: Path to the CAMS processed dataset.
            transforms: list of transforms to apply to the data after loading it.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.models = models
        self.lead_times = lead_times
        self.species = species
        self.levels = levels
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
            raise FileNotFoundError("CAMS dataset empty: no run found.")

        # Set dates if they are not defined
        start_date = start_date if start_date else run_dates[0]
        end_date = end_date if end_date else run_dates[-1]
        self.train_dates = []
        self.val_dates = []
        for date in run_dates:
            _, last_day_month = calendar.monthrange(date.year, date.month)
            last_date_month = dt.datetime(date.year, date.month, last_day_month)
            first_date_month = dt.datetime(date.year, date.month, 1)

            val_start_date = last_date_month - dt.timedelta(days=val_days)
            train_end_date = last_date_month - dt.timedelta(
                days=val_days + train_val_separation
            )
            if first_date_month <= date <= train_end_date:
                self.train_dates.append(date)
            elif val_start_date < date <= last_date_month:
                self.val_dates.append(date)

        if (
            any(date in self.val_dates for date in self.train_dates)
            or any(date in self.train_dates for date in self.val_dates)
            or len(self.train_dates) == 0
            or len(self.val_dates) == 0
        ):
            raise ValueError(
                "Start and end dates given for CAMS "
                f"dataset {self.processed_dir.parent} are invalid and "
                "produce an empty dataset."
            )

        # Display dataset availability informations
        print(f"--> {len(run_dates)} runs available in whole dataset.")
        print(f"from {run_dates[0]} to {run_dates[-1]}")
        print(f"--> {len(run_dates)} runs available in selected dataset.")
        print(f"--> {len(self.train_dates)} train dates in selected dataset.")
        print(f"--> {len(self.val_dates)} val dates in selected dataset.")
        print(f"from {self.val_dates[0]} to {self.val_dates[-1]}")

        self.save_hyperparameters()

    @override
    def setup(self, stage: Literal["fit", "val", "validate"] | str) -> None:
        """Called by lightning, at the start of a stage.

        Args:
            stage: Selects which dataset is loaded,
                either 'fit', 'val', 'validate' or 'test'.
        """
        dataset_kwargs = {
            "models": self.models,
            "lead_times": self.lead_times,
            "species": self.species,
            "levels": self.levels,
            "processed_dir": self.processed_dir,
            "transform_sequence": self.transform_sequence,
        }
        if stage == "fit":
            self.train_dataset = (
                CAMSDataset(self.train_dates, **dataset_kwargs)
                if self.train_dataset is None
                else self.train_dataset
            )
            print("--> Train dataset length: ", len(self.train_dataset))
        if stage in ["fit", "val", "validate"]:
            self.val_dataset = (
                CAMSDataset(self.val_dates, **dataset_kwargs)
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
    dm = CAMSDataModule(
        models=["CHIMERE", "MOCAGE"],
        lead_times=[15, 24],
        species=["O3", "NO2"],
        levels=[0],
    )
    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))
    print(x, y)
