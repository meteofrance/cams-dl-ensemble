from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader


class CamsDataModule(LightningDataModule):
    """
    A Lightning DataModule wrapping the dataset.
    It defines the train/valid/test datasets and their dataloaders.
    """

    def __init__(self) -> None:
        """
        Args:
            not implemented yet
        """
        raise NotImplementedError()

    def setup(self, stage: str) -> None:
        """Called by lighthning before requesting a dataloader.
        Instantiate the dataset that correspond to the given stage.
        """
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )
