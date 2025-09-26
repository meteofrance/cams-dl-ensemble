from lightning.pytorch.core import LightningDataModule
from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import DataLoader
from typing_extensions import override


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

    @override
    def setup(self, stage: str) -> None:
        """Called by lighthning before requesting a dataloader.
        Instantiate the dataset that correspond to the given stage.
        """
        raise NotImplementedError()

    @override
    def train_dataloader(self) -> DataLoader[tuple[NamedTensor, NamedTensor]]:
        """Returns the train dataloader."""
        raise NotImplementedError()

    @override
    def val_dataloader(self) -> DataLoader[tuple[NamedTensor, NamedTensor]]:
        """Returns the validation dataloader."""
        raise NotImplementedError()

    @override
    def test_dataloader(self) -> DataLoader[tuple[NamedTensor, NamedTensor]]:
        """Returns the test dataloader."""
        raise NotImplementedError()
