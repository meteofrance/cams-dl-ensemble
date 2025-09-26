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
