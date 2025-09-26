from typing import Any

from mfai.pytorch.lightning_modules import SegmentationLightningModule
from typing_extensions import override


class CamsLightningModule(SegmentationLightningModule):
    """Cams lightning module.
    Responsibilities:
        - Training loop
        - Model test
        - Model eval
        - Logging
    """

    def __init__(
        self,
    ) -> None:
        """
        Args:
            model: the model to train.
            loss: the loss function used to train the model.
        """
        raise NotImplementedError()

    @override
    def get_hparams(self) -> dict[str, Any]:
        """Returns the hparams we want to save in logger"""
        raise NotImplementedError
