from logging import warning
from typing import Any

import torch
from mfai.pytorch.lightning_modules import SegmentationLightningModule
from mfai.pytorch.models.base import BaseModel
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor


class CamsLightningModule(SegmentationLightningModule):
    def __init__(
        self,
        model: BaseModel,
        loss: torch.nn.modules.loss._Loss,
    ) -> None:
        super().__init__(model=model, type_segmentation="regression", loss=loss)
        self.save_hyperparameters()

    def get_hparams(self) -> dict:
        """Return the hparams we want to save in logger"""
        raise NotImplementedError
