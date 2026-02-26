from pathlib import Path
from typing import Any

import torch
import torchmetrics as tm
from lightning import LightningModule
from mfai.pytorch.models.base import BaseModel
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor
from torch.optim import AdamW
from typing_extensions import override


class CAMSLightningModule(LightningModule):
    """CAMS lightning module.
    Responsibilities:
        - Training loop
        - Model test
        - Model eval
        - Logging
    """

    def __init__(
        self,
        model: BaseModel,
        loss: torch.nn.Module,
        learning_rate: float = 0.0001,
    ) -> None:
        """CAMS lightning module

        Args:
            model: A model inheriting from mfai.BaseModel
            loss: The loss function.
            learning_rate: The optimizer's learning rate. Defaults to 0.0001.
        """
        super().__init__()
        self.model = model
        self.model = torch.compile(self.model)
        self.loss = loss
        self.learning_rate = learning_rate

        self.metrics = self.get_metrics()
        self.save_hyperparameters()

    ####################################################################################
    #                                      SETUP                                       #
    ####################################################################################

    def get_metrics(self) -> tm.MetricCollection:
        """Defines the metrics that will be computed during train and valid steps."""
        metrics_dict = {
            "MSE": tm.MeanSquaredError(squared=False),
            "MAE": tm.MeanAbsoluteError(),
            "MeanAbsPercError": tm.MeanAbsolutePercentageError(),
        }
        return tm.MetricCollection(metrics_dict)

    @override
    def configure_optimizers(self) -> AdamW:
        """Lightning method to define optimizers and learning-rate schedulers"""
        return AdamW(self.parameters(), lr=self.learning_rate)

    ####################################################################################
    #                                      SHARED STEPS                                #
    ####################################################################################

    def last_activation(self, y_hat: Tensor) -> Tensor:
        """Applies appropriate activation according to task."""
        return torch.nn.functional.relu(y_hat)  # Appropriate for O3 but not for others?

    @override
    def forward(self, inputs: NamedTensor) -> NamedTensor:
        """Runs data through the model. Separate from training step."""
        output = self.last_activation(self.model(inputs.tensor))
        y_hat = NamedTensor(output, names=inputs.names, feature_names=["AI_Forecast"])
        return y_hat

    def _shared_forward_step(
        self, x: NamedTensor, y: NamedTensor
    ) -> tuple[NamedTensor, Any]:
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps
        """
        output = self.last_activation(self.model(x.tensor))
        loss = self.loss(output, y.tensor)
        y_hat = NamedTensor(output, names=y.names, feature_names=["AI_Forecast"])
        return y_hat, loss

    ####################################################################################
    #                                      TRAIN STEPS                                 #
    ####################################################################################
    @override
    def on_train_start(self) -> None:
        """Print log directory at training start"""
        if self.logger and self.logger.log_dir:
            print(f"Logs will be saved in \033[96m{self.logger.log_dir}\033[0m")  # cyan

    @override
    def training_step(
        self, batch: tuple[NamedTensor, NamedTensor], batch_idx: int
    ) -> Any:
        """Defines the training step"""
        x, y = batch
        _, loss = self._shared_forward_step(x, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    ####################################################################################
    #                                 VALIDATION STEPS                                 #
    ####################################################################################

    def val_plot_step(
        self,
        batch_idx: int,
        x: NamedTensor,
        y: NamedTensor,
        y_hat: NamedTensor,
        mode: str,
    ) -> None:
        """Plots images on some batches and log them in tensorboard."""
        if self.logger is None:
            return
        interesting_batches = [0, 6, 12, 42, 66]
        if batch_idx not in interesting_batches:
            return
        pass # TODO

    @override
    def validation_step(
        self, batch: tuple[NamedTensor, NamedTensor], batch_idx: int
    ) -> Any:
        """Defines the validation step"""
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.metrics.update(y_hat.tensor, y.tensor)
        self.val_plot_step(batch_idx, x, y, y_hat, mode="val")
        return loss

    @override
    def on_validation_epoch_end(self) -> None:
        """Computes and logs metrics at validation end"""
        if self.logger is None:
            return
        self.log_dict(
            self.metrics.compute(), logger=True, sync_dist=True, on_epoch=True
        )
        self.metrics.reset()


def load_model(last_ckpt: Path) -> CAMSLightningModule:
    """Loads a trained model, ready for inference."""
    model = CAMSLightningModule.load_from_checkpoint(last_ckpt)
    model.eval()  # disable randomness, dropout, etc...
    return model
