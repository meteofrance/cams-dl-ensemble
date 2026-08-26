from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import torch
from lightning import LightningModule
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mfai.pytorch.models.base import BaseModel, ModelABC
from mfai.pytorch.namedtensor import NamedTensor
from mlflow import MlflowClient
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW
from torchmetrics import MetricCollection
from typing_extensions import override

from cams.metrics import (
    Accuracy,
    F1Score,
    FalseAlarmRate,
    FalsePositiveRate,
    MeanAbsoluteError,
    MeanSquaredError,
)
from cams.plots import plot_y_vs_yhat_vs_median
from cams.transforms import ExtractInputStatisticalFeatures


class CAMSLightningModule(LightningModule):
    """CAMS lightning module.
    Responsibilities:
        - Training loop
        - Model test
        - Model evaluation
        - Logging
    """

    def __init__(
        self,
        model: BaseModel | ModelABC,
        loss: torch.nn.Module,
        learning_rate: float = 0.0001,
        training_mode: Literal["residual", "classic"] = "classic",
    ) -> None:
        """CAMS lightning module

        Args:
            model: A model inheriting from mfai.BaseModel
            loss: The loss function.
            learning_rate: The optimizer's learning rate. Defaults to 0.0001.
            training_mode: Training mode, classic (y = f(x)) or residual (y = f(x) + x).
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.training_mode = training_mode
        self.metrics = self.get_metrics()
        self.save_hyperparameters()

    ####################################################################################
    #                                      SETUP                                       #
    ####################################################################################

    @override
    def setup(self, stage: str):
        """Setup lightning module and check that arguments are compatible."""
        if self.training_mode == "residual":
            check_median = False
            for transform in self.trainer.datamodule.transform_sequence:  # type: ignore[reportAttributeAccessIssue]
                if isinstance(transform, ExtractInputStatisticalFeatures):
                    if "median" in transform.statistic_types:
                        check_median = True
                        break
            if not check_median:
                raise Exception(
                    "Model is in residual training mode but datamodule does not "
                    "contain an ExtractInputStatisticalFeatures transform with median "
                    "as statistic_type. Please add it in your data configuration."
                )

    def get_metrics(self) -> MetricCollection:
        """Defines the metrics that will be computed during train and valid steps."""
        metrics = MetricCollection(
            [
                MetricCollection(
                    [MeanSquaredError(squared=False), MeanAbsoluteError()]
                ),
                MetricCollection(
                    [
                        Accuracy("target - O3 - +15h - 0m", threshold=120),
                        F1Score("target - O3 - +15h - 0m", threshold=120),
                        FalseAlarmRate("target - O3 - +15h - 0m", threshold=120),
                        FalsePositiveRate("target - O3 - +15h - 0m", threshold=120),
                    ],
                    prefix="O3-15h-0m/",
                    postfix="_120",
                ),
            ]
        )
        return metrics

    @override
    def configure_optimizers(self) -> AdamW:
        """Lightning method to define optimizers and learning-rate schedulers"""
        return AdamW(self.parameters(), lr=self.learning_rate)

    ####################################################################################
    #                                      SHARED STEPS                                #
    ####################################################################################

    @override
    def forward(self, inputs: NamedTensor) -> NamedTensor:
        """Runs data through the model. Separate from training step."""
        output = self.model(inputs.tensor)  # pyright: ignore[reportCallIssue]
        if self.training_mode == "residual":
            y_hat_tensor = inputs["median"] + output
        else:
            y_hat_tensor = output
        y_hat = NamedTensor(
            y_hat_tensor, names=inputs.names, feature_names=inputs.feature_names
        )
        _, y_hat = self.trainer.datamodule.undo_transforms(inputs, y_hat)  # type: ignore[reportAttributeAccessIssue]
        return y_hat

    def _shared_forward_step(
        self, x: NamedTensor, y: NamedTensor
    ) -> tuple[NamedTensor, Any]:
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps.
        """
        output = self.model(x.tensor)  # pyright: ignore[reportCallIssue]
        if self.training_mode == "residual":
            y_hat_tensor = x["median"] + output
        else:
            y_hat_tensor = output
        loss = self.loss(y_hat_tensor, y.tensor)
        y_hat = NamedTensor(y_hat_tensor, names=y.names, feature_names=y.feature_names)
        return y_hat, loss

    ####################################################################################
    #                                      TRAIN STEPS                                 #
    ####################################################################################
    @rank_zero_only
    @override
    def on_train_start(self) -> None:
        """Print log directory at training start"""
        print("\033[96m-----TRAINING START------\033[0m")
        if isinstance(self.logger, MLFlowLogger):
            print("\033[96mTracking run with MLFlow:\033[0m")
            print(
                f"-> Experiment: {self.logger._experiment_name} - "  # type: ignore[reportPrivateUsage]
                f"Id: {self.logger.experiment_id}"
            )
            print(f"-> Run: {self.logger._run_name} - Id: {self.logger.run_id}")  # type: ignore[reportPrivateUsage]
        if self.trainer.checkpoint_callback:
            print(f"-> Checkpoint path: {self.trainer.checkpoint_callback.dirpath}")  # type: ignore[reportAttributeAcessIssue]
        print("\033[96m-------------------------\033[0m")

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
    ) -> None:
        """Plots images on some batches and log them in mlflow."""

        # Guard conditions
        if (
            # Skip  if no mlflow logger
            not isinstance(self.logger, MLFlowLogger)
            # Only plot every 15 epochs and the 2 last epochs
            or (
                self.trainer.max_epochs is not None
                and self.trainer.current_epoch % 15 != 0
                and self.trainer.current_epoch != self.trainer.max_epochs
                and self.trainer.current_epoch != self.trainer.max_epochs - 1
                # Only plot the first batch of the evaluation
                or batch_idx not in [0]
            )
            # No run id
            or self.logger.run_id is None
        ):
            return

        # Open temporary file
        with NamedTemporaryFile(
            prefix=f"epoch_{self.trainer.current_epoch}_", suffix=".png"
        ) as file:
            # First save the plot in a temporary PNG file
            plot_y_vs_yhat_vs_median(
                x.select_dim("batch", 0),
                y.select_dim("batch", 0),
                y_hat.select_dim("batch", 0),
                Path(file.name),
                f"Epoch {self.trainer.current_epoch}",
            )

            # Then open the image with PIL and log it in mlflow
            with Image.open(file.name) as img:
                mlf_logger: MlflowClient = self.logger.experiment
                mlf_logger.log_image(
                    self.logger.run_id,
                    image=img,
                    key="val_plot",
                    step=self.current_epoch,
                )

    @override
    def validation_step(
        self, batch: tuple[NamedTensor, NamedTensor], batch_idx: int
    ) -> Any:
        """Defines the validation step."""
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        _, y_hat = self.trainer.datamodule.undo_transforms(x, y_hat)  # type: ignore[reportAttributeAccessIssue]
        x, y = self.trainer.datamodule.undo_transforms(x, y)  # type: ignore[reportAttributeAccessIssue]
        self.metrics.update(y_hat, y)
        self.val_plot_step(batch_idx, x, y, y_hat)
        return loss

    @override
    def on_validation_epoch_end(self) -> None:
        """Computes and logs metrics at validation end."""
        if self.logger is None:
            return
        self.log_dict(
            self.metrics.compute(), logger=True, sync_dist=True, on_epoch=True
        )
        self.metrics.reset()
