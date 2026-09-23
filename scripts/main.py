"""Script used to interact directly with the lightning cli."""

from codecarbon import OfflineEmissionsTracker
from mfai.pytorch.callbacks import MLFlowSaveConfigCallback

from cams.cli import CAMSLightningCLI
from cams.datamodule import CAMSDataModule
from cams.plmodule import CAMSLightningModule

if __name__ == "__main__":
    with OfflineEmissionsTracker() as tracker:
        CAMSLightningCLI(
            model_class=CAMSLightningModule,
            datamodule_class=CAMSDataModule,
            save_config_kwargs={"overwrite": True},
            save_config_callback=MLFlowSaveConfigCallback,
        )
