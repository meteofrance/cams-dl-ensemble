"""Script used to interact directly with the lightning cli."""

from lightning.pytorch.cli import LightningCLI

from cams.datamodule import CAMSDataModule
from cams.plmodule import CAMSLightningModule

if __name__ == "__main__":
    LightningCLI(
        model_class=CAMSLightningModule,
        datamodule_class=CAMSDataModule,
        save_config_kwargs={"overwrite": True},
        save_config_callback = None,
    )
