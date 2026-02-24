"""Script used to interact directly with the lightning cli.
It is recommended to use the specialized scripts to interact with your models,
for better checkpoint management.
- Use `scripts/fit_and_val.py` to fit and validate from scratch or a checkpoint.
- Use `scripts/predict.py` to run inference from a checkpoint.
"""

from lightning.pytorch.cli import LightningCLI

from cams.datamodule import CAMSDataModule
from cams.plmodule import CAMSLightningModule


def cli_main() -> None:
    """Entry point into the cams lightnig cli."""
    LightningCLI(CAMSLightningModule, CAMSDataModule)


if __name__ == "__main__":
    cli_main()
