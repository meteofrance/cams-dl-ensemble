"""Script used to interact directly with the lightning cli.
It is recommended to use the specialized scripts to interact with your models,
for better checkpoint management.
- Use `bin/fit_and_val.py` to fit and validate from scratch or a checkpoint.
- Use `bin/predict.py` to run inference from a checkpoint.
"""

from lightning.pytorch.core import LightningDataModule

from src.cli import CamsCli
from src.datamodule import CamsDataModule
from src.plmodule import CamsLightningModule


def cli_main(
    datamodule: LightningDataModule = CamsDataModule,
    args: list[str] = None,
) -> None:
    cli = CamsCli(
        model_class=CamsLightningModule,
        datamodule_class=datamodule,
        args=args,
    )


if __name__ == "__main__":
    cli_main()
