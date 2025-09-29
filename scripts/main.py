"""Script used to interact directly with the lightning cli.
It is recommended to use the specialized scripts to interact with your models,
for better checkpoint management.
- Use `scripts/fit_and_val.py` to fit and validate from scratch or a checkpoint.
- Use `scripts/predict.py` to run inference from a checkpoint.
"""

from lightning.pytorch.core import LightningDataModule

from cams.datamodule import CamsDataModule


def cli_main(
    datamodule: type[LightningDataModule] = CamsDataModule,
    args: list[str] = None,
) -> None:
    """Entry point into the cams lightnig cli."""
    raise NotImplementedError()


if __name__ == "__main__":
    cli_main()
