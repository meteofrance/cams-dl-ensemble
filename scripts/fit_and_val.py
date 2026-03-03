"""Script that fits a model and runs the validation stage on the best checkpoint.
Use cases:
1. Train from scratch:
`fit_and_val.py --config config/path.yaml`
2. Retrain from checkpoint:
```
fit_and_val.py
    --ckpt_path experiment/folder/checkpoint.ckpt
    --config experiment/folder/config.yaml
```

=> error if mismatch between ckpt and current init args
    in LightningModule or DataModule.

"""

import argparse
import sys
from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from cams.datamodule import CAMSDataModule
from cams.plmodule import CAMSLightningModule


def fit_and_val(
    args: list[str] | None = None,
    ckpt_path: Path | None = None,
) -> None | Path:
    """Fits a model and runs the validation stage on the best checkpoint.

    Args:
        args: arguments givent to the LightningCLI object.
            Allows configuration arguments such as:
                ['--config', 'config/file/path.yaml']
        ckpt_path: loads a model from a checkpoint before fitting and validation.
    """
    # Create cli object with `run=False` to parse and instantiate
    # LightningModule and DataModule, but not run subcommands
    cli = LightningCLI(
        model_class=CAMSLightningModule,
        datamodule_class=CAMSDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
        run=False,
    )

    # Train
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Validate on the best checkpoint
    cli.trainer.validate(cli.model, cli.datamodule.val_dataloader(), ckpt_path="best")

    if cli.trainer.checkpoint_callback:
        return Path(cli.trainer.checkpoint_callback.dirpath)


if __name__ == "__main__":
    # Parse ckpt_path argument and remove it from sys.argv as
    #   it is not expected by LightningCli
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, dest="ckpt_path")
    parsed_arguments, unparsed_arguments = (
        parser.parse_known_args()
    )  # Only parse ckpt_path
    sys.argv = (
        sys.argv[:1] + unparsed_arguments
    )  # Remove parsed arguments from sys.argv

    # Update args with checkpoint's config file path if necessary
    config_args = None
    if parsed_arguments.ckpt_path is not None:
        if not ("--config" in sys.argv or "-c" in sys.argv):
            config_path = Path(parsed_arguments.ckpt_path).parent.parent / "config.yaml"
            config_args = ["--config", str(config_path)]

    fit_and_val(
        args=config_args,
        ckpt_path=parsed_arguments.ckpt_path,
    )
