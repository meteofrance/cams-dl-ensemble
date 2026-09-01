from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from cams.datamodule import CAMSDataModule
from cams.plmodule import CAMSLightningModule
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def fit_model(args: list[str] | None = None) -> None | Path:
    """Fits a model, with the same arguments as in command line, and returns ckpt path.

    Args:
        args: arguments givent to the LightningCLI object.
            Allows configuration arguments such as:
                ['--config', 'config/file/path.yaml']
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

    # Forward
    inputs, _ = next(iter(cli.datamodule.train_dataloader()))
    cli.model(inputs)

    if cli.trainer.checkpoint_callback:
        return Path(cli.trainer.checkpoint_callback.dirpath)  # type: ignore[reportAttributeAcessIssue]


def test_full_pipeline(tmp_dataset_dir: Path) -> None:
    """Test the full project life cycle.
    Test the cli interface entry points.

    - Training.
    - Checkpoint writting.
    - Retrain from a checkpoint. X TODO
    - Predict from checkpoint. X TODO
    - Export to onnx. X TODO
    - Predict from onnx. X TODO
    """

    # Create fake dataset
    img_size = (64, 64)  # Small images to lighten the pipeline
    for day in range(1, 32):
        input_path = (
            tmp_dataset_dir
            / f"mocage/2022_07_{day:02}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        )
        create_dummy_input_netcdf(input_path, *img_size)
    target_path = tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"
    create_dummy_target_netcdf(target_path, *img_size)

    # Train with a test config
    ckpt_folder = fit_model(
        args=[
            "--config",
            "tests/test_config.yaml",
            "--data.processed_dir",
            str(tmp_dataset_dir),
        ],
    )

    # Check checkpoint writing
    assert isinstance(ckpt_folder, Path)
    assert ckpt_folder.exists()
    ckpt_paths = list(ckpt_folder.glob("*.ckpt"))
    assert len(ckpt_paths) > 0
