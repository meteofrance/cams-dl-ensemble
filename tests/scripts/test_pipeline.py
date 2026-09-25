import os
from pathlib import Path
from typing import Iterator

import pytest

# MLflow 3.x blocks the filesystem tracking backend (used by MLflowLogger with
# save_dir) unless explicitly allowed. Keep the test config behaviour stable.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# A remote MLFLOW_TRACKING_URI in the environment shadows the local save_dir of
# tests/test_config.yaml, making MLFlowLogger.save_dir return None and tripping
# Lightning's SaveConfigCallback. Unset it for the test so runs go to the local
# file store, and restore any previous value afterwards.
_PREV_MLFLOW_TRACKING_URI = os.environ.pop("MLFLOW_TRACKING_URI", None)

# These imports must run after the env var is unset: the Lightning CLI and
# MLflow logger read MLFLOW_TRACKING_URI at instantiation/import time.
from cams.cli import CAMSLightningCLI  # noqa: E402
from cams.datamodule import CAMSDataModule  # noqa: E402
from cams.plmodule import CAMSLightningModule  # noqa: E402
from tests.conftest import (  # noqa: E402
    create_dummy_input_netcdf,
    create_dummy_target_netcdf,
)


@pytest.fixture(autouse=True)
def _restore_mlflow_tracking_uri() -> Iterator[None]:
    """Restore the previously unset MLFLOW_TRACKING_URI after the test."""
    yield
    if _PREV_MLFLOW_TRACKING_URI is not None:
        os.environ["MLFLOW_TRACKING_URI"] = _PREV_MLFLOW_TRACKING_URI


def fit_model(args: list[str] | None = None) -> None | Path:
    """Fits a model, with the same arguments as in command line, and returns ckpt path.

    Args:
        args: arguments givent to the LightningCLI object.
            Allows configuration arguments such as:
                ['--config', 'config/file/path.yaml']
    """
    if args is None:
        args = []
    # Pin the tracking_uri to the local file store of tests/test_config.yaml:
    # MLFlowLogger caches the env-based default at import time, so it can stay
    # remote in the full suite even after MLFLOW_TRACKING_URI is unset above.
    override_args = args + [
        "--trainer.logger.init_args.tracking_uri",
        "file:/tmp/cams_tests/",
    ]
    # Create cli object with `run=False` to parse and instantiate
    # LightningModule and DataModule, but not run subcommands
    cli = CAMSLightningCLI(
        model_class=CAMSLightningModule,
        datamodule_class=CAMSDataModule,
        save_config_kwargs={"overwrite": True},
        args=override_args,
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


def test_pipeline_invalid_val_leadtimes(tmp_dataset_dir: Path) -> None:
    """Raises ValueError when a validation leadtime is not in the selected ones.

    The CLI links the data selection arguments onto the module, so a validation
    leadtime absent from ``data.lead_times`` must fail at module instantiation.
    """
    with pytest.raises(ValueError):
        fit_model(
            args=[
                "--config",
                "tests/test_config.yaml",
                "--data.processed_dir",
                str(tmp_dataset_dir),
                "--model.val_leadtimes=[15,24]",
            ],
        )
