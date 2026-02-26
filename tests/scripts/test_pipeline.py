from pathlib import Path
import datetime as dt

from cams.datamodule import CAMSDataModule

from scripts.fit_and_val import fit_and_val
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_full_pipeline(tmp_dataset_dir: Path) -> None:
    """Test the full project life cycle.
    Test the cli interface entry points.

    - Training.
    - Retrain a checkpoint. X TODO
    - Checkpoint writting. X TODO
    - Predict from checkpoint. X TODO
    - Export to onnx. X TODO
    - Predict from onnx. X TODO
    """

    # Create fake dataset
    dates = [dt.datetime(2000, 1, i) for i in range(1, 32)]
    for date in dates:
        input_path = tmp_dataset_dir / f"input/{date.strftime('%Y_%m_%d')}.netcdf"
        target_path = tmp_dataset_dir / f"target/{date.strftime('%Y_%m_%d_15')}.netcdf"
        create_dummy_input_netcdf(input_path)
        create_dummy_target_netcdf(target_path)

    # Train with a test config.
    fit_and_val(
        datamodule_cls=CAMSDataModule,
        args=["--config", "tests/test_config.yaml", "--data.processed_dir", str(tmp_dataset_dir)],
    )

