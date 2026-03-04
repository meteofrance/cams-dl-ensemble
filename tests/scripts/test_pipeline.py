import datetime as dt
from pathlib import Path

from scripts.fit_and_val import fit_and_val
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


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
    dates = [dt.datetime(2000, 1, i) for i in range(1, 32)]
    for date in dates:
        input_path = tmp_dataset_dir / f"input/{date.strftime('%Y_%m_%d')}.netcdf"
        target_path = tmp_dataset_dir / f"target/{date.strftime('%Y_%m_%d_15')}.netcdf"
        create_dummy_input_netcdf(input_path, *img_size)
        create_dummy_target_netcdf(target_path, *img_size)

    # Train with a test config
    ckpt_folder = fit_and_val(
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
