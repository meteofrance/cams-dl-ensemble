from pathlib import Path
import datetime as dt
from cams.dataset import CAMSDataset
from scripts.data.compute_stats import compute_stats
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_compute_stats(tmp_dataset_dir: Path):
    # Create dummy files
    input_path1 = tmp_dataset_dir / "input/2022_01_01.netcdf"
    input_path2 = tmp_dataset_dir / "input/2022_01_02.netcdf"
    target_path1 = tmp_dataset_dir / "target/2022_01_01_15.netcdf"
    target_path2 = tmp_dataset_dir / "target/2022_01_02_15.netcdf"

    create_dummy_input_netcdf(input_path1)
    create_dummy_input_netcdf(input_path2)
    create_dummy_target_netcdf(target_path1)
    create_dummy_target_netcdf(target_path2)

    dates = [dt.datetime(2022, 1, i) for i in range(1, 3)]

    dataset = CAMSDataset(run_dates=dates, models=[""], processed_dir=tmp_dataset_dir)

    assert len(dataset.samples) == 2

    stats = compute_stats(dataset)

    assert stats == {"O3": {"min": 0, "max": 0}}
