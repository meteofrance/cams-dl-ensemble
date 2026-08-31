from pathlib import Path
import datetime as dt
from cams.dataset import CAMSDataset
from scripts.data.compute_stats import compute_stats
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


def test_compute_stats(tmp_dataset_dir: Path):
    # Create dummy files
    input_path1 = tmp_dataset_dir / "mocage/2022_07_01-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    input_path2 = tmp_dataset_dir / "mocage/2022_07_02-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
    target_path1 = tmp_dataset_dir / "reanalysis/cams.eaq.ira.ENSa.o3.l0.2022-07.nc"

    create_dummy_input_netcdf(input_path1)
    create_dummy_input_netcdf(input_path2)
    create_dummy_target_netcdf(target_path1)

    dates = [dt.datetime(2022, 7, i) for i in range(1, 3)]

    dataset = CAMSDataset(dates, models=["mocage"], lead_times=[15], species=["O3"], levels=[0], processed_dir=tmp_dataset_dir)

    assert len(dataset.samples) == 2

    stats = compute_stats(dataset, species=["O3"])

    assert stats == {"O3": {"min": 0, "max": 0}}
