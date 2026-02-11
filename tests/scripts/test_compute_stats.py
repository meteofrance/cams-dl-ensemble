from pathlib import Path

from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf
from scripts.compute_stats import get_list_samples, compute_stats

def test_compute_stats(setup_cams_directories: Path):
    # Create dummy files
    input_path1 = setup_cams_directories / "input/2022_01_01.netcdf"
    input_path2 = setup_cams_directories / "input/2022_01_02.netcdf"
    target_path1 = setup_cams_directories / "target/2022_01_01_15.netcdf"
    target_path2 = setup_cams_directories / "target/2022_01_02_15.netcdf"

    create_dummy_input_netcdf(input_path1)
    create_dummy_input_netcdf(input_path2)
    create_dummy_target_netcdf(target_path1)
    create_dummy_target_netcdf(target_path2)

    samples = get_list_samples(setup_cams_directories)

    assert len(samples) == 2

    stats = compute_stats(samples)

    assert stats == {"O3": {"min": 0, "max":0}}
