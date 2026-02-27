import datetime as dt
from pathlib import Path

import pytest
from mfai.pytorch.namedtensor import NamedTensor

from cams.sample import Sample
from cams.settings import MODEL_NAMES
from tests.conftest import create_dummy_input_netcdf, create_dummy_target_netcdf


@pytest.mark.parametrize(
    "run_date, lead_time",
    [
        (dt.datetime(2022, 7, 22), 15),
        (dt.datetime(2023, 1, 1), 3),
        (dt.datetime(2023, 12, 31), 96),
    ],
)
def test_sample_creation(run_date: dt.datetime, lead_time: int):
    sample = Sample(run_date, lead_time)
    assert sample.date_run == run_date
    assert sample.lead_time == lead_time
    expected_valid_time = run_date + dt.timedelta(hours=lead_time)
    assert sample.valid_time == expected_valid_time


def test_sample_str():
    sample = Sample(dt.datetime(2022, 7, 22, 12), 15)
    result = str(sample)
    assert "2022-07-22 12:00" in result
    assert "+15h" in result


def test_sample_paths():
    sample = Sample(dt.datetime(2022, 7, 22), 15)
    assert sample.input_path.name == "2022_07_22.netcdf"
    assert sample.target_path.name == "2022_07_22_15.netcdf"


def test_sample_is_valid_false():
    sample = Sample(dt.datetime(2022, 7, 22, 12), 15)
    assert not sample.is_valid


def test_sample_is_valid_true(tmp_dataset_dir: Path):
    # Create dummy files in directories
    input_path = tmp_dataset_dir / "input/2022_07_22.netcdf"
    target_path = tmp_dataset_dir / "target/2022_07_22_15.netcdf"

    create_dummy_input_netcdf(input_path)
    create_dummy_target_netcdf(target_path)

    sample = Sample(dt.datetime(2022, 7, 22), 15, tmp_dataset_dir)
    assert sample.is_valid


def test_sample_input_data(tmp_dataset_dir: Path):
    input_path = tmp_dataset_dir / "input/2022_07_22.netcdf"
    create_dummy_input_netcdf(input_path)

    sample = Sample(dt.datetime(2022, 7, 22), 15, tmp_dataset_dir)
    data = sample.input_data

    assert isinstance(data, NamedTensor)
    assert data.tensor.shape == (11, 420, 700)
    assert data.feature_names == MODEL_NAMES


def test_sample_target_data(tmp_dataset_dir: Path):
    target_path = tmp_dataset_dir / "target/2022_07_22_15.netcdf"
    create_dummy_target_netcdf(target_path)

    sample = Sample(dt.datetime(2022, 7, 22), 15, tmp_dataset_dir)
    data = sample.target_data

    assert isinstance(data, NamedTensor)
    assert data.tensor.shape == (1, 420, 700)
    assert list(data.feature_names) == ["Analysis"]
