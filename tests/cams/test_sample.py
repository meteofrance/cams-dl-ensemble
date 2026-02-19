import datetime as dt
from pathlib import Path

import pytest
from mfai.pytorch.namedtensor import NamedTensor

from cams.sample import Sample
from cams.settings import MODEL_NAMES

PROCESSED_DIR = Path("tests/data/")

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


@pytest.mark.parametrize(
    "run_date, lead_time",
    [
        (dt.datetime(2023, 1, 1), 15),
        (dt.datetime(2023, 1, 2), 15),
    ],
)
def test_sample_is_valid_true(run_date: dt.datetime, lead_time: int):
    sample = Sample(run_date, lead_time, PROCESSED_DIR)
    assert sample.is_valid


def test_sample_input_data():
    sample = Sample(dt.datetime(2023, 1, 1), 15, PROCESSED_DIR)
    data = sample.input_data

    assert isinstance(data, NamedTensor)
    assert data.tensor.shape == (11, 420, 700)
    assert set(data.feature_names) == set(MODEL_NAMES)


def test_sample_target_data():
    sample = Sample(dt.datetime(2023, 1, 1), 15, PROCESSED_DIR)
    data = sample.target_data

    assert isinstance(data, NamedTensor)
    assert data.tensor.shape == (1, 420, 700)
    assert list(data.feature_names) == ["Analysis"]
