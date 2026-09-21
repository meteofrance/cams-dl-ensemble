import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor

from cams.transforms import (
    ExtractInputStatisticalFeatures,
    FillMissingModels,
    Normalize,
)
from cams.types import MODELS_NAMES


def make_input_ds(data_vars: dict[str, np.ndarray]) -> xr.Dataset:
    """Builds an input xarray dataset with one data_var per model."""
    return xr.Dataset(
        {name: (["latitude", "longitude"], arr) for name, arr in data_vars.items()}
    )


def test_ExtractInputStatisticalFeatures():
    """Test of ExtractInputStatisticalFeatures tranform."""
    os.environ["SCIPY_ARRAY_API"] = "0"
    with pytest.raises(RuntimeError):
        transform = ExtractInputStatisticalFeatures(
            ["mean", "amin", "argmin", "amax", "argmax", "median", "skew", "kurtosis"]
        )
    os.environ["SCIPY_ARRAY_API"] = "1"

    input_ds = make_input_ds(
        {
            "model_A": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "model_B": np.array([[5.0, 6.0], [7.0, 8.0]]),
            "model_C": np.array([[9.0, 10.0], [11.0, 12.0]]),
        }
    )
    target_ds = xr.Dataset({"analysis": (["latitude", "longitude"], np.ones((2, 2)))})

    transform = ExtractInputStatisticalFeatures(
        ["mean", "amin", "argmin", "amax", "argmax", "median", "skew", "kurtosis"]
    )
    result_ds, target_result = transform((input_ds, target_ds))

    # Test 1: number of statistics and unchanged target
    assert list(result_ds.data_vars) == [
        "mean",
        "amin",
        "argmin",
        "amax",
        "argmax",
        "median",
        "skew",
        "kurtosis",
    ]
    assert target_result.equals(target_ds)

    # Test 2: Mean
    expected_means = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(result_ds["mean"].values, expected_means)

    # Test 3: Min and max
    expected_mins = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_maxs = np.array([[9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_allclose(result_ds["amin"].values, expected_mins)
    np.testing.assert_allclose(result_ds["amax"].values, expected_maxs)

    # Test 4: Argmin and argmax
    expected_argmins = np.array([[0.0, 0.0], [0.0, 0.0]])
    expected_argmaxs = np.array([[2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_allclose(result_ds["argmin"].values, expected_argmins)
    np.testing.assert_allclose(result_ds["argmax"].values, expected_argmaxs)

    # Test 5: Median
    expected_medians = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(result_ds["median"].values, expected_medians)

    # Test 6: Skew and kurtosis
    expected_skews = np.array([[0.0, 0.0], [0.0, 0.0]])
    expected_kurtosis = np.array([[-1.5, -1.5], [-1.5, -1.5]])
    np.testing.assert_allclose(result_ds["skew"].values, expected_skews)
    np.testing.assert_allclose(result_ds["kurtosis"].values, expected_kurtosis)

    # Test 7: Empty statistic list
    module = ExtractInputStatisticalFeatures([])
    result_ds, target_result = module((input_ds, target_ds))

    assert len(result_ds.data_vars) == 0
    assert target_result.equals(target_ds)


def test_FillMissingModels():
    """Test of FillMissingModels tranform."""
    input_ds = make_input_ds({model: np.ones((2, 2)) for model in MODELS_NAMES[:9]})
    target_ds = xr.Dataset({"analysis": (["latitude", "longitude"], np.ones((2, 2)))})

    transform = FillMissingModels(fill_value=0)
    result_ds, target_result = transform((input_ds, target_ds))

    # Test 1: all 11 models present and unchanged target
    assert list(result_ds.data_vars) == list(MODELS_NAMES)
    assert target_result.equals(target_ds)

    # Test 2: output added models contains only 0
    missing_models = MODELS_NAMES[9:]
    assert len(missing_models) == 2
    for model in missing_models:
        np.testing.assert_allclose(result_ds[model].values, np.zeros((2, 2)))


@pytest.fixture
def x_named_ds() -> xr.Dataset:
    """Fixture used by the transform tests that returns fake input data."""
    tensor = np.array([[float("nan"), 1.0], [2.0, float("nan")]])
    return make_input_ds({"O3": tensor})


@pytest.fixture
def y_named_ds() -> xr.Dataset:
    """Fixture used by the transform tests that returns fake target data."""
    tensor = np.array([[float("nan"), 5.0], [9.0, float("nan")]])
    return xr.Dataset({"O3": (["latitude", "longitude"], tensor)})


@pytest.fixture
def stats_file_path(tmp_path: Path) -> Path:
    """Creates a fake file of CAMS data statistics"""
    stats_dict = {
        "O3": {"min": 1.0, "max": 5.0},
    }
    path_file = tmp_path / "stats.json"
    with open(path_file, "w") as f:
        json.dump(stats_dict, f, indent=4)
    return path_file


expected_x = np.array(
    [[float("nan"), 0], [0.25, float("nan")]],
    dtype=np.float64,
)
expected_y = np.array(
    [[float("nan"), 1.0], [2.0, float("nan")]],
    dtype=np.float64,
)


def test_normalize(
    x_named_ds: xr.Dataset, y_named_ds: xr.Dataset, stats_file_path: Path
):
    """Test of Normalize transform."""
    transform = Normalize(stats_file_path=stats_file_path)
    x_processed, y_processed = transform((x_named_ds, y_named_ds))
    np.testing.assert_allclose(
        np.nan_to_num(x_processed["O3"].values), np.nan_to_num(expected_x)
    )
    np.testing.assert_allclose(
        np.nan_to_num(y_processed["O3"].values), np.nan_to_num(expected_y)
    )


def test_reverse_normalize(x_named_ds: xr.Dataset, stats_file_path: Path):
    """Test the NamedTensor based reverse of the Normalize transform."""
    transform = Normalize(stats_file_path=stats_file_path)
    x_processed, _ = transform((x_named_ds, x_named_ds))

    x_nt = NamedTensor(
        tensor=torch.tensor(x_processed["O3"].values[None, ...]).float(),
        names=["features", "lat", "lon"],
        feature_names=["O3"],
    )
    y_nt = NamedTensor(
        tensor=torch.ones(1, 2, 2),
        names=["features", "lat", "lon"],
        feature_names=["O3"],
    )

    reversed_transform = transform.reverse_transform()
    x_reversed, _ = reversed_transform((x_nt, y_nt))

    expected = torch.tensor(x_named_ds["O3"].values[None, ...]).float()
    assert torch.allclose(torch.isnan(x_reversed.tensor), torch.isnan(expected))
    assert torch.allclose(
        torch.nan_to_num(x_reversed.tensor), torch.nan_to_num(expected)
    )
