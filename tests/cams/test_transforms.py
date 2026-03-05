import torch
from mfai.pytorch.namedtensor import NamedTensor

from cams.transforms import ExtractInputStatisticalFeatures, Normalize
import pytest
import json
from pathlib import Path

def test_ExtractInputStatisticalFeatures():
    """Test of ExtractInputStatisticalFeatures tranform."""
    input_data = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                              [[5.0, 6.0], [7.0, 8.0]],
                              [[9.0, 10.0], [11.0, 12.0]]])
    input_nt = NamedTensor(
        input_data, names=["features", "lat", "lon"], feature_names=["model_A", "model_B", "model_C"]
    )
    target_nt = NamedTensor(torch.ones(1, 2, 2), names=["features", "lat", "lon"], feature_names=["analysis"])

    transform = ExtractInputStatisticalFeatures(
        ["mean", "amin", "argmin", "amax", "argmax", "median", "skew", "kurtosis"]
    )
    result_nt, target_nt_result = transform((input_nt, target_nt))

    # Test 1: output shape and unchanged target
    assert result_nt.tensor.shape == (8, 2, 2)
    assert target_nt_result == target_nt

    # Test 2: Mean
    expected_means = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    torch.testing.assert_close(result_nt["mean"], expected_means)

    # Test 3: Min and max
    expected_mins = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    expected_maxs = torch.tensor([[[9.0, 10.0], [11.0, 12.0]]])
    torch.testing.assert_close(result_nt["amin"], expected_mins)
    torch.testing.assert_close(result_nt["amax"], expected_maxs)

    # Test 4: Argmin and argmax
    expected_argmins = torch.tensor([[[0., 0.], [0., 0.]]])
    expected_argmaxs = torch.tensor([[[2., 2.], [2., 2.]]])
    torch.testing.assert_close(result_nt["argmin"], expected_argmins)
    torch.testing.assert_close(result_nt["argmax"], expected_argmaxs)

    # Test 5: Median
    expected_medians = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    torch.testing.assert_close(result_nt["median"], expected_medians)

    # Test 6: Skew and kurtosis
    expected_skews = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    expected_kurtosis = torch.tensor([[[-1.5, -1.5], [-1.5, -1.5]]])
    torch.testing.assert_close(result_nt["skew"], expected_skews)
    torch.testing.assert_close(result_nt["kurtosis"], expected_kurtosis)

    # Test 7: Empty statistic list
    module = ExtractInputStatisticalFeatures([])
    result_nt, target_nt_result = module((input_nt, target_nt))

    assert result_nt.tensor.shape == (0, 2, 2)
    assert target_nt_result == target_nt

@pytest.fixture
def x_named_tensor() -> NamedTensor:
    """Fixture used by the transform tests that returns fake input data."""
    tensor = torch.tensor([[[float("nan"), 1.0], [2.0, float("nan")]]])
    return NamedTensor(
        tensor=tensor,
        names=["features", "lat", "lon"],
        feature_names=["O3"],
        feature_dim_name="features",
    )


@pytest.fixture
def y_named_tensor() -> NamedTensor:
    """Fixture used by the transform tests that returns fake target data."""
    tensor = torch.tensor([[[float("nan"), 5.0], [9.0, float("nan")]]])
    return NamedTensor(
        tensor=tensor,
        names=["features", "lat", "lon"],
        feature_names=["O3",],
        feature_dim_name="features",
    )

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


expected_x = torch.tensor(
    [[[float("nan"), 0], [0.25, float("nan")]]],
    dtype=torch.float32,
)
expected_y = torch.tensor(
    [[[float("nan"), 1.0], [2.0, float("nan")]]],
    dtype=torch.float32,
)

def test_normalize(x_named_tensor: NamedTensor, y_named_tensor:NamedTensor, stats_file_path: Path):
    transform = Normalize(stats_file_path=stats_file_path)
    x_processed, y_processed = transform((x_named_tensor, y_named_tensor))
    assert torch.allclose(
        torch.nan_to_num(x_processed.tensor), torch.nan_to_num(expected_x)
    )
    assert torch.allclose(
        torch.nan_to_num(y_processed.tensor), torch.nan_to_num(expected_y)
    )

    reversed_transform = transform.reverse_transform()
    x_reversed, y_reversed = reversed_transform((x_processed, y_processed))
    assert torch.allclose(
        torch.isnan(x_reversed.tensor), torch.isnan(x_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(x_reversed.tensor), torch.nan_to_num(x_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.isnan(y_reversed.tensor), torch.isnan(y_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(y_reversed.tensor), torch.nan_to_num(y_named_tensor.tensor)
    )