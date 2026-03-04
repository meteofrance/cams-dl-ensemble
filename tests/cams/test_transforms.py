import torch
from mfai.pytorch.namedtensor import NamedTensor

from cams.transforms import ExtractInputStatisticalFeatures, Normalize
import pytest
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
    result_nt, target_nt_result = transform(input_nt, target_nt)
    
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
    result_nt, target_nt_result = module(input_nt, target_nt)

    assert result_nt.tensor.shape == (0, 2, 2)
    assert target_nt_result == target_nt

@pytest.fixture
def x_named_tensor() -> NamedTensor:
    """Fixture used by the transform tests that returns fake input data."""
    feature = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
    landsea_mask = torch.tensor([[0, 1], [1, 0]])
    tensor = torch.stack([feature, landsea_mask], 0)
    return NamedTensor(
        tensor=tensor,
        names=["features", "lat", "lon"],
        feature_names=["feature0", "landsea_mask"],
        feature_dim_name="features",
    )


@pytest.fixture
def y_named_tensor() -> NamedTensor:
    """Fixture used by the transform tests that returns fake target data."""
    feature1 = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
    feature2 = torch.tensor([[float("nan"), 4.0], [3.0, float("nan")]])
    tensor = torch.stack([feature1, feature2], 0)
    return NamedTensor(
        tensor=tensor,
        names=["features", "lat", "lon"],
        feature_names=["feature1", "feature2"],
        feature_dim_name="features",
    )

@pytest.fixture
def stats_file_path(tmp_path: Path) -> Path:
    """Creates a fake file of CAMS data statistics"""
    stats_dict = {
        "feature0": {"min": 1.0, "max": 5.0},
        "feature1": {"min": 0.0, "max": 2.0},
        "feature2": {"min": -1.0, "max": 3.0},
    }
    path_file = tmp_path / "stats.pt"
    torch.save(stats_dict, path_file)
    return path_file


expected_x = torch.tensor(
    [[[float("nan"), 0], [0.25, float("nan")]], [[0, 1], [1, 0]]],
    dtype=torch.float32,
)
expected_y = torch.tensor(
    [
        [[float("nan"), 0.5], [1.0, float("nan")]],
        [[float("nan"), 1.25], [1.0, float("nan")]],
    ],
    dtype=torch.float32,
)

def test_normalize(c: NamedTensor, y:NamedTensor, stats_file_path: Path):
    transform = Normalize(stats_file_path=stats_file_path)
    x_processed, y_processed = transform(c, y)

    print(x_processed.tensor)
    print(y_processed.tensor)
    assert torch.allclose(
        torch.nan_to_num(x_processed.tensor), torch.nan_to_num(expected_x)
    )
    assert torch.allclose(
        torch.nan_to_num(y_processed.tensor), torch.nan_to_num(expected_y)
    )

    reversed_transform = transform.reverse_transform()
    x_reversed, y_reversed = reversed_transform(x_processed, y_processed)
    assert torch.allclose(
        torch.isnan(x_reversed.tensor), torch.isnan(c.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(x_reversed.tensor), torch.nan_to_num(c.tensor)
    )
    assert torch.allclose(
        torch.isnan(y_reversed.tensor), torch.isnan(y.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(y_reversed.tensor), torch.nan_to_num(y.tensor)
    )