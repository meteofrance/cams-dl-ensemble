import torch
from mfai.pytorch.namedtensor import NamedTensor

from cams.transforms import ExtractInputStatisticalFeatures


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
    module = ReplaceEnsembleByStatisctics([])
    result_nt, target_nt_result = module(input_nt, target_nt)

    assert result_nt.tensor.shape == (0, 2, 2)
    assert target_nt_result == target_nt
