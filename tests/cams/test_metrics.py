import pytest
import torch
from mfai.pytorch.namedtensor import NamedTensor
import torchmetrics as tm
from cams.metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Accuracy,
    F1Score,
    FalseAlarmRate,
    Bias,
    FalsePositiveRate
)

@pytest.fixture
def preds() -> NamedTensor:
    """Fake predicted NamedTensor."""
    preds = torch.tensor([[[125, 100], [125, 50]]], dtype=torch.float32)
    return NamedTensor(preds, names=['features', 'lat', 'lon'], feature_names=["O3"])

@pytest.fixture
def target() -> NamedTensor:
    """Fake target NamedTensor."""
    target = torch.tensor([[[125, 125], [100, 75]]], dtype=torch.float32)
    return NamedTensor(target, names=['features', 'lat', 'lon'], feature_names=["O3"])


def test_MeanSquaredError(preds: NamedTensor, target: NamedTensor) -> None:
    """Testing MeanSquaredError wrapper."""
    cams_metric = MeanSquaredError()
    tm_metric = tm.MeanSquaredError()
    torch.testing.assert_close(cams_metric(preds, target), tm_metric(preds.tensor, target.tensor))

    cams_metric = MeanSquaredError(squared=False)
    tm_metric = tm.MeanSquaredError(squared=False)
    torch.testing.assert_close(cams_metric(preds, target), tm_metric(preds.tensor, target.tensor))

def test_MeanAbsoluteError(preds: NamedTensor, target: NamedTensor) -> None:
    """Testing MeanAbsoluteError wrapper."""
    cams_metric = MeanAbsoluteError()
    tm_metric = tm.MeanAbsoluteError()
    torch.testing.assert_close(cams_metric(preds, target), tm_metric(preds.tensor, target.tensor))


# According to the NamedTensor above, we have: TP = TN = FP = FN = 1
@pytest.mark.parametrize("metric_cls,expected",
    [
        (Accuracy, 0.5),  # Accuracy = 2/4
        (Bias, 1.),  # Bias = 2 / 2
        (F1Score, 0.5),  # Precision = Recall = 0.5 -> F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5)
    ]
)
def test_metrics_wrapper(metric_cls: tm.Metric, expected: float, preds: NamedTensor, target: NamedTensor) -> None:
    """Testing all classification metrics wrapper."""
    metric = metric_cls(feature="O3", threshold=120)
    torch.testing.assert_close(metric(preds, target), torch.tensor(expected), rtol=1e-3, atol=1e-5)

    metric = metric_cls(feature="O3", threshold=20)
    torch.testing.assert_close(metric(preds, target), torch.tensor(1.0))

    with pytest.raises(ValueError):
        metric = metric_cls(feature="fake_feature", threshold=120)
        metric(preds, target)

def test_metrics_FalseAlarmRate(preds: NamedTensor, target: NamedTensor) -> None:
    """Testing FalseAlarmRate metric wrapper."""
    metric = FalseAlarmRate(feature="O3", threshold=120)
    torch.testing.assert_close(metric(preds, target), torch.tensor(0.5), rtol=1e-3, atol=1e-5)

    metric = FalseAlarmRate(feature="O3", threshold=150)
    torch.testing.assert_close(metric(preds, target), torch.tensor(1.0))

    with pytest.raises(ValueError):
        metric = FalseAlarmRate(feature="fake_feature", threshold=120)
        metric(preds, target)

def test_metrics_FalsePositiveRate(preds: NamedTensor, target: NamedTensor) -> None:
    """Testing FalsePositiveRate metric."""
    metric = FalsePositiveRate(feature="O3", threshold=120)
    torch.testing.assert_close(metric(preds, target), torch.tensor(0.5), rtol=1e-3, atol=1e-5)

    metric = FalsePositiveRate(feature="O3", threshold=50)
    assert torch.isnan(metric(preds, target))

    with pytest.raises(ValueError):
        metric = FalsePositiveRate(feature="fake_feature", threshold=120)
        metric(preds, target)