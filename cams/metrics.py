import torch
from mfai.pytorch.metrics import FAR
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor
import torchmetrics as tm
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from typing_extensions import override


class MeanSquaredError(tm.MeanSquaredError):
    """MeanSquaredError wrapper to works with NamedTensor."""

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        super().update(preds.tensor, target.tensor)


class MeanAbsoluteError(tm.MeanAbsoluteError):
    """MeanSquaredError wrapper to works with NamedTensor."""

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        super().update(preds.tensor, target.tensor)


class Accuracy(BinaryAccuracy):
    """Compute Accuracy over a feature with a specific threshold.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """

    def __init__(self, feature: str, threshold: float) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            threshold: Threshold value used to binarize both predictions
            and target features.
        """
        super().__init__()
        self.feature: str = feature
        self.feature_threshold: Tensor = torch.tensor(threshold)

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = target[self.feature]
        assert preds_feature.shape == target_feature.shape

        binary_preds = torch.where(preds_feature >= self.feature_threshold, 1, 0)
        binary_target = torch.where(target_feature >= self.feature_threshold, 1, 0)

        super().update(binary_preds, binary_target)


class F1Score(BinaryF1Score):
    """Compute F1 Score over a feature with a specific threshold.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """

    def __init__(self, feature: str, threshold: float) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            threshold: Threshold value used to binarize both predictions
            and target features.
        """
        super().__init__()
        self.feature: str = feature
        self.feature_threshold: Tensor = torch.tensor(threshold)

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = target[self.feature]
        assert preds_feature.shape == target_feature.shape

        binary_preds = torch.where(preds_feature >= self.feature_threshold, 1, 0)
        binary_target = torch.where(target_feature >= self.feature_threshold, 1, 0)

        super().update(binary_preds, binary_target)


class FalseAlarmRate(FAR):
    """Compute False Alarm Rate over a feature with a specific threshold.

    FAR = FP / (FP + TP)
    """

    def __init__(self, feature: str, threshold: float) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            threshold: Threshold value used to binarize both predictions
            and target features.
        """
        super().__init__(task="binary")
        self.feature = feature
        self.feature_threshold = threshold

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = target[self.feature]
        assert preds_feature.shape == target_feature.shape

        binary_preds = torch.where(preds_feature >= self.feature_threshold, 1, 0)
        binary_target = torch.where(target_feature >= self.feature_threshold, 1, 0)

        super().update(binary_preds, binary_target)


class Bias(Metric):
    """Compute Bias over a feature with a specific threshold.
    https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2015/Roebber2009.pdf

    Bias = (TP + FP) / (TP + FN)
    """

    def __init__(self, feature: str, threshold: float) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            threshold: Threshold value used to binarize both predictions
            and target values.
        """
        super().__init__()
        full_state_update = True  # noqa
        self.feature: str = feature
        self.threshold: Tensor = torch.tensor(threshold)

        self.true_positives: Tensor
        self.false_positives: Tensor
        self.false_negatives: Tensor
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = target[self.feature]
        assert preds_feature.shape == target_feature.shape

        binary_preds = torch.where(preds_feature >= self.threshold, 1, 0)
        binary_target = torch.where(target_feature >= self.threshold, 1, 0)
        self.true_positives += torch.sum((binary_preds == 1) & (binary_target == 1))
        self.false_positives += torch.sum((binary_preds == 1) & (binary_target == 0))
        self.false_negatives += torch.sum((binary_preds == 0) & (binary_target == 1))

    @override
    def compute(self) -> Tensor:
        return (self.true_positives + self.false_positives) / (
            self.true_positives + self.false_negatives
        )


class FalsePositiveRate(Metric):
    """Compute False Positive Rate over a feature with a specific threshold.

    FPR = FP / (FP + TN)
    """

    def __init__(self, feature: str, threshold: float) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            threshold: Threshold value used to binarize both predictions
            and target values.
        """
        super().__init__()
        full_state_update = True  # noqa
        self.feature: str = feature
        self.threshold: Tensor = torch.tensor(threshold)

        self.true_negatives: Tensor
        self.false_positives: Tensor
        self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = target[self.feature]
        assert preds_feature.shape == target_feature.shape

        binary_preds = torch.where(preds_feature >= self.threshold, 1, 0)
        binary_target = torch.where(target_feature >= self.threshold, 1, 0)
        self.true_negatives += torch.sum((binary_preds == 0) & (binary_target == 0))
        self.false_positives += torch.sum((binary_preds == 1) & (binary_target == 0))

    @override
    def compute(self) -> Tensor:
        return self.false_positives / (self.false_positives + self.true_negatives)
