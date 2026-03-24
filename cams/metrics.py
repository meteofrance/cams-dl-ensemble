from typing import overload

import torch
import torchmetrics as tm
from mfai.pytorch.metrics import FAR
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from typing_extensions import override


class MeanSquaredError(tm.MeanSquaredError):
    """MeanSquaredError wrapper to works with NamedTensor."""

    @overload
    def update(self, preds: NamedTensor, target: NamedTensor) -> None: ...
    @overload
    def update(self, preds: Tensor, target: Tensor) -> None: ...
    @override
    def update(self, preds: NamedTensor | Tensor, target: NamedTensor | Tensor) -> None:
        preds = preds.tensor if isinstance(preds, NamedTensor) else preds
        target = target.tensor if isinstance(target, NamedTensor) else target
        super().update(preds, target)


class MeanAbsoluteError(tm.MeanAbsoluteError):
    """MeanSquaredError wrapper to works with NamedTensor."""

    @overload
    def update(self, preds: NamedTensor, target: NamedTensor) -> None: ...
    @overload
    def update(self, preds: Tensor, target: Tensor) -> None: ...
    @override
    def update(self, preds: NamedTensor | Tensor, target: NamedTensor | Tensor) -> None:
        preds = preds.tensor if isinstance(preds, NamedTensor) else preds
        target = target.tensor if isinstance(target, NamedTensor) else target
        super().update(preds, target)


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

    @overload
    def update(self, preds: NamedTensor, target: NamedTensor) -> None: ...
    @overload
    def update(self, preds: Tensor, target: Tensor) -> None: ...
    @override
    def update(self, preds: NamedTensor | Tensor, target: NamedTensor | Tensor) -> None:
        preds = preds[self.feature] if isinstance(preds, NamedTensor) else preds
        target = target[self.feature] if isinstance(target, NamedTensor) else target

        assert preds.shape == target.shape

        binary_preds = torch.where(preds >= self.feature_threshold, 1, 0)
        binary_target = torch.where(target >= self.feature_threshold, 1, 0)

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

    @overload
    def update(self, preds: NamedTensor, target: NamedTensor) -> None: ...
    @overload
    def update(self, preds: Tensor, target: Tensor) -> None: ...
    @override
    def update(self, preds: NamedTensor | Tensor, target: NamedTensor | Tensor) -> None:
        preds = preds[self.feature] if isinstance(preds, NamedTensor) else preds
        target = target[self.feature] if isinstance(target, NamedTensor) else target

        assert preds.shape == target.shape

        binary_preds = torch.where(preds >= self.feature_threshold, 1, 0)
        binary_target = torch.where(target >= self.feature_threshold, 1, 0)

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

    @overload
    def update(self, preds: NamedTensor, target: NamedTensor) -> None: ...
    @overload
    def update(self, preds: Tensor, target: Tensor) -> None: ...
    @override
    def update(self, preds: NamedTensor | Tensor, target: NamedTensor | Tensor) -> None:
        preds = preds[self.feature] if isinstance(preds, NamedTensor) else preds
        target = target[self.feature] if isinstance(target, NamedTensor) else target
        assert preds.shape == target.shape

        binary_preds = torch.where(preds >= self.feature_threshold, 1, 0)
        binary_target = torch.where(target >= self.feature_threshold, 1, 0)

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
