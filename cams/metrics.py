import torch
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor
from torchmetrics import Metric
from typing_extensions import override


class Accuracy(Metric):
    """Compute Accuracy over a feature with a specific threshold.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """

    def __init__(self, feature: str, threshold: float) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            title: Threshold value used to binarize both predictions and target values.
        """
        super().__init__()
        full_state_update = True  # noqa
        self.feature = feature
        self.threshold = threshold

        self.true_positives: Tensor
        self.true_negatives: Tensor
        self.false_positives: Tensor
        self.false_negatives: Tensor
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = preds[self.feature]
        assert preds_feature.shape == target_feature.shape

        binary_preds = torch.where(preds_feature >= self.threshold, 1, 0)
        binary_target = torch.where(target_feature >= self.threshold, 1, 0)
        self.true_positives += torch.sum((binary_preds == 1) & (binary_target == 1))
        self.true_positives += torch.sum((binary_preds == 0) & (binary_target == 0))
        self.false_positives += torch.sum((binary_preds == 1) & (binary_target == 0))
        self.false_negatives += torch.sum((binary_preds == 0) & (binary_target == 1))

    @override
    def compute(self) -> Tensor:
        return (self.true_positives + self.true_negatives) / (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )


class Bias(Metric):
    """Compute Accuracy over a feature with a specific threshold.
    https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2015/Roebber2009.pdf

    Bias = (TP + FP) / (TP + FN)
    """

    def __init__(self, feature: str = None, threshold: float = None) -> None:
        """
        Args:
            feature: Name of the feature on which the score is computed.
            title: Threshold value used to binarize both predictions and target values.
        """
        super().__init__()
        full_state_update = True  # noqa
        self.feature = feature
        self.threshold = threshold

        self.true_positives: Tensor
        self.false_positives: Tensor
        self.false_negatives: Tensor
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, preds: NamedTensor, target: NamedTensor) -> None:
        preds_feature: Tensor = preds[self.feature]
        target_feature: Tensor = preds[self.feature]
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
