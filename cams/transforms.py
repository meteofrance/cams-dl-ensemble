from typing import Literal

import scipy.stats
import torch
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor, nn


class ReplaceEnsembleByStatisctics(nn.Module):
    """Replace ensemble data by statistical features.

    This module computes various statistical measures from ensemble data and
    replaces the original ensemble data with these statistics while preserving
    the target data unchanged.

    Attributes:
        statistic_types: List of statistical measures to compute from the input.
            Supported statistics include: 'mean', 'amin', 'argmin', 'amax', 
            'argmax', 'median', 'skew', 'kurtosis'.

    Note:
        For 'skew' and 'kurtosis', you should first enable the array API standard support.
        -> https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
    """

    def __init__(
        self,
        statistic_types: list[
            Literal[
                "mean", "amin", "argmin", "amax", "argmax", "median", "skew", "kurtosis"
            ]
        ],
    ):
        """
        Args:
            statistic_types: List of statistical measures to compute.
                Must be one or more of: 'mean', 'amin', 'argmin', 'amax', 
                'argmax', 'median', 'skew', 'kurtosis'.
        """
        super().__init__()
        self.statistic_types = statistic_types

    def forward(
        self, input_nt: NamedTensor, target_nt: NamedTensor
    ) -> tuple[NamedTensor, NamedTensor]:
        """Compute statistical features from input and return them with target.

        Args:
            input_nt: Input NamedTensor containing ensemble data with spatial dimensions.
            target_nt: Target NamedTensor to be preserved unchanged.

        Returns:
            NamedTensor: Computed statistics as features.
            NamedTensor: Target unchanged.

        Note:
            For skew and kurtosis, scipy.stats is used with nan_policy="omit".
            For median, only the values are returned (not indices) from torch.median().
        """
        input_tensor: Tensor = input_nt.tensor
        stat_tensor: Tensor = torch.empty(
            len(self.statistic_types),
            *[input_tensor.shape[idx] for idx in input_nt.spatial_dim_idx],
        )
        for idx, statistic_type in enumerate(self.statistic_types):
            if statistic_type in ["skew", "kurtosis"]:
                stat_tensor[idx, :, :] = getattr(scipy.stats, statistic_type)(
                    input_tensor, axis=0, nan_policy="omit"
                )
            elif (
                statistic_type == "median"
            ):
                # Tensor.median() returns a tuple[values, indices], so we keep values
                stat_tensor[idx, :, :] = getattr(input_tensor, statistic_type)(dim=0)[0]
            else:
                stat_tensor[idx, :, :] = getattr(input_tensor, statistic_type)(dim=0)
        stat_nt = NamedTensor(
            stat_tensor,
            names=["features", "lat", "lon"],
            feature_names=self.statistic_types,
        )
        return stat_nt, target_nt
