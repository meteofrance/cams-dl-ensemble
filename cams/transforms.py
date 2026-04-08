import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Sequence

import scipy.stats
import torch
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor, nn
from typing_extensions import override

from cams.settings import MODEL_NAMES, STATS_PATH
from cams.types import STATISTICS_NAMES, StatisticsNames


class ExtractInputStatisticalFeatures(nn.Module):
    """Replace ensemble data by statistical features.

    This module computes various statistical measures from ensemble data and
    replaces the original ensemble data with these statistics while preserving
    the target data unchanged.

    Attributes:
        statistic_types: List of statistical measures to compute from the input.
            Supported statistics include: 'mean', 'amin', 'argmin', 'amax',
            'argmax', 'median', 'skew', 'kurtosis'.

    Note:
        For 'skew' and 'kurtosis', you should enable the array API standard support.
        -> https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
    """

    def __init__(self, statistic_types: Sequence[StatisticsNames]):
        """
        Args:
            statistic_types: List of statistical measures to compute.
                Must be one or more of: 'mean', 'amin', 'argmin', 'amax',
                'argmax', 'median', 'skew', 'kurtosis'.
        """
        super().__init__()
        self.statistic_types = statistic_types

        if "skew" in self.statistic_types or "kurtosis" in self.statistic_types:
            scipy_array_api = os.getenv("SCIPY_ARRAY_API")
            if scipy_array_api != "1":
                raise RuntimeError(
                    "Environement variable 'SCIPY_ARRAY_API' should be set to '1' to "
                    + "use 'skew' and/or 'kurtosis' statistics. See "
                    + "https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html for"
                    + "more details about scipy array API support."
                )

    @override
    def forward(
        self, inputs: tuple[NamedTensor, NamedTensor]
    ) -> tuple[NamedTensor, NamedTensor]:
        """Compute statistical features from input and return them with target.

        Args:
            inputs: tuple NamedTensor containing ensemble data with spatial dimensions.

        Returns:
            NamedTensor: Computed statistics as features.
            NamedTensor: Target unchanged.

        Note:
            For skew and kurtosis, scipy.stats is used with nan_policy="omit".
            For median, only the values are returned (not indices) from torch.median().
        """
        x, y = inputs
        input_tensor: Tensor = x.tensor
        stat_tensor: Tensor = torch.empty(
            len(self.statistic_types),
            *[input_tensor.shape[idx] for idx in x.spatial_dim_idx],
        )
        for idx, statistic_type in enumerate(self.statistic_types):
            if statistic_type in ["skew", "kurtosis"]:
                stat_tensor[idx, :, :] = getattr(scipy.stats, statistic_type)(
                    input_tensor, axis=0, nan_policy="omit"
                )
            elif statistic_type == "median":
                # Tensor.median() returns a tuple[values, indices], so we keep values
                stat_tensor[idx, :, :] = getattr(input_tensor, statistic_type)(dim=0)[0]
            else:
                stat_tensor[idx, :, :] = getattr(input_tensor, statistic_type)(dim=0)
        stat_nt = NamedTensor(
            stat_tensor,
            names=["features", "lat", "lon"],
            feature_names=self.statistic_types,
        )
        return stat_nt, y


class ReversibleTransformMixin:
    """Mixin class that enables a transform to specify a reverse transform."""

    @abstractmethod
    def reverse_transform(self) -> nn.Module:
        """Another transform that reverses the current transform."""
        raise NotImplementedError()


def load_stats(stats_path: Path) -> dict[str, Any]:
    """Loads statistics about the species from a file."""
    with open(stats_path, "r") as file:
        stats = json.load(file)
    return stats

class FillMissingModels(nn.Module):
    """Add missing models at the right index with values of zeros"""

    @override
    def forward(
        self, inputs: tuple[NamedTensor, NamedTensor]
    ) -> tuple[NamedTensor, NamedTensor]:
        """

        Args:
            input: NamedTensor containing missing models

        Returns:
            NamedTensor: NamedTensor containing all the 11 models

        """
        x, y = inputs
        model_names_sorted = sorted(MODEL_NAMES)
        t_final = torch.zeros(
            len(model_names_sorted), 
            x.tensor.shape[1], 
            x.tensor.shape[2], 
            dtype=x.tensor.dtype, 
            device=x.tensor.device
        )
        for idx, model in enumerate(model_names_sorted):
            if model in x.feature_names:
                t_final[idx] = x[model]
        return NamedTensor(t_final, x.names, model_names_sorted), y

class Normalize(nn.Module, ReversibleTransformMixin):
    """Normalizes data.
    Requires a pre processed stats file generated by `python scripts/compute_stats.py`
    """

    def __init__(
        self,
        stats_file_path: Path = STATS_PATH,
        interval: list[int] = [0, 1],
    ) -> None:
        """A transform that normalizes data."""
        super().__init__()

        if not stats_file_path.exists():
            raise FileNotFoundError(
                f"Statistics file not found: {stats_file_path}. "
                "Please run `python scripts/compute_stats.py`."
            )
        self.stats_file_path = stats_file_path
        self.stats_dict = load_stats(self.stats_file_path)

    @override
    def reverse_transform(self) -> "ReverseNormalize":
        """Another transform that reverses the current transform."""
        return ReverseNormalize(self.stats_file_path)

    def normalize_namedtensor(self, nt: NamedTensor) -> NamedTensor:
        """Normalize a NamedTensor btw 0 and 1 with min/max normalization."""
        normalized_features: list[Tensor] = []
        for feature_name in nt.feature_names:
            mini = self.stats_dict["O3"]["min"]
            maxi = self.stats_dict["O3"]["max"]
            normalized_feature = (nt[feature_name] - mini) / (maxi - mini)
            normalized_features.append(normalized_feature)

        # Build normalized feature tensor
        normalized_features_tensor = torch.cat(
            tensors=normalized_features, dim=nt.feature_dim_idx
        )
        # Recreate a NamedTensor with the normalized features
        return NamedTensor.new_like(tensor=normalized_features_tensor, other=nt)

    @override
    def forward(
        self, inputs: tuple[NamedTensor, NamedTensor]
    ) -> tuple[NamedTensor, NamedTensor]:
        """Applies normalization."""
        x, y = inputs
        return self.normalize_namedtensor(x), self.normalize_namedtensor(y)


class ReverseNormalize(nn.Module):
    """Inverse normalization of data."""

    def __init__(
        self,
        stats_file_path: Path = STATS_PATH,
    ) -> None:
        """Inverse normalization of data."""
        super().__init__()
        self.stats_file_path = stats_file_path
        self.stats_dict = load_stats(self.stats_file_path)

    def denormalize_namedtensor(self, nt: NamedTensor) -> NamedTensor:
        """Undoes min/max normalization."""
        denormalized_features: list[Tensor] = []
        for feature_name in nt.feature_names:
            mini = self.stats_dict["O3"]["min"]
            maxi = self.stats_dict["O3"]["max"]
            denormalized_features.append(nt[feature_name] * (maxi - mini) + mini)
        denormalized_features_tensor = torch.cat(
            tensors=denormalized_features, dim=nt.feature_dim_idx
        )
        return NamedTensor.new_like(tensor=denormalized_features_tensor, other=nt)

    @override
    def forward(
        self, inputs: tuple[NamedTensor, NamedTensor]
    ) -> tuple[NamedTensor, NamedTensor]:
        """Applies denormalization."""
        x, y = inputs
        return self.denormalize_namedtensor(x), self.denormalize_namedtensor(y)


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a Transform
    import datetime as dt
    from pathlib import Path

    from mfai.pytorch.namedtensor import NamedTensor

    from cams.plots import plot_named_tensor
    from cams.sample import Sample
    from cams.types import STATISTICS_NAMES

    sample = Sample(dt.datetime(2024, 7, 30), 15)
    x, y = sample.input_data, sample.target_data
    transform = ExtractInputStatisticalFeatures(STATISTICS_NAMES)
    x_transformed, _ = transform((x, y))
    nt = NamedTensor.concat([x, x_transformed, y])
    print(nt)
    plot_named_tensor(nt, "O3", Path("test_transform.png"))
