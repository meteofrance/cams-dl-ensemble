import json
import os
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.stats
import torch
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor
from torch import nn
from typing_extensions import override

from cams.settings import STATS_PATH
from cams.types import MODELS_NAMES, STATISTICS_NAMES, StatisticsNames

SPATIAL_DIMS = ["latitude", "longitude"]


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

    def __init__(self, statistic_types: Sequence[str]):
        """
        Args:
            statistic_types: List of statistical measures to compute.
                Must be one or more of: 'mean', 'amin', 'argmin', 'amax',
                'argmax', 'median', 'skew', 'kurtosis', 'std'.
        """
        super().__init__()
        if not all(stat in STATISTICS_NAMES for stat in statistic_types):
            raise ValueError(
                "Transform ExtractInputStatisticalFeatures init parameter "
                f"statistic_types to contain values {STATISTICS_NAMES} "
            )
        self.statistic_types = cast(StatisticsNames, statistic_types)

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
        self, inputs: tuple[xr.Dataset, xr.Dataset]
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Compute statistical features from input and return them with target.

        Args:
            inputs: Tuple of input and target xarray datasets.

        Returns:
            xr.Dataset: Computed statistics as features.
            xr.Dataset: Target unchanged.

        Note:
            For skew and kurtosis, scipy.stats is used with nan_policy="omit".
            For the other statistics, NaN values propagate like with torch,
            so skipna is deactivated.
        """
        x, y = inputs
        ensemble = x.to_array(dim="model")
        stat_ds = xr.Dataset()
        for statistic_type in self.statistic_types:
            if statistic_type in ["skew", "kurtosis"]:
                statistic = xr.apply_ufunc(
                    getattr(scipy.stats, statistic_type),
                    ensemble,
                    input_core_dims=[["model"]],
                    kwargs={"nan_policy": "omit", "axis": -1},
                )
            elif statistic_type in ["argmin", "argmax"]:
                statistic = xr.apply_ufunc(
                    getattr(np, statistic_type),
                    ensemble,
                    input_core_dims=[["model"]],
                    kwargs={"axis": -1},
                )
            else:
                xarray_method = "min" if statistic_type == "amin" else statistic_type
                xarray_method = "max" if statistic_type == "amax" else xarray_method
                statistic = getattr(ensemble, xarray_method)(dim="model", skipna=False)
            stat_ds[statistic_type] = statistic.astype(float)
        return stat_ds, y


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
    """Add missing models at the right index with a given value."""

    def __init__(self, fill_value: int = -1) -> None:
        """
        Args:
            fill_value: Value used to fill missing models spaces with.
                Defaults to -1.
        """
        super().__init__()
        self.fill_value = fill_value

    @override
    def forward(
        self, inputs: tuple[xr.Dataset, xr.Dataset]
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Create an xarray Dataset that has all the 11 models.

        Args:
            inputs: Tuple of input and target xarray datasets. The input may
                contain missing models.

        Returns:
            xr.Dataset: Input dataset containing all the 11 models.
            xr.Dataset: Target unchanged.
        """
        x, y = inputs
        model_template = x[next(iter(x.data_vars))]
        for model in MODELS_NAMES:
            if model not in x:
                x[model] = xr.full_like(model_template, self.fill_value)
        return x, y


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
                "Please run `scripts/data/compute_stats.py`."
            )
        self.stats_file_path = stats_file_path
        self.stats_dict = load_stats(self.stats_file_path)

    @override
    def reverse_transform(self) -> "ReverseNormalize":
        """Another transform that reverses the current transform."""
        return ReverseNormalize(self.stats_file_path)

    def normalize_xarray(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalize an xarray Dataset btw 0 and 1 with min/max normalization.

        The dataset has one data variable per model, with a ``species``
        dimension. Each species is normalized with its own min/max statistics.
        """
        all_species_da = {}
        for model in ds.data_vars:
            da = ds[model]
            normalized_channels = []
            for species in da["species"].values:
                mini = self.stats_dict[species]["min"]
                maxi = self.stats_dict[species]["max"]
                normalized_channels.append(
                    (da.sel(species=species) - mini) / (maxi - mini)
                )
            all_species_da[model] = xr.concat(normalized_channels, dim="species")
        return xr.Dataset(all_species_da)

    @override
    def forward(
        self, inputs: tuple[xr.Dataset, xr.Dataset]
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Applies normalization."""
        x, y = inputs
        return self.normalize_xarray(x), self.normalize_xarray(y)


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
        denormalized_features: list[torch.Tensor] = []
        for feature_name in nt.feature_names:
            species = feature_name.split(" - ")[1]
            mini = self.stats_dict[species]["min"]
            maxi = self.stats_dict[species]["max"]
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

    from cams.plots import plot_named_tensor
    from cams.sample import Sample
    from cams.types import STATISTICS_NAMES

    sample = Sample(
        dt.datetime(2024, 7, 30),
        lead_times=[15],
        species=["O3"],
        levels=[0],
        models=["CHIMERE", "MOCAGE"],
    )
    ds = sample.data
    x, y = ds.drop_vars("TARGET"), ds[["TARGET"]]
    transform = ExtractInputStatisticalFeatures(STATISTICS_NAMES)
    x_transformed, _ = transform((x, y))
    nt = Sample.convert_data_to_nt(xr.concat([x, x_transformed], dim="model"))
    print(nt)
    plot_named_tensor(nt, "O3", Path("test_transform.png"))
