import datetime as dt
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor
from torch import nn
from torch.utils.data import Dataset
from typing_extensions import override

from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR
from cams.transforms import SPATIAL_DIMS
from cams.types import Leadtimes, Levels, ModelsNames, SpeciesNames


def get_run_dates(processed_dir: Path) -> list[dt.date]:
    """Retrieves the dates of all the runs available in a directory."""
    print("--> Retrieving run dates...")

    run_dates: list[dt.date] = []
    files_not_parsed: list[Path] = []
    for path in sorted(list(set(processed_dir.glob("**/*.netcdf")))):
        try:
            date = dt.datetime.strptime(path.stem.split("-")[0], r"%Y_%m_%d").date()
            run_dates.append(date)
        except Exception as e:
            files_not_parsed.append(path)
            print(e)
            continue
    if files_not_parsed:
        print(
            f"WARNING: {len(files_not_parsed)} files could not be parsed: "
            f"{', '.join([str(f) for f in files_not_parsed])}"
        )
    run_dates = sorted(list(set(run_dates)))  # remove duplicates
    return run_dates


def dataset_to_namedtensor(ds: xr.Dataset) -> NamedTensor:
    """Converts an xarray dataset into a NamedTensor of shape (features, lat, lon).

    Model-like variables (with species, time and level dims) are expanded into
    one channel per (species, leadtime, level) combination. Variables reduced to
    the spatial dims only (e.g. statistics) become a single channel named after
    the variable.

    Args:
        ds: The xarray dataset to convert.

    Returns:
        NamedTensor: The converted data.
    """
    channel_arrays: list[np.ndarray] = []
    channel_names: list[str] = []
    for var_name in ds.data_vars:
        da = ds[var_name]
        if not any(dim not in SPATIAL_DIMS for dim in da.dims):
            channel_arrays.append(np.nan_to_num(da.values, nan=0.0))
            channel_names.append(str(var_name))
            continue
        da = da.transpose("species", "time", "level", "latitude", "longitude")
        species_values = da.coords["species"].values
        time_values = da.coords["time"].values
        level_values = da.coords["level"].values
        for i_species, species in enumerate(species_values):
            for i_time in range(len(time_values)):
                for i_level, level in enumerate(level_values):
                    arr = da.isel(species=i_species, time=i_time, level=i_level).values
                    channel_arrays.append(np.nan_to_num(arr, nan=0.0))
                    leadtime = da.coords["lead_time"].values[i_time]
                    channel_name = (
                        f"{var_name} - {species} - +{leadtime}h - {int(level)}m"
                    )
                    channel_names.append(channel_name)

    tensor = torch.tensor(np.stack(channel_arrays, axis=0)).to(torch.float32)
    return NamedTensor(tensor, ["features", "lat", "lon"], channel_names)


class CAMSDataset(Dataset):
    """CAMS dataset, see [dataset doc](docs/data.md) for complete description."""

    def __init__(
        self,
        run_dates: list[dt.date],
        models: list[ModelsNames],
        lead_times: list[Leadtimes],
        species: list[SpeciesNames],
        levels: list[Levels],
        processed_dir: Path = PROCESSED_DATA_DIR,
        transform_sequence: nn.Sequential = nn.Sequential(*[]),
    ) -> None:
        """Loads the dataset's sample points for the given split.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            run_dates: The list of date to process.
            models: Models to load in the dataset.
            lead_times: Leadtimes to load in the dataset.
            species: Species to load in the dataset.
            levels: Levels to load in the dataset.
                '0' corresponds to 'ground' level.
            processed_dir: Path to the CAMS dataset's processed data.
            transform_sequence: transforms sequence applied to the data after loading.
        """
        self.run_dates = run_dates
        self.models = models
        self.lead_times = lead_times
        self.species = species
        self.levels = levels
        self.processed_dir = processed_dir
        self.transform_sequence = transform_sequence

    @cached_property
    def samples(self) -> list[Sample]:
        """Returns the list of valid samples in the dataset."""
        samples = [
            Sample(
                date_run=date_run,
                models=self.models,
                lead_times=self.lead_times,
                species=self.species,
                levels=self.levels,
                processed_dir=self.processed_dir,
            )
            for date_run in self.run_dates
        ]
        return [sample for sample in samples if sample.is_valid]

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, idx: int) -> tuple[NamedTensor, NamedTensor]:
        """Returns one sample of training data."""
        ds = self.samples[idx].data
        x_ds = ds.drop_vars("TARGET")
        y_ds = ds[["TARGET"]]
        x_ds, y_ds = self.transform_sequence((x_ds, y_ds))
        return dataset_to_namedtensor(x_ds), dataset_to_namedtensor(y_ds)


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a CAMSDataset

    run_dates: list[dt.date] = get_run_dates(PROCESSED_DATA_DIR)
    print(len(run_dates))
    dataset = CAMSDataset(
        run_dates,
        models=["CHIMERE", "MOCAGE"],
        lead_times=[15, 24],
        species=["O3", "NO2"],
        levels=[0],
    )
    print("Len dataset : ", len(dataset))

    sample = dataset.samples[10]
    print(sample)

    x, y = dataset[10]
    print(x, y)
