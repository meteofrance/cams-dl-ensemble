import datetime as dt
from pathlib import Path
from typing import Hashable

import numpy as np
import torch
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor
from typing_extensions import override

from cams.settings import PROCESSED_DATA_DIR
from cams.types import Leadtimes, Levels, ModelsNames, SpeciesNames


class Sample:
    """CAMS sample.

    Responsibilities:
        Load a sample from the CAMS dataset from a given run date,
        and list of species, levels, models and leadtimes.
    """

    def __init__(
        self,
        date_run: dt.date,
        models: list[ModelsNames],
        lead_times: list[Leadtimes],
        species: list[SpeciesNames],
        levels: list[Levels],
        processed_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        """
        Args:
            date_run: The run date of the CTMs from which to load the sample.
            models: Models to load.
            lead_times: Leadtimes to load.
            species: Species to load.
            levels: Levels to load.
            processed_dir: Path to the CAMS processed dataset.
        """
        self.date_run = date_run
        self.models = models
        self.lead_times = lead_times
        self.valid_times = [
            (
                dt.datetime(date_run.year, date_run.month, date_run.day, 0, 0, 0)
                + dt.timedelta(hours=lt)
            )
            for lt in lead_times
        ]
        self.species = species
        self.levels = levels
        self.processed_dir = processed_dir

    @override
    def __str__(self) -> str:
        date_run_str = self.date_run.strftime("%Y-%m-%d")
        return (
            f"Sample(date_run={date_run_str}, "
            f"lead_times=+{self.lead_times}h, "
            f"species={self.species}), "
            f"models={self.models}), "
            f"levels={self.levels})"
        )

    @property
    def input_filename(self) -> str:
        """The standard filename for all input files."""
        date_run_str = self.date_run.strftime("%Y_%m_%d")
        filename = f"{date_run_str}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        return filename

    @property
    def input_paths(self) -> list[Path]:
        """The path to the netcdf of input ensemble data."""
        return [
            self.processed_dir / model.lower() / self.input_filename
            for model in self.models
        ]

    @property
    def target_paths(self) -> list[Path]:
        """The paths to the netcdf of targets reanalysis data.
        Files are grouped by months and species.
        """
        months_str = sorted({date.strftime("%Y-%m") for date in self.valid_times})
        return [
            self._target_path(species, month)
            for month in months_str
            for species in self.species
        ]

    def _target_path(self, species: SpeciesNames, month: str) -> Path:
        """Return the reanalysis file path for a species and month.

        Falls back to the Intermediate analysis (IRA) file when the
        VRA file does not exist.

        Args:
            folder: Directory containing the reanalysis files.
            species: The species to load.
            month: The month (YYYY-MM) of the desired file.

        Returns:
            Path to the reanalysis file for the given species and month.
        """
        folder = self.processed_dir / "reanalysis"
        filename = f"cams.eaq.vra.ENSa.{species.lower()}.l0.{month}.nc"
        if not (folder / filename).exists():
            # if VRA Reanalysis file does not exist
            # Use Intermediate analysis (IRA) as replacement
            filename = filename.replace("vra", "ira")
        return folder / filename

    @property
    def is_valid(self) -> bool:
        """Returns True if the Sample is valid: if input and target files exist."""
        return all([path.exists() for path in self.input_paths]) and all(
            [path.exists() for path in self.target_paths]
        )

    def load_input_data_for_one_model(self, model: ModelsNames) -> xr.Dataset:
        """Loads data for one pollutant model.

        Args:
            model: The name of the desired pollutant model.

        Returns:
            A xr.Dataset containing all the input data for this model.
        """
        model_path = self.processed_dir / model.lower() / self.input_filename
        data = xr.open_dataset(model_path)
        data = data.sel(level=self.levels, time=self.lead_times)
        data = data.assign_coords(
            time=np.datetime64(self.date_run)
            + data.time.values.astype("timedelta64[h]")
        )
        selected_species = [f"{s.lower()}_conc" for s in self.species]
        data = data[selected_species]
        # Longitude coords range btw 0° and 360°, we put them btw -180° and 180°:
        data = data.assign_coords(longitude=((data.longitude + 180) % 360) - 180)
        data = data.sortby("longitude")
        return data[selected_species]

    def load_target_data(self) -> xr.Dataset:
        """Returns the target analysis data."""
        all_species_da = {}
        months_str = sorted({date.strftime("%Y-%m") for date in self.valid_times})
        for species in self.species:
            species_das = []
            for month in months_str:
                month_times = [
                    time for time in self.valid_times if time.strftime("%Y-%m") == month
                ]
                data = xr.open_dataset(self._target_path(species, month))
                data_of_interest = data.sel(time=month_times)[species.lower()]
                species_das.append(data_of_interest)
            all_species_da[species] = xr.concat(species_das, dim="time")
        target = xr.Dataset(all_species_da)
        target = target.rename(
            {
                "lat": "latitude",
                "lon": "longitude",
            }
        )
        target = target.sortby("latitude", ascending=False)
        target = target.expand_dims(level=[0.0])
        target = target.transpose("time", "level", "latitude", "longitude")
        return target

    @property
    def data(self) -> xr.Dataset:
        """Combination of models and reanalysis data.

        Returns:
            This methods returns a xarray.Dataset with the format:
            <xarray.Dataset> Size: 64MB
            Dimensions:  (latitude: 420, level: 1, time: 3, longitude: 700, species: 6)
            Coordinates:
            * latitude  (latitude) float32 2kB 71.95 71.85 71.75 ... 30.25 30.15 30.05
            * level     (level) float32 4B 0.0
            * time      (time) datetime64[us] 24B 2025-05-10T15:00:00 ... 2025-05-11T1..
                lead_time  (time) int64 24B 15 24 36
            * longitude  (longitude) float32 3kB -24.95 -24.85 -24.75 ... 44.85 44.95
            * species    (species) <U5 120B 'O3' 'CO' 'NO2' 'PM10' 'PM2P5' 'SO2'
            Data variables:
                CHIMERE (species, time, level, latitude, longitude) float32 21MB 66.77..
                MOCAGE  (species, time, level, latitude, longitude) float32 21MB 69.86..
                TARGET  (species, time, level, latitude, longitude) float32 21MB 69.72..
        """
        models = {
            model: self.load_input_data_for_one_model(model) for model in self.models
        }
        reanalysis = self.load_target_data()
        first_model = models[self.models[0]]
        # Align type of coordinates, btw models and reanalysis
        reanalysis = reanalysis.assign_coords(
            latitude=first_model.latitude,
            longitude=first_model.longitude,
            time=first_model.time,
            level=first_model.level,
        )
        models["TARGET"] = reanalysis
        combined = xr.Dataset()
        for model_name, ds in models.items():
            da = ds.to_array(dim="species")
            da = da.assign_coords(  # Format name of species
                species=[s.replace("_conc", "").upper() for s in da.species.values]
            )
            combined[model_name.upper()] = da
        combined.coords["lead_time"] = (("time",), self.lead_times)
        return combined

    @staticmethod
    def convert_data_to_nt(ds: xr.Dataset) -> NamedTensor:
        """Converts xarray dataset to a NamedTensor of shape (features, lat, lon).
        Dimensions other than lat/lon are stacked along the features axis.
        For example:
            --- NamedTensor ---
            Names: ['features', 'lat', 'lon']
            Tensor Shape: torch.Size([12, 420, 700]))
            Features:
            ┌─────────────────────────────┬──────────────┬───────────┐
            │ Feature name                │          Min │       Max │
            ├─────────────────────────────┼──────────────┼───────────┤
            │ CHIMERE - O3 - +15h - 0m    │ 34.9141      │  139.932  │
            │ CHIMERE - CO - +15h - 0m    │ 84.701       │ 1169.37   │
            │ CHIMERE - NO2 - +15h - 0m   │  0.00369518  │   61.7275 │
            │ CHIMERE - PM10 - +15h - 0m  │  0.0479899   │ 1024.85   │
            │ CHIMERE - PM2P5 - +15h - 0m │  0.0419867   │  144.313  │
            │ CHIMERE - SO2 - +15h - 0m   │  1.06274e-16 │  198.59   │
            │ MOCAGE - O3 - +15h - 0m     │ 14.1139      │  165.537  │
            │ MOCAGE - CO - +15h - 0m     │ 53.1523      │ 2596.93   │
            │ MOCAGE - NO2 - +15h - 0m    │  0.00493127  │  543.657  │
            │ MOCAGE - PM10 - +15h - 0m   │  0.0434407   │ 1401.03   │
            │ MOCAGE - PM2P5 - +15h - 0m  │  0.0391834   │ 1363.39   │
            │ MOCAGE - SO2 - +15h - 0m    │  1.6078e-09  │ 2427.39   │
            └─────────────────────────────┴──────────────┴───────────┘
        """
        channel_arrays = []
        channel_names = []

        model_names = list(ds.data_vars)
        for model in model_names:
            da = ds[model]
            da = da.transpose("species", "time", "level", "latitude", "longitude")
            species_values = (
                da.coords["species"].values if "species" in da.coords else [None]
            )
            time_values = da.coords["time"].values if "time" in da.coords else [None]
            level_values = da.coords["level"].values if "level" in da.coords else [None]

            for i_species, species in enumerate(species_values):
                for i_time in range(len(time_values)):
                    for i_level, level in enumerate(level_values):
                        arr = da.isel(
                            species=i_species, time=i_time, level=i_level
                        ).values  # extract 2D channel
                        arr = np.nan_to_num(arr, nan=0.0)
                        channel_arrays.append(arr)
                        leadtime = (
                            da.coords["lead_time"].values[i_time]
                            if "lead_time" in da.coords
                            else None
                        )
                        channel_names.append(
                            Sample._channel_name(model, species, level, leadtime)
                        )

        tensor = torch.tensor(np.stack(channel_arrays, axis=0)).to(torch.float32)
        nt = NamedTensor(tensor, ["features", "lat", "lon"], channel_names)
        return nt

    @staticmethod
    def _channel_name(
        model: Hashable,
        species: str | None = None,
        level: str | None = None,
        leadtime: str | None = None,
    ) -> str:
        """Builds a channel name from whichever coordinates are present.

        Args:
            model: Name of the model data variable.
            species: Species value, or None when the coordinate is absent.
            level: Level value, or None when the coordinate is absent.
            leadtime: Lead time value, or None when the coordinate is absent.

        Returns:
            The formatted channel name.
        """
        name = str(model)
        if species is not None:
            name += f" - {species}"
        if leadtime is not None:
            name += f" - +{leadtime}h"
        if isinstance(level, (int, float, np.integer, np.floating)):
            name += f" - {int(level)}m"
        return name

    def get_input_and_target(self) -> tuple[NamedTensor, NamedTensor]:
        """Returns inputs and target as NamedTensor"""
        ds = self.data
        x = Sample.convert_data_to_nt(ds.drop_vars("TARGET"))
        y = Sample.convert_data_to_nt(ds[["TARGET"]])
        return x, y


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a Sample

    sample = Sample(
        dt.datetime(2025, 5, 10),
        lead_times=[15, 24, 36],
        species=["O3", "CO", "NO2", "PM10", "PM2P5", "SO2"],
        levels=[0],
        models=["CHIMERE", "MOCAGE"],
    )
    print(sample)

    print("Sample is valid ? ->", sample.is_valid)
    for input_path in sample.input_paths:
        print(input_path, input_path.exists())
    for target_path in sample.target_paths:
        print(target_path, target_path.exists())

    print(sample.data)
    x, y = sample.get_input_and_target()
    print(x)
    print(y)
