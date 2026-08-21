import datetime as dt
from pathlib import Path

import torch
from torch import Tensor
import numpy as np
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor
from typing_extensions import override

from cams.settings import PROCESSED_DATA_DIR


class Sample:
    """CAMS sample.

    Responsibilities:
        Load a sample from the CAMS dataset from a given run date,
        and list of species, levels, models and leadtimes.
    """

    def __init__(
        self,
        date_run: dt.date,
        models: list[str],
        lead_times: list[int],
        species: list[str],
        levels: list[int],
        processed_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        """
        Args:
            date_run: The run date of the CTMs from which to load the sample.
            models: the l models to load.
            lead_times: Which forecast leadtimes to load the sample from.
                The accepted values for one leadtime are [0, 1, ..., 96].
            specie: the species to load.
            level: the levels to load.
            processed_dir: Path to the CAMS processed dataset.
        """
        self.date_run = date_run
        self.models = models
        self.lead_times = lead_times
        self.valid_times = [self.date_run + dt.timedelta(hours=lt) for lt in lead_times]
        self.species = species
        if any([s not in ["CO", "NO2", "O3", "PM10", "PM2P5", "SO2"] for s in species]):
            raise NotImplementedError(
                "For now, only use the following species: CO, NO2, O3, PM10, PM2P5, SO2."
            )
        self.levels = levels
        if levels != [0]:
            raise NotImplementedError("For now, only use ground level.")
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
            self.processed_dir / model / self.input_filename for model in self.models
        ]

    @property
    def target_paths(self) -> list[Path]:
        """The paths to the netcdf of targets reanalysis data.
        Files are grouped by months and species."""
        date_run_str = self.date_run.strftime("%Y-%m")
        folder = self.processed_dir / "reanalysis"
        paths = []
        for species in self.species:
            filename = f"cams.eaq.vra.ENSa.{species.lower()}.l0.{date_run_str}.nc"
            if not (folder / filename).exists():
                # if VRA Reanalysis file does not exist
                # Use Intermediate analysis (IRA) as replacement
                filename = filename.replace("vra", "ira")
            paths.append(folder / filename)
        return paths

    @property
    def is_valid(self) -> bool:
        """Returns True if the Sample is valid: if input and target files exist."""
        return all([path.exists() for path in self.input_paths]) and all(
            [path.exists() for path in self.target_paths]
        )

    def load_input_data_for_one_model(self, model: str) -> xr.Dataset:
        """Loads data for one pollutant model."""
        model_path = self.processed_dir / model / self.input_filename
        data = xr.open_dataset(model_path)
        data = data.sel(level=self.levels, time=self.lead_times)
        data = data.assign_coords(
            time=np.datetime64(self.date_run)
            + data.time.values.astype("timedelta64[h]")
        )
        selected_species = [f"{s.lower()}_conc" for s in self.species]
        data = data[selected_species]
        data = data.assign_coords(longitude=((data.longitude + 180) % 360) - 180)
        data = data.sortby("longitude")
        return data[selected_species]

    def load_target_data(self) -> xr.Dataset:
        """Returns the target analysis data."""
        all_species_da = {}
        for i, path in enumerate(self.target_paths):
            data = xr.open_dataset(path)
            data_of_interest = data.sel(time=self.valid_times)
            data_of_interest = data_of_interest[self.species[i].lower()]
            all_species_da[self.species[i]] = data_of_interest
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
        """Combination of models and reanalysis data."""
        models = {
            model: self.load_input_data_for_one_model(model) for model in self.models
        }
        rean = self.load_target_data()
        first_model = models[self.models[0]]
        # Align type of coordinates, btw models and reanalysis
        rean = rean.assign_coords(
            latitude=first_model.latitude,
            longitude=first_model.longitude,
            time=first_model.time,
            level=first_model.level,
        )
        models["target"] = rean
        combined = xr.Dataset()
        for model_name, ds in models.items():
            da = ds.to_array(dim="species")
            da = da.assign_coords(  # Format name of species
                species=[s.replace("_conc", "").upper() for s in da.species.values]
            )
            combined[model_name] = da
        return combined

    @property
    def data_as_nt(self) -> NamedTensor:
        """Converts the data to a NamedTensor of shape (features, latitude, longitude)."""
        ds = self.data
        channel_arrays = []
        channel_names = []

        model_names = list(ds.data_vars)
        for model in model_names:
            da = ds[model]
            da = da.transpose("species", "time", "level", "latitude", "longitude")
            species_values = da.coords["species"].values
            time_values = da.coords["time"].values
            level_values = da.coords["level"].values

            for i_species, species in enumerate(species_values):
                for i_time, time in enumerate(time_values):
                    for i_level, level in enumerate(level_values):
                        arr = da.isel(
                            species=i_species, time=i_time, level=i_level
                        ).values  # extract 2D channel
                        arr = np.nan_to_num(arr, nan=0.0)
                        channel_arrays.append(arr)

                        if np.issubdtype(type(time), np.datetime64):
                            time_str = np.datetime_as_string(time, unit="h")
                        else:
                            time_str = str(time)

                        channel_name = (
                            f"{model} - {species} - {time_str} - {float(level)}"
                        )
                        channel_names.append(channel_name)

        tensor = torch.tensor(np.stack(channel_arrays, axis=0))
        nt = NamedTensor(tensor, ["features", "lat", "lon"], channel_names)
        return nt


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a Sample

    sample = Sample(
        dt.datetime(2025, 5, 10),
        lead_times=[15, 24, 36],
        species=["O3", "CO", "NO2", "PM10", "PM2P5", "SO2"],
        levels=[0],
        models=["chimere", "mocage"],
    )
    print(sample)

    print("Sample is valid ? ->", sample.is_valid)
    for input_path in sample.input_paths:
        print(input_path, input_path.exists())
    for target_path in sample.target_paths:
        print(target_path, target_path.exists())

    print(sample.data)
    print(sample.data_as_nt)

# TODO :
# - définir un type pour les modèles, et les autres paramètres
# - fix docker build
# - compute stats for all species
# - répercuter sur dataset et datamodule
# - vérifier que toute la pipeline fonctionne
