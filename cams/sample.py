import datetime as dt
from pathlib import Path

import torch
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
            lead_time: Which forecast leadtime to load the sample from.
                The accepted values for lead_time are [0, 1, ..., 96]
            specie: the specie to load.
            level: the level to load.
            processed_dir: Path to the CAMS processed dataset.
        """
        self.date_run = date_run
        self.models = models
        if len(models) > 1:
            raise NotImplementedError("For now, only use one model.")
        self.lead_times = lead_times
        self.valid_times = [self.date_run + dt.timedelta(hours=lt) for lt in lead_times]
        self.species = species
        if any([s not in ["CO", "NO2", "O3", "PM10", "PM3P5", "SO2"] for s in species]):
            raise NotImplementedError(
                "For now, only use the following species: CO, NO2, O3, PM10, PM3P5, SO2."
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
            f"lead_times=+{self.lead_times}, "
            f"species={self.species}), "
            f"models={self.models}), "
            f"levels={self.levels})"
        )

    @property
    def input_path(self) -> Path:
        """The path to the netcdf of input ensemble data."""
        date_run_str = self.date_run.strftime("%Y_%m_%d")
        filename = f"{date_run_str}-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf"
        return self.processed_dir / self.models[0] / filename

    @property
    def target_path(self) -> Path:
        """The path to the netcdf of target analysis data."""
        date_run_str = self.date_run.strftime("%Y-%m-01")  # TODO : changer quand netcdf
        filename = f"-{date_run_str}-CO_NO2_PM10_PM25_SO2_O3-0m-ira.zip"  # TODO : changer quand netcdf
        return self.processed_dir / "reanalysis" / filename

    @property
    def is_valid(self) -> bool:
        """Returns True if the Sample is valid: if input and target files exist."""
        return self.input_path.exists() and self.target_path.exists()

    @property
    def input_data(self) -> NamedTensor:
        """Returns the input ensemble data as a NamedTensor."""
        data = xr.open_dataset(self.input_path)
        print(data)
        selected_species = [f"{s.lower()}_conc" for s in self.species]
        data_of_interest: xr.DataArray = data[selected_species]
        data_of_interest = data_of_interest.sel(level=self.levels, time=self.lead_times)
        print(data_of_interest)
        tensors = []
        for species in selected_species:
            tensor = torch.Tensor(data_of_interest[species].values)
            tensors.append(tensor)
            print(tensor.shape)
        data_tensor = torch.stack(tensors)
        data_tensor = data_tensor.unsqueeze(0)
        print(data_tensor.shape)
        nt = NamedTensor(
            data_tensor,
            ["features", "species", "leadtimes", "levels", "lat", "lon"],
            self.models,
        )
        return nt

    @property
    def target_data(self) -> NamedTensor:
        """Returns the target analysis data as a NamedTensor."""
        data: xr.DataArray = xr.open_dataarray(self.target_path)
        data_of_interest: xr.DataArray = data.sel(species=self.specie, level=self.level)
        tensor = torch.Tensor(data_of_interest.values)
        tensor = tensor.unsqueeze(dim=0)
        nt = NamedTensor(tensor, ["features", "lat", "lon"], [self.specie])
        return nt


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a Sample

    sample = Sample(
        dt.datetime(2025, 5, 10),
        lead_times=[15, 20],
        species=["O3", "CO", "NO2"],
        levels=[0],
        models=["chimere"],
    )
    print(sample)

    print("Sample is valid ? ->", sample.is_valid)
    print(sample.input_path, sample.input_path.exists())
    print(sample.target_path, sample.target_path.exists())

    x = sample.input_data
    print(x)

# TODO :
# - définir un type pour les modèles, et les autres paramètres
# - fix docker build
