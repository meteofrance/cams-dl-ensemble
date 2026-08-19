import datetime as dt
from pathlib import Path

import torch
from torch import Tensor
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
    def target_path(self) -> Path:
        """The path to the netcdf of target analysis data."""
        date_run_str = self.date_run.strftime("%Y-%m-01")  # TODO : changer quand netcdf
        filename = f"-{date_run_str}-CO_NO2_PM10_PM25_SO2_O3-0m-ira.zip"  # TODO : changer quand netcdf
        return self.processed_dir / "reanalysis" / filename

    @property
    def is_valid(self) -> bool:
        """Returns True if the Sample is valid: if input and target files exist."""
        return (
            all([path.exists() for path in self.input_paths])
            and self.target_path.exists()
        )

    def load_tensor_for_one_model(self, model: str) -> Tensor:
        """Loads data tensor for one pollutant model."""
        model_path = self.processed_dir / model / self.input_filename
        data = xr.open_dataset(model_path)
        data_of_interest = data.sel(level=self.levels, time=self.lead_times)
        selected_species = [f"{s.lower()}_conc" for s in self.species]
        tensors = [
            torch.Tensor(data_of_interest[species].values)
            for species in selected_species
        ]
        return torch.stack(tensors)

    @property
    def input_data(self) -> NamedTensor:
        """Returns the input ensemble data as a NamedTensor."""
        tensors = [self.load_tensor_for_one_model(model) for model in self.models]
        data = torch.stack(tensors)
        nt = NamedTensor(
            data,
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
        lead_times=[15, 24, 36, 48],
        species=["O3", "CO", "NO2"],
        levels=[0],
        models=["chimere", "mocage"],
    )
    print(sample)

    print("Sample is valid ? ->", sample.is_valid)
    for input_path in sample.input_paths:
        print(input_path, input_path.exists())
    print(sample.target_path, sample.target_path.exists())

    x = sample.input_data
    print(x)

# TODO :
# - définir un type pour les modèles, et les autres paramètres
# - fix docker build
# - zip target data
# - charger target data
# - inventer des plots pour représenter tout ça lol
# - répercuter sur dataset et datamodule
# - vérifier que toute la pipeline fonctionne
