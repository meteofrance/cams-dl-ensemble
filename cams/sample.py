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
    - Load a datapoint from the CAMS dataset from a given date and leadtime.
    """

    def __init__(
        self,
        date_run: dt.datetime,
        lead_time: int,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        """
        Args:
            date_run: The run date of the CTMs from which to load the sample.
            lead_time: Which forecast lead time to load the sample from.
                The lead times step is 3h, from the given date at 00h00 to +96h.
                The accepted values for lead_time are [3, 6, 9, ..., 93, 96]
            processed_dir: Path to the CAMS processed dataset.
        """
        self.date_run = date_run
        self.lead_time = lead_time
        self.valid_time = self.date_run + dt.timedelta(hours=self.lead_time)
        self.processed_dir = processed_dir

    @override
    def __str__(self) -> str:
        date_run_str = self.date_run.strftime("%Y-%m-%d %H:%M")
        return f"Sample(date_run={date_run_str}, lead_time=+{self.lead_time}h)"

    @property
    def input_path(self) -> Path:
        """The path to the netcdf of input ensemble data."""
        date_run_str = self.date_run.strftime("%Y_%m_%d")
        return self.processed_dir / f"input/{date_run_str}.netcdf"

    @property
    def target_path(self) -> Path:
        """The path to the netcdf of target analysis data."""
        valid_time_str = self.valid_time.strftime("%Y_%m_%d_%H")
        return self.processed_dir / f"target/{valid_time_str}.netcdf"

    @property
    def is_valid(self) -> bool:
        """Returns True if the Sample is valid: if input and target files exist."""
        return self.input_path.exists() and self.target_path.exists()

    @property
    def input_data(self) -> NamedTensor:
        """Returns the input ensemble data as a NamedTensor."""
        data = xr.open_dataarray(self.input_path)
        tensor = torch.Tensor(data.to_numpy())
        # For now, we work with all models, the first species, level, and leadtime:
        tensor = tensor[:, 0, 0, 0]
        names = [name.replace("PMACC", "") for name in data.model.values]
        nt = NamedTensor(tensor, ["features", "lat", "lon"], names)
        return nt

    @property
    def target_data(self) -> NamedTensor:
        """Returns the target analysis data as a NamedTensor."""
        data = xr.open_dataarray(self.target_path)
        tensor = torch.Tensor(data.to_numpy())
        tensor = tensor[0, 0]  # For now, select the first species and level
        tensor = tensor.unsqueeze(dim=0)  # Add feature dimension
        nt = NamedTensor(tensor, ["features", "lat", "lon"], ["Analysis"])
        return nt


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a Sample

    sample = Sample(dt.datetime(2022, 7, 22), 15)
    print(sample)
    print("Sample is valid ? ->", sample.is_valid)
    print(sample.input_path)
    x, y = sample.input_data, sample.target_data
    print(x, y)
