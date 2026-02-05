import datetime as dt
from settings import CAMS_DATASET
import xarray as xr

from mfai.pytorch.namedtensor import NamedTensor
import torch


class Sample:
    """Cams sample.
    Responsibilities:
    - Load a datapoint from the Cams dataset from a given date and leadtime.
    """

    def __init__(self, date_run: dt.datetime, lead_time: int) -> None:
        """
        Args:
            date_run: The run date of the CTMs from which to load the sample.
            lead_time: Which forecast lead time to load the sample from.
                The lead times step is 3h, from the given date at 00h00 to +96h.
                The accepted values for lead_time are [3, 6, 9, ..., 93, 96]
        """
        self.date_run = date_run
        self.lead_time = lead_time
        self.valid_time = self.date_run + dt.timedelta(hours=self.lead_time)

    def __str__(self) -> str:
        date_run_str = self.date_run.strftime("%Y-%m-%d %H:00")
        return f"Sample(date_run={date_run_str}, lead_time=+{self.lead_time}h)"

    @property
    def input_data(self) -> NamedTensor:
        date_run_str = self.date_run.strftime("%Y_%m_%d_%H")
        data_path = CAMS_DATASET / f"input/{date_run_str}.netcdf"
        data = xr.open_dataset(data_path)
        tensor = torch.Tensor(data.O3.values)
        names = [name.replace("PMACC", "") for name in data.model.values]
        nt = NamedTensor(tensor, ["features", "lat", "lon"], names)
        return nt

    @property
    def target_data(self) -> NamedTensor:
        date_run_str = self.date_run.strftime("%Y_%m_%d_%H")
        data_path = CAMS_DATASET / f"target/{date_run_str}.netcdf"
        data = xr.open_dataset(data_path)
        tensor = torch.Tensor(data.O3.values).unsqueeze(dim=0)
        nt = NamedTensor(tensor, ["features", "lat", "lon"], ["Analysis"])
        return nt


if __name__ == "__main__":
    sample = Sample(dt.datetime(2022, 7, 22, 15), 6)
    print(sample)
    x = sample.input_data
    print(x)
    y = sample.target_data
    print(y)
