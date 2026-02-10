import datetime as dt

from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import Dataset
from typing_extensions import override

from cams.sample import Sample
from cams.settings import CAMS_DATASET_DIR


class CamsDataset(Dataset):
    """Cams dataset, see [dataset doc](docs/data.md) for complete description."""

    def __init__(self, start_date: dt.datetime, end_date: dt.datetime) -> None:
        """Loads the dataset's sample points for the given split.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            start_date (dt.datetime): the first date of this dataset.
            end_date (dt.datetime): the last date of this dataset.
        """
        list_runs = sorted(list(CAMS_DATASET_DIR.glob("input/*.netcdf")))
        list_dates = [dt.datetime.strptime(path.stem, "%Y_%m_%d") for path in list_runs]
        list_dates = [date for date in list_dates if date >= start_date]
        list_dates = [date for date in list_dates if date < end_date]
        list_samples = [Sample(date_run, lead_time=15) for date_run in list_dates]
        self.samples = [sample for sample in list_samples if sample.is_valid]

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, idx: int) -> tuple[NamedTensor, NamedTensor]:
        """Returns one sample of training data."""
        sample = self.samples[idx]
        return sample.input_data, sample.target_data


if __name__ == "__main__":
    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime(2022, 3, 19)
    dataset = CamsDataset(start_date, end_date)
    print("Len dataset : ", len(dataset))

    sample = dataset.samples[10]
    print(sample)

    x, y = dataset[10]
    print(x)
    print(y)
