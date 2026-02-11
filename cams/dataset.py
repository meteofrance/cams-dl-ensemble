import datetime as dt
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import Dataset
from typing_extensions import override

from cams.sample import Sample
from cams.settings import CAMS_DATASET_DIR


class CamsDataset(Dataset):
    """Cams dataset, see [dataset doc](docs/data.md) for complete description."""

    def __init__(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        data_dir: Path = CAMS_DATASET_DIR,
    ) -> None:
        """Loads the dataset's sample points for the given split.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            start_date: The first date of this dataset.
            end_date: The last date of this dataset.
            data_dir: Path to the CAMS dataset.
        """
        # Gather run dates
        run_dates: list[dt.datetime] = [
            dt.datetime.strptime(path.stem, "%Y_%m_%d")
            for path in sorted(list(data_dir.glob("input/*.netcdf")))
        ]
        run_dates = [
            date
            for date in run_dates
            if date >= start_date
            if date < end_date
        ]
        # For now, we only use the leadtime = 15h:
        list_samples = [Sample(date_run, 15, data_dir) for date_run in list_dates]
        self.samples = [sample for sample in list_samples if sample.is_valid]

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, idx: int) -> tuple[NamedTensor, NamedTensor]:
        """Returns one sample of training data."""
        sample = self.samples[idx]
        return sample.input_data, sample.target_data


if __name__ == "__main__":
    start_date, end_date = dt.datetime(2022, 1, 1), dt.datetime(2022, 3, 19)
    dataset = CamsDataset(start_date, end_date)
    print("Len dataset : ", len(dataset))

    sample = dataset.samples[10]
    print(sample)

    x, y = dataset[10]
    print(x, y)
