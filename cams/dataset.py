import datetime as dt
from functools import cached_property
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import Dataset
from typing_extensions import override

from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR


def get_run_dates(processed_dir: Path) -> list[dt.datetime]:
    """Retrieves the dates of all the runs available in a directory."""
    return [
        dt.datetime.strptime(path.stem, "%Y_%m_%d")
        for path in sorted(list(processed_dir.glob("input/*.netcdf")))
    ]


class CAMSDataset(Dataset):
    """CAMS dataset, see [dataset doc](docs/data.md) for complete description."""

    def __init__(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        """Loads the dataset's sample points for the given split.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            start_date: The first date of this dataset.
            end_date: The last date of this dataset.
            processed_dir: Path to the CAMS dataset's processed data.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.processed_dir = processed_dir

    @cached_property
    def samples(self) -> list[Sample]:
        """Returns the list of valid samples in the dataset."""
        run_dates = get_run_dates(self.processed_dir)
        run_dates = [
            date
            for date in run_dates
            if date >= self.start_date
            if date <= self.end_date
        ]
        # For now, we only use the leadtime = 15h:
        samples = [Sample(date_run, 15, self.processed_dir) for date_run in run_dates]
        return [sample for sample in samples if sample.is_valid]

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, idx: int) -> tuple[NamedTensor, NamedTensor]:
        """Returns one sample of training data."""
        sample = self.samples[idx]
        return sample.input_data, sample.target_data


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a CAMSDataset

    start_date, end_date = dt.datetime(2022, 1, 1), dt.datetime(2022, 3, 19)
    dataset = CAMSDataset(start_date, end_date)
    print("Len dataset : ", len(dataset))

    sample = dataset.samples[10]
    print(sample)

    x, y = dataset[10]
    print(x, y)
