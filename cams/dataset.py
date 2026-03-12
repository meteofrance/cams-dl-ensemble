import datetime as dt
from functools import cached_property
from pathlib import Path

from mfai.pytorch.namedtensor import NamedTensor
from torch import nn
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
        start_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
        processed_dir: Path = PROCESSED_DATA_DIR,
        transform_sequence: nn.Sequential = nn.Sequential(*[]),
    ) -> None:
        """Loads the dataset's sample points for the given split.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            start_date: The first date of this dataset, inclusive.
                Defaults to the first date in the processed directory.
            end_date: The last date of this dataset, inclusive.
                Defaults to the last date in the processed directory.
            processed_dir: Path to the CAMS dataset's processed data.
            transform_sequence: transforms sequence applied to the data after loading.
        """
        self.processed_dir = processed_dir
        self.transform_sequence = transform_sequence

        # Define start and end dates if not given
        availabe_dates = get_run_dates(processed_dir)
        self.start_date = start_date if start_date else availabe_dates[0]
        self.end_date = end_date if end_date else availabe_dates[-1]

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
        return self.transform_sequence(
            (self.samples[idx].input_data, self.samples[idx].target_data)
        )


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a CAMSDataset

    start_date, end_date = dt.datetime(2022, 1, 1), dt.datetime(2022, 3, 19)
    dataset = CAMSDataset(start_date, end_date)
    print("Len dataset : ", len(dataset))

    sample = dataset.samples[10]
    print(sample)

    x, y = dataset[10]
    print(x, y)
