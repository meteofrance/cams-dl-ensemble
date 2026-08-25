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
    print("--> Retrieving run dates...")
    all_files = sorted(list(set(processed_dir.glob("**/*.netcdf"))))
    run_dates, files_not_parsed = [], []
    for file in all_files:
        try:
            date = dt.datetime.strptime(file.stem.split("-")[0], r"%Y_%m_%d")
            run_dates.append(date)
        except:
            files_not_parsed.append(file)
            continue
    if len(files_not_parsed) > 0:
        print(
            f"WARNING: {len(files_not_parsed)} files could not be parsed: {files_not_parsed}"
        )
    run_dates = list(set(run_dates))  # remove duplicates
    return run_dates


class CAMSDataset(Dataset):
    """CAMS dataset, see [dataset doc](docs/data.md) for complete description."""

    def __init__(
        self,
        run_dates: list[dt.datetime],
        models: list[str],
        lead_times: list[int] = [15],
        species: list[str] = ["O3"],
        levels: list[int] = [0],
        processed_dir: Path = PROCESSED_DATA_DIR,
        transform_sequence: nn.Sequential = nn.Sequential(*[]),
    ) -> None:
        """Loads the dataset's sample points for the given split.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            run_dates: The list of date to process.
            models: the models to load in the dataset.
            lead_times: the lead_times to load in the dataset.
            species: the species to load in the dataset.
            levels: the levels to load in the dataset.
                '0' corresponds to 'ground' level.
            processed_dir: Path to the CAMS dataset's processed data.
            transform_sequence: transforms sequence applied to the data after loading.
        """
        self.run_dates = run_dates
        self.models = models
        self.lead_times = lead_times
        self.species = species
        self.levels = levels
        self.processed_dir = processed_dir
        self.transform_sequence = transform_sequence

    @cached_property
    def samples(self) -> list[Sample]:
        """Returns the list of valid samples in the dataset."""
        # For now, we only use the leadtime = 15h:
        samples = [
            Sample(
                date_run=date_run,
                models=self.models,
                lead_times=self.lead_times,
                species=self.species,
                levels=self.levels,
                processed_dir=self.processed_dir,
            )
            for date_run in self.run_dates
        ]
        return [sample for sample in samples if sample.is_valid]

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, idx: int) -> tuple[NamedTensor, NamedTensor]:
        """Returns one sample of training data."""
        x, y = self.samples[idx].get_input_and_target()
        return self.transform_sequence((x, y))


if __name__ == "__main__":
    # This is a simple example of how to instanciate and use a CAMSDataset

    run_dates: list[dt.datetime] = get_run_dates(PROCESSED_DATA_DIR)
    print(len(run_dates))
    dataset = CAMSDataset(
        run_dates,
        models=["chimere", "mocage"],
        lead_times=[15, 24],
        species=["O3", "NO2"],
        levels=[0],
    )
    print("Len dataset : ", len(dataset))

    sample = dataset.samples[10]
    print(sample)

    x, y = dataset[10]
    print(x, y)
