import datetime as dt
from pathlib import Path

import torch
from typing_extensions import override

from cams.datamodule import CAMSDataModule
from cams.dataset import CAMSDataset
from cams.sample import NamedTensor, Sample


class SampleTest(Sample):
    @property
    def is_valid(self) -> bool:
        """Always returns True, because we use fake data."""
        return True

    @property
    def input_data(self) -> NamedTensor:
        """Returns fake input ensemble data as a NamedTensor."""
        num_models = 11
        tensor = torch.zeros((num_models, 128, 128))
        names = [str(i) for i in range(num_models)]
        nt = NamedTensor(tensor, ["features", "lat", "lon"], names)
        return nt

    @property
    def target_data(self) -> NamedTensor:
        """Returns fake target analysis data as a NamedTensor."""
        tensor = torch.zeros((1, 128, 128))
        nt = NamedTensor(tensor, ["features", "lat", "lon"], ["Analysis"])
        return nt


class CAMSDatasetTest(CAMSDataset):
    @override
    @property
    def run_dates(self):
        """Returns a fake list of available run dates for CAMS models"""
        run_dates = [dt.datetime(2000, 1, i) for i in range(1, 32)]
        return run_dates

    @override
    def create_sample(
        self, date_run: dt.datetime, lead_time: int, path: Path
    ) -> Sample:
        return SampleTest(date_run, lead_time, path)


class CAMSDataModuleTest(CAMSDataModule):
    @override
    @property
    def run_dates(self):
        """Returns a fake list of available run dates for CAMS models"""
        return [dt.datetime(2000, 1, i) for i in range(1, 32)]

    @override
    def create_dataset(
        self, start: dt.datetime, end: dt.datetime, dir: Path
    ) -> CAMSDataset:
        return CAMSDatasetTest(start, end, dir)
