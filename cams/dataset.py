import datetime as dt
from typing import Literal

import torch
from mfai.pytorch.namedtensor import NamedTensor
from typing_extensions import override

from cams.sample import Sample


class CamsDataset(torch.utils.data.Dataset[tuple[NamedTensor, NamedTensor]]):
    """Cams dataset, see [dataset doc](docs/data.md) for complete description."""

    def __init__(self, stage: Literal["train", "test", "val"]) -> None:
        """Loads the dataset's sample points for the given stage.
        A sample point is a date and a forecast id, used to instantiate a Sample.

        Args:
            stage: Wich split of the dataset to load.
        """
        self.sample_points: list[tuple[dt.datetime, int]] = []
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.sample_points)

    @override
    def __getitem__(self, idx: int) -> tuple[NamedTensor, NamedTensor]:
        sample = Sample(*self.sample_points[idx])
        return sample.input, sample.target
