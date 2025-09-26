import pytest

from cams.dataset import CamsDataset


def test_CamsDataset() -> None:
    with pytest.raises(NotImplementedError):
        dataset = CamsDataset()
