import pytest

from cams.datamodule import CamsDataModule


def test_CamsDataModule() -> None:
    with pytest.raises(NotImplementedError):
        datamodule = CamsDataModule()
