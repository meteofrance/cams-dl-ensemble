import pytest

from cams.plmodule import CamsLightningModule


def test_CamsLightningModule() -> None:
    with pytest.raises(NotImplementedError):
        plmodule = CamsLightningModule()
