import pytest

from cams.plmodule import CAMSLightningModule


def test_CAMSLightningModule() -> None:
    with pytest.raises(NotImplementedError):
        plmodule = CAMSLightningModule()
