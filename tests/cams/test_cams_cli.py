import pytest

from cams.cli import CAMSCli
from cams.datamodule import CAMSDataModule
from cams.plmodule import CAMSLightningModule


def test_CAMSCli() -> None:
    with pytest.raises(Exception):
        CAMSCli(
            model_class=CAMSLightningModule,
            datamodule_class=CAMSDataModule,
            run=False,
            args=[],
        )
