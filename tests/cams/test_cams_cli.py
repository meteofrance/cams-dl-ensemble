from cams.cli import CamsCli
from cams.datamodule import CamsDataModule
from cams.plmodule import CamsLightningModule


def test_CamsCli() -> None:
    with pytest.raises(NotImplementedError):
        CamsCli(
            model_class=CamsLightningModule,
            datamodule_class=CamsDataModule,
            run=False,
        )
