from src.cli import CamsCli
from src.datamodule import CamsDataModule
from src.plmodule import CamsLightningModule


def cli_main() -> None:
    cli = CamsCli(
        CamsLightningModule,
        CamsDataModule,
        run=False,
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
