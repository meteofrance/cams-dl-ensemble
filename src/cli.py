from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class CamsCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        pass

    def after_instantiate_classes(self) -> None:
        """Called after the LightningModule, Datamodule and Trainer are instantiated,
        and before the start of a LightningCLI command (fit, test, val or predict).
        """
        pass
