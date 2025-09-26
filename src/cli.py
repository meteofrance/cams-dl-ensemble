from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class CamsCli(LightningCLI):
    """Cli interface used to interact with the cams-dl-ensemble project."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add extra arguments to the parser or link arguments.

        Args:
            parser: Ther parser object to which arguments can be added.
        """
        pass

    def after_instantiate_classes(self) -> None:
        """Called after the LightningModule, Datamodule and Trainer are instantiated,
        and before the start of a LightningCLI command (fit, test, val or predict).
        """
        pass
