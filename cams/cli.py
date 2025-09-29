from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from typing_extensions import override


class CamsCli(LightningCLI):
    """Cli interface used to interact with the cams-dl-ensemble project."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add extra arguments to the parser or link arguments.

        Args:
            parser: Ther parser object to which arguments can be added.
        """
        pass

    @override
    def after_instantiate_classes(self) -> None:
        """Called after the LightningModule, Datamodule and Trainer are instantiated,
        and before the start of a LightningCLI command (fit, test, val or predict).
        """
        pass
