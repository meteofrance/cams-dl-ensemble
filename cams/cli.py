"""Lightning CLI for CAMS training."""

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from typing_extensions import override


class CAMSLightningCLI(LightningCLI):
    """Lightning CLI wiring the datamodule arguments to the lightning module.

    The data selection arguments (models, lead_times, species, levels) are
    defined on the datamodule and mapped onto the lightning module after
    instantiation, so they only have to be provided once on the command line.
    """

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Copy the data selection arguments from the datamodule to the module."""
        parser.link_arguments(
            source="data.lead_times",
            target="model.lead_times",
        )
        parser.link_arguments(
            source="data.species",
            target="model.species",
        )
        parser.link_arguments(
            source="data.levels",
            target="model.levels",
        )
