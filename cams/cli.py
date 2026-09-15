"""Lightning CLI for CAMS training."""

from lightning.pytorch.cli import LightningCLI
from typing_extensions import override


class CAMSLightningCLI(LightningCLI):
    """Lightning CLI wiring the datamodule arguments to the lightning module.

    The data selection arguments (models, lead_times, species, levels) are
    defined on the datamodule and mapped onto the lightning module after
    instantiation, so they only have to be provided once on the command line.
    """

    @override
    def after_instantiate_classes(self) -> None:
        """Copy the data selection arguments from the datamodule to the module."""
        super().after_instantiate_classes()
        self.model.models = self.datamodule.models
        self.model.lead_times = self.datamodule.lead_times
        self.model.species = self.datamodule.species
        self.model.levels = self.datamodule.levels
