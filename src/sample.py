import datetime as dt

from mfai.pytorch.namedtensor import NamedTensor


class Sample:
    def __init__(self, date: dt.datetime, lead_time: int) -> None:
        """
        Args:
            date: The date from wich to load the sample.
            lead_time: Wich forecast lead time to load the sample from.
                The lead times step is 3h, from the given date at 00h00 to +96h.
                The accepted values for lead_time are [3, 6, 9, ..., 93, 96]
        """
        raise NotImplementedError

    def is_valid(self) -> bool:
        """Wether this sample's data is complete."""
        raise NotImplementedError

    @property
    def input(self) -> NamedTensor:
        """The CAMS DL-ENSEMBLE model input."""
        raise NotImplementedError

    @property
    def target(self) -> NamedTensor:
        """The CAMS DL-ENSEMBLE model target."""
        raise NotImplementedError
