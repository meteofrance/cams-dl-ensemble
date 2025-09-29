import datetime as dt


class Sample:
    """Cams sample.
    Responsibilities:
    - Load a datapoint from the Cams dataset from a date and a leadtime.
    """

    def __init__(self, date: dt.datetime, lead_time: int) -> None:
        """
        Args:
            date: The date from wich to load the sample.
            lead_time: Wich forecast lead time to load the sample from.
                The lead times step is 3h, from the given date at 00h00 to +96h.
                The accepted values for lead_time are [3, 6, 9, ..., 93, 96]
        """
        raise NotImplementedError
