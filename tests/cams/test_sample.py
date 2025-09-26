import datetime as dt

import pytest

from cams.sample import Sample


def test_Sample() -> None:
    with pytest.raises(NotImplementedError):
        sample = Sample(date=dt.datetime(year=2024, month=11, day=22), lead_time=9)
