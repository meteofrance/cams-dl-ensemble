import pytest

from cams.sample import Sample


def test_Sample() -> None:
    with pytest.raises(NotImplementedError):
        sample = Sample()
