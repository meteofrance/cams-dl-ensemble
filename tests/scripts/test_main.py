import pytest

from scripts.main import cli_main


def test_cli_main() -> None:
    with pytest.raises(NotImplementedError):
        cli_main()
