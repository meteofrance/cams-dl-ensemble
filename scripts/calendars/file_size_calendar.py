import datetime as dt
from collections.abc import Generator
from pathlib import Path

from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.colors import RDYLGN, color_from_pct
from typing_extensions import override

from cams.settings import RAW_DATA_DIR

RAW_DATA_DIR = Path("/scratch/shared/cams-dl-ensemble/all_from_ads/")
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


class FileSizeInspector(InspectorABC):
    name = "File size"

    # model * 11 + reanalysis + weighted_ensemble * 6
    _target_size_one_file = 684439444

    def _paths_for_date(self, date: dt.date) -> Generator[Path]:
        return RAW_DATA_DIR.glob(f"**/{date.strftime(r'%Y_%m_%d')}*.netcdf")

    def _pct_for_date(self, date: dt.date) -> float:
        paths = list(self._paths_for_date(date))
        total = 0
        for path in paths:
            total += 1 if path.stat().st_size > self._target_size_one_file * 0.90 else 0

        if len(paths) > 0:
            return total / len(paths)
        else:
            return 0

    @override
    def color_for_date(self, date: dt.date) -> RichString:
        """Returns the color for a given date.

        Args:
            date: date.

        Returns:
            RichString: The text and color associated
                to the given date.
        """

        pct = self._pct_for_date(date)
        if pct < 0:
            return UNDER_0_COLOR
        if pct > 1:
            return OVER_1_COLOR
        return color_from_pct(pct, RDYLGN)

    @override
    def as_color_bar(self, size: int) -> list[RichString]:
        """Returns values for a color bar of the given size.

        Args:
            size: Size of the colorbar to generate.

        Returns:
            list[RichString]: A list of length "size"
                containing one character TTkStrings, one
                for each cell of the color bar.
        """
        # Assigns the closest available color to each percentage
        # displayed in the color bar
        colors: list[RichString] = []
        for pct in [y / (size - 1) for y in range(size)]:
            colors.append(color_from_pct(pct, RDYLGN))

        return colors

    @override
    def popup_content(self, date: dt.date) -> tuple[str, str]:
        """Return the information displayed when a date is selected.

        Args:
            date: Date selected.

        Returns:
            str: The pop-up window title.
            str: The pop-up window content.
        """
        title = (
            date.strftime(r"%A %d %B %Y") + f" {self._pct_for_date(date) * 100:.2f}%"
        )
        content = "\n".join(
            f"{path.stat().st_size}B {path.relative_to(path.parents[1])}"
            for path in self._paths_for_date(date)
        )

        return title, content


if __name__ == "__main__":
    start_app(
        inspector_cls=FileSizeInspector,
        years=[2024, 2025, 2026],
        nb_processes=12,
    )
