from collections.abc import Generator
from pathlib import Path

from calendardataviz import InspectorABC, start_app, RichString
import datetime as dt
from typing_extensions import override
from cams.settings import RAW_DATA_DIR

RAW_DATA_DIR = Path("/scratch/shared/cams-dl-ensemble/all_from_ads/")
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")

class FileSizeInspector(InspectorABC):
    name = "File size"

    # model * 11 + reanalysis + weighted_ensemble * 6
    _target_size = 684439444 * 11

    # Colors from https://colorbargenerator.com/
    _colors: dict[float, RichString] = {
        0 / 11: RichString("◦", "#000000", "#b7b7b7"),
        1 / 11: RichString(" ", "#A50026", "#ffffff"),
        2 / 11: RichString(" ", "#D22B27", "#ffffff"),
        3 / 11: RichString(" ", "#EE613D", "#000000"),
        4 / 11: RichString(" ", "#FA9B58", "#000000"),
        5 / 11: RichString(" ", "#FECC7A", "#000000"),
        6 / 11: RichString(" ", "#ECE88B", "#000000"),
        7 / 11: RichString(" ", "#C5E67E", "#000000"),
        8 / 11: RichString(" ", "#93D168", "#000000"),
        9 / 11: RichString(" ", "#57B65F", "#000000"),
        10 / 11: RichString(" ", "#17934E", "#ffffff"),
        11 / 11: RichString("1", "#006837", "#ffffff"),
    }


    def _paths_for_date(self, date: dt.date) -> Generator[Path]:
        return RAW_DATA_DIR.glob(f"**/{date.strftime(r'%Y_%m_%d')}*.netcdf")

    def _pct_for_date(self, date: dt.date) -> float:
        total_size = 0
        for path in self._paths_for_date(date):
            total_size += path.stat().st_size

        return total_size / self._target_size

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
        closest_pct = min(self._colors.keys(), key=lambda x: abs(x - pct))
        return self._colors[closest_pct]

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
            closest_pct = min(self._colors.keys(), key=lambda x: abs(x - pct))
            colors.append(self._colors[closest_pct])

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
            date.strftime(r"%A %d %B %Y")
            + f" {self._pct_for_date(date) * 100:.2f}%"
        )
        content = "\n".join(
            f"{path.stat().st_size}B "
            f"{path.relative_to(path.parents[1])}"
            for path in self._paths_for_date(date)
        )

        return title, content

if __name__ == "__main__":
    start_app(
        inspector_cls=FileSizeInspector,
        years=[2024, 2025, 2026],
        nb_processes=12,
    )