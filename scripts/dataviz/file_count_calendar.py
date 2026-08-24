import datetime as dt
from pathlib import Path
from typing import override

from calendardataviz import InspectorABC, start_app, RichString
from cams.settings import RAW_DATA_DIR

RAW_DATA_DIR = Path("/scratch/shared/cams-dl-ensemble/all_from_ads/")
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")




class FileCountInspector(InspectorABC):
    """Implementation of the inspector class for the CAMS dataset."""

    name = "Target file count"
    root_dir = Path("/scratch/shared/cams-dl-ensemble/all_from_ads")
    target_nb_files_total = 3 * 365 * 11
    target_nb_files_per_day = 11

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
        11 / 11: RichString(" ", "#006837", "#ffffff"),
    }

    def _files_for_date(self, date: dt.date) -> tuple[Path, ...]:
        # Finds files named like:
        # `/lotos/2024_07_14-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.zip`
        date_str = date.strftime(r"%Y_%m_%d")
        return tuple(self.root_dir.glob(f"**/{date_str}*.netcdf"))

    @override
    def color_for_date(self, date: dt.date) -> RichString:
        """Returns the color for a given date.

        Args:
            date: date.

        Returns:
            RichString: The text and color associated
                to the given date.
        """
        pct = len(self._files_for_date(date)) / self.target_nb_files_per_day
        if pct in self._colors:
            return self._colors[pct]

        return RichString("?", "#ff00ff")

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
        files = sorted(self._files_for_date(date))
        content = "\n".join(
            f"{file.parent.stem:>7} {file.stem.replace('_', ' ')}" for file in files
        )
        title = (
            f"{date.strftime(r'%Y_%m_%d')} "
            f"({len(files)}/{self.target_nb_files_per_day})"
        )

        return title, content


if __name__ == "__main__":
    start_app(
        inspector_cls=FileCountInspector,
        years=[2024, 2025, 2026],
        nb_processes=8,
    )
