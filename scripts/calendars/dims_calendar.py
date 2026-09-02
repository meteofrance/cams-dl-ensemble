from collections import defaultdict
from collections.abc import Generator
from functools import cache
from pathlib import Path
from typing import Any

from calendardataviz import InspectorABC, start_app, RichString
from calendardataviz.colors import RDYLGN, color_from_pct
import datetime as dt
from typing_extensions import override
from cams.settings import RAW_DATA_DIR
import xarray as xr

RAW_DATA_DIR = Path("/scratch/shared/cams-dl-ensemble/all_from_ads/")
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


class DimsInspector(InspectorABC):
    name = "Dimentions"

    def _paths_for_date(self, date: dt.date) -> Generator[Path]:
        return RAW_DATA_DIR.glob(f"**/{date.strftime(r'%Y_%m_%d')}*.netcdf")

    @cache
    def _dims_coords_for_date(self, date: dt.date) -> tuple[set[str], set[str]]:
        paths = list(self._paths_for_date(date))
        coords: set[str] = set()
        variables: set[str] = set()
        for path in paths:
            data = xr.open_dataset(path)
            coords.add(str(data.coords))
            variables.add(str(data.variables))

        return coords, variables

    def _pct_for_date(self, date: dt.date) -> float:
        coords, variables = self._dims_coords_for_date(date)        

        return (11 - max(len(coords), len(variables))) / 10

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
    def popup_content(self, date: dt.date) -> tuple[str, RichString]:
        """Return the information displayed when a date is selected.

        Args:
            date: Date selected.

        Returns:
            str: The pop-up window title.
            str: The pop-up window content.
        """

        # Define the title
        title = (
            date.strftime(r"%A %d %B %Y")
            + f" {self._pct_for_date(date) * 100:.2f}%"
        )

        # Get coords and variables
        paths = list(self._paths_for_date(date))
        coords: dict[str, set[str]] = defaultdict(set)
        variables: dict[str, set[str]] = defaultdict(set)
        for path in paths:
            data = xr.open_dataset(path)
            coords[str(data.coords)].add(path.parent.name)
            variables[str(data.variables)].add(path.parent.name)

        # Define colors
        bgs = ["#000000", "#373737"]
        diff_color = "#780000"

        content = RichString("")
        split_coord = [coord.split(" ") for coord in coords.keys()]
        coords_headers = list(coords.values())
        for j, coord in enumerate(split_coord):
            content += RichString(", ".join(coords_headers[j]) + "\n")
            for i, word in enumerate(coord):
                color = bgs[j % 2]
                if any(i >= len(c) or c[i] != coord[i] for c in split_coord):
                    color = diff_color
                content += RichString(word, color)
            content += RichString("\n\n")

        content += RichString("\n\n")
        split_variables = [variable.split(" ") for variable in variables.keys()]
        variables_headers = list(variables.values())
        for j, variable in enumerate(split_variables):
            content += RichString(", ".join(variables_headers[j]) + "\n")
            for i, word in enumerate(variable):
                color = bgs[(j + len(coords)) % 2]
                if any(i >= len(v) or v[i] != variable[i] for v in split_variables):
                    color = diff_color
                content += RichString(word, color)
            content += RichString("\n\n")


        return title, content

if __name__ == "__main__":
    start_app(
        inspector_cls=DimsInspector,
        years=[2024, 2025, 2026],
        nb_processes=12,
    )