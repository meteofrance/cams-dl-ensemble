import datetime as dt
from collections import defaultdict
from collections.abc import Generator
from functools import cache
from pathlib import Path

import xarray as xr
from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.colors import RDYLGN, color_from_pct
from typing_extensions import override

from cams.settings import RAW_DATA_DIR

# Colors for out‑of‑range percentages
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


class DimsInspector(InspectorABC):
    """Inspector showing how consistent the file dimensions are per date.

    For a given date it compares the coordinate and variable signatures across
    all NetCDF files and derives a quality percentage from how many distinct
    signatures are found.
    """

    name = "Dimensions"

    def _paths_for_date(self, date: dt.date) -> Generator[Path, None, None]:
        """Returns the files associated with the given date.

        Args:
            date: Date for wich to return paths.
        
        Yields:
            Path: Paths to the different files associated to the given date.
        """
        pattern = f"**/{date.strftime('%Y_%m_%d')}*.netcdf"
        yield from RAW_DATA_DIR.rglob(pattern)

    @cache
    def _dims_info_for_date(
        self, date: dt.date
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Return mappings of coordinate/variable string representation to
        model names. Cached to avoid opening multiple times the same netcdf
        file.

        Args:
            date: Date for wich to return dimension informations.
        
        Returns:
            dict[str, set[str]]: {coordinates: set of model names}
            dict[str, set[str]]: {variables: set of model names}
        """
        coords_map: dict[str, set[str]] = defaultdict(set)
        vars_map: dict[str, set[str]] = defaultdict(set)
        for path in self._paths_for_date(date):
            with xr.open_dataset(path) as ds:
                coords_map[str(ds.coords)].add(path.parent.name)
                vars_map[str(ds.variables)].add(path.parent.name)
        return coords_map, vars_map

    def _pct_for_date(self, date: dt.date) -> float:
        """Calculate a quality percentage for *date*.

        The percentage represents the similarity between the files for a date.
        If all the files have the same coordinates and variables, returns 1.
        if all the files have different coordinates and variables, returns 0.

        Args:
            date: Date for wich to return a percentage.

        Returns:
            float: Percentage in [0, 1] of similarity between files for a date.
        """

        # Get the number of distinct coordinate and variable signatures
        coords_map, vars_map = self._dims_info_for_date(date)
        nb_different_coords_variables = max(len(coords_map), len(vars_map))

        # Represent the number of different files as a percentage.
        # If all our files have the same coordinates and variables, we have
        # nb_different_coords_variables = 1 -> 100%.
        # If all our files have different coordinates and/or variables:
        # nb_different_coords_variables = 11 -> 0%
        return (11 - nb_different_coords_variables) / 10

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
        """Returns the popup's title and content displayed when clicking a date.

        The pop-up window's content displays the different coordinates and
        variables present in the files for a date. They are displayed with
        Alternating grey background color to differentiate successive entries,
        and if there are more than one coordinate or variable, the differences
        between them are highlighted with a red background.

        Args:
            date: Date selected.

        Returns:
            str: The pop-up window title.
            RichString: The pop-up window content.
        """

        # Define the title
        title = (
            date.strftime(r"%A %d %B %Y") + f" {self._pct_for_date(date) * 100:.2f}%"
        )

        # Retrieve cached coordinate and variable mappings
        coords, variables = self._dims_info_for_date(date)

        # Colors used for alternating rows and mismatches
        bgs = ["#000000", "#373737"]
        diff_color = "#780000"
        title_color = "#969696"

        # Helper to render a block (coordinates or variables)
        def _render_block(items: dict[str, set[str]], offset: int = 0) -> RichString:
            """Returns a RichString representing the formatted block.

            *items* maps a signature string to the set of subdirectory names that
            contain it. *offset* is added to the block index to keep background
            colours alternating correctly when rendering multiple blocks.

            Args:
                items:  {representation of a coord or variable: set of model names}.
                offset: The position offset, necessary for consistent background color
                    alternating.
            """
            block_content = RichString("")
            split_keys = [key.split(" ") for key in items]
            headers = list(items.values())
            for block_idx, parts in enumerate(split_keys):
                # Header line listing the subdirectories containing this signature
                block_content += RichString(
                    ", ".join(headers[block_idx]) + "\n",
                    title_color,
                )
                for idx, word in enumerate(parts):
                    # Alternate background colours per block
                    color = bgs[(block_idx + offset) % 2]
                    # Highlight if any other signature differs at this position
                    if any(
                        idx >= len(other) or other[idx] != word for other in split_keys
                    ):
                        color = diff_color
                    block_content += RichString(word + " ", color)
                block_content += RichString("\n\n")
            return block_content

        # Render coordinates first, then variables (variables offset by number
        # of coord blocks)
        content = _render_block(coords)
        content += _render_block(variables, offset=len(coords))

        return title, content


if __name__ == "__main__":
    start_app(
        inspector=DimsInspector(),
        years=[2024, 2025, 2026],
        nb_processes=12,
    )
