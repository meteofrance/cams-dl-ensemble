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

    name = "Dimentions"

    def _paths_for_date(self, date: dt.date) -> Generator[Path, None, None]:
        """Yield all NetCDF files for *date*.

        Uses a glob pattern matching the date prefix (YYYY_MM_DD) anywhere
        under :data:`RAW_DATA_DIR`.
        """
        pattern = f"**/{date.strftime('%Y_%m_%d')}*.netcdf"
        yield from RAW_DATA_DIR.rglob(pattern)

    @cache
    def _dims_info_for_date(
        self, date: dt.date
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Return mappings of coordinate/variable signatures to subdirectory names.

        The result is cached because the underlying NetCDF files are immutable
        during a single run. Each key is the stringified signature, and the value
        is the set of immediate parent directory names that contain that file.
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

        The metric is ``(11 - max(num_coords, num_variables)) / 10`` which
        yields a float where values below 0 or above 1 are considered out of
        range.
        """
        coords_map, vars_map = self._dims_info_for_date(date)
        # Number of distinct coordinate and variable signatures
        num_coords = len(coords_map)
        num_vars = len(vars_map)
        return (11 - max(num_coords, num_vars)) / 10

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
            """Return a RichString representing the formatted block.

            *items* maps a signature string to the set of subdirectory names that
            contain it. *offset* is added to the block index to keep background
            colours alternating correctly when rendering multiple blocks.
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
        inspector_cls=DimsInspector,
        years=[2024, 2025, 2026],
        nb_processes=12,
    )
