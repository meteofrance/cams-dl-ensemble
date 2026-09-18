import datetime as dt
from pathlib import Path
import re

from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.colors import RDYLGN, color_from_pct
from typing_extensions import override

from cams.sample import Sample
from cams.types import LEADTIMES, MODELS_NAMES

UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


class SampleValidityInspector(InspectorABC):
    """Inspector that shows if the sample for each date is valid."""

    name = "Sample Validity"
    root_dir = Path("/scratch/shared/cams-dl-ensemble/all_from_ads")
    target_nb_files_total = 3 * 365 * 11
    target_nb_files_per_day = 11

    def _sample_for_date(self, date: dt.date) -> Sample:
        """Returns a sample for the given date.

        Args:
            date: Date to look up.

        Returns:
            Sample: The sample for the given date, valid or not.
        """
        return Sample(
            date_run=date,
            models=MODELS_NAMES,
            lead_times=LEADTIMES,
            levels=[0],
            species=["CO", "NO2", "PM10", "PM2P5", "SO2", "O3"],
        )

    @override
    def color_for_date(self, date: dt.date) -> RichString:
        """Returns the color for a given date.

        Args:
            date: date.

        Returns:
            RichString: The text and color associated
                to the given date.
        """
        if self._sample_for_date(date).is_valid:
            pct = 1
        else:
            pct = 0
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
        content = str(self._sample_for_date(date))
        content = "\n".join(
            [
                content[i: i+40]
                for i in range(0, len(content) - 41, 40)
            ]
        )

        return date.strftime(r"%Y %m %d"), content


if __name__ == "__main__":
    start_app(
        inspector=SampleValidityInspector(),
        years=[2023, 2024, 2025, 2026],
        nb_processes=8,
    )
