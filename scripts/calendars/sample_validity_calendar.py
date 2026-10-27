import datetime as dt
from pathlib import Path

from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.colors import RDYLGN, color_from_pct
from typing_extensions import override

from cams.sample import Sample
from cams.types import LEADTIMES, LEVELS, MODELS_NAMES, SPECIES_NAMES

UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


class SampleValidityInspector(InspectorABC):
    """Inspector that shows if the sample for each date is valid."""

    name = "Sample Validity"
    root_dir = Path("/scratch/shared/cams-dl-ensemble/median_dataset")
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
            models=["ensemble"],
            lead_times=LEADTIMES,
            levels=LEVELS,
            species=[
                "NH3",
                "CO",
                "HCHO",
                "CHOCHO",
                "NO2",
                "NO",
                "NMVOC",
                "O3",
                "PM25",
                # "EC",
                # "RES",
                "SIA",
                # "EC",
                # "TOT",
                "PM25",
                # "OM",
                "PM10",
                "DUST",
                "DYNSAL",
                # "PM",
                # "WF",
                "PANS",
                "SO2",
            ],
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
        sample = self._sample_for_date(date)
        content = f"{sample.is_valid=}\n\n"
        nb_exists = 0
        for path in sorted(sample.input_paths + sample.target_paths):
            nb_exists += 1 if path.exists() else 0
            content += f"{path.exists()} {path.parent.name}/{path.name}\n"
            if not path.exists():
                zip_path = path.with_suffix('.zip')
                content += f"    {zip_path.exists()} {zip_path.name}\n"
        content += "\n"

        content += f"{nb_exists}/{len(sample.input_paths + sample.target_paths)}\n"

        content += "\n"
        str_sample = str(sample)
        content += "\n".join(
            [str_sample[i : i + 40] for i in range(0, len(str_sample) - 41, 40)]
        )



        return date.strftime(r"%Y %m %d"), content


if __name__ == "__main__":
    start_app(
        inspector=SampleValidityInspector(),
        years=[2023, 2024, 2025, 2026],
        nb_processes=8,
    )
