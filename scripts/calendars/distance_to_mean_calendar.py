import datetime as dt
import json
from collections import defaultdict
from collections.abc import Generator
from functools import cache
from pathlib import Path
from tqdm import tqdm
from typing import Hashable
from cams.stats import DateStats, load_stats

import xarray as xr
from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.colors import RDYLGN, color_from_pct
from typing_extensions import override

from cams.types import SpeciesNames

RAW_DATA_DIR = Path("/scratch/shared/cams-dl-ensemble/all_from_ads/")

# Path to cached overall species means (JSON)
OVERALL_MEANS_PATH = Path(__file__).with_name("overall_means.json")

# Colors for out‑of‑range percentages (same palette as dims calendar)
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


class DistanceToMeanInspector(InspectorABC):
    """Inspector that shows how far each (model, species) mean deviates from
    the overall mean.

    For a given date the calendar collects the mean value of each variable (species)
    from every model's NetCDF file, computes the overall mean per species, and then
    displays the absolute distance of each model's mean to that overall mean.
    """
    name = "Distance to Mean"

    def __init__(self) -> None:
        """Load the per-date species means and compute the summary statistics.

        ``self._means`` maps ``date -> {"mean": overall mean distance to mean,
        "min_distance": ..., "max_distance": ...}`` across all models/species.
        """
        self._stats: dict[dt.date, DateStats] = load_stats()
        self._mean_per_species: dict[SpeciesNames, float] = {
            # TODO: define the mean per species dict
        }

        

    @override
    def color_for_date(self, date: dt.date) -> RichString:
        """Colour based on the relative distance of the worst‑case model/species.

        The percentage is ``max_distance / (max_distance + 1)`` to keep the value
        in the ``0‑1`` range; this mirrors the behaviour of the dims calendar.
        """
        dist = max(self._means[date].values())
        # Avoid division by zero – treat zero distance as the middle of the scale
        pct = dist + abs(self.min_dist_to_mean) / (self.max_dist_to_mean - self.min_dist_to_mean)
        if pct < 0:
            return UNDER_0_COLOR
        if pct > 1:
            return OVER_1_COLOR
        return color_from_pct(pct, RDYLGN)

    @override
    def as_color_bar(self, size: int) -> tuple[list[RichString], list[str]]:
        """Generate a colour bar from 0 % to 100 % distance.

        The bar simply interpolates the colour palette; the caller decides how it
        maps to actual data values.
        """
        colors = [color_from_pct(i / (size - 1), RDYLGN) for i in range(size)]
        step = (self.max_dist_to_mean - self.min_dist_to_mean) / size
        labels = []
        current = self.min_dist_to_mean
        i = 0
        while current <= self.max_dist_to_mean:
            if i != size and i % 5 !=0:
                labels.append("")
            else:
                labels.append(f"{current:.2f}")

            current += step

        return colors, labels

    @override
    def popup_content(self, date: dt.date) -> tuple[str, RichString]:
        """Return a title and a RichString with the distance table for *date*.

        The table lists ``model | species | distance`` rows, colour‑coded per row
        based on the distance proportion of the maximum distance for that date.
        """
        title = (
            date.strftime(r"%A %d %B %Y")
            + f" – max distance {self._max_distance(date):.3f}"
        )

        distances = self._distances(date)
        max_dist = self._max_distance(date) or 1.0  # avoid zero division

        content = RichString("")
        header = RichString("Model | Species | Distance\n", "#373737")
        content += header
        for (model, species), dist in sorted(distances.items()):
            # Normalise distance to 0‑1 range for colour selection
            pct = dist / max_dist
            colour = color_from_pct(pct, RDYLGN)
            line = f"{model} | {species} | {dist:.3f}\n"
            content += RichString(line, colour)
        return title, content


if __name__ == "__main__":
    import argparse

    start_app(
        inspector_cls=DistanceToMeanInspector,
        years=[2023, 2024, 2025, 2026],
        nb_processes=12,
    )
