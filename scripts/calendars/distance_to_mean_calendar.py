import datetime as dt
import json
from collections import defaultdict
from collections.abc import Generator
from functools import cache
from pathlib import Path
from typing import Hashable

import xarray as xr
from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.colors import RDYLGN, color_from_pct
from typing_extensions import override

RAW_DATA_DIR = Path("/scratch/shared/cams-dl-ensemble/all_from_ads/")

# Path to cached overall species means (JSON)
OVERALL_MEANS_PATH = Path(__file__).with_name("overall_means.json")

# Colors for out‑of‑range percentages (same palette as dims calendar)
UNDER_0_COLOR = RichString("X", "#77CBFF", "#cb31ff")
OVER_1_COLOR = RichString("X", "#FF003C", "#e9a7ff")


# Compute or load overall species means (cached across runs)
def _compute_means() -> dict[Hashable, float]:
    """Iterate over all NetCDF files and compute the mean value per species.

    The result is a mapping ``species -> mean`` across the whole dataset.
    """
    sums: defaultdict[Hashable, float] = defaultdict(float)
    counts: defaultdict[Hashable, int] = defaultdict(int)
    for path in RAW_DATA_DIR.rglob("*.netcdf"):
        with xr.open_dataset(path) as ds:
            for var in ds.data_vars:
                val = float(ds[var].mean().item())
                sums[var] += val
                counts[var] += 1
    return {var: sums[var] / counts[var] for var in sums}


def _load_overall_means() -> dict[str, float]:
    """Load cached overall means from JSON or compute them if missing."""
    if OVERALL_MEANS_PATH.is_file():
        with open(OVERALL_MEANS_PATH, "r") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    overall = _compute_overall_means()
    with open(OVERALL_MEANS_PATH, "w") as f:
        json.dump(overall, f, indent=2)
    return overall


# Load once at import time
OVERALL_SPECIES_MEANS: dict[str, float] = _load_overall_means()


class DistanceToMeanInspector(InspectorABC):
    """Inspector that shows how far each (model, species) mean deviates from the overall mean.

    For a given date the calendar collects the mean value of each variable (species)
    from every model's NetCDF file, computes the overall mean per species, and then
    displays the absolute distance of each model's mean to that overall mean.
    """

    name = "Distance to Mean"

    def _paths_for_date(self, date: dt.date) -> Generator[Path, None, None]:
        """Yield all NetCDF files that belong to *date*."""
        pattern = f"**/{date.strftime('%Y_%m_%d')}*.netcdf"
        yield from RAW_DATA_DIR.rglob(pattern)

    @cache
    def _model_species_means(self, date: dt.date) -> dict[tuple[str, str], float]:
        """Return a mapping ``(model, species) -> mean value`` for *date*.

        *model* is inferred from the immediate parent directory name of the file.
        *species* is the variable name inside the dataset.
        """
        result: dict[tuple[str, str], float] = {}
        for path in self._paths_for_date(date):
            model = path.parent.name
            with xr.open_dataset(path) as ds:
                for var_name in ds.data_vars:
                    # Compute the mean over all dimensions of the variable
                    mean_val = float(ds[var_name].mean().item())
                    result[(model, var_name)] = mean_val
        return result

    def _distances(self, date: dt.date) -> dict[tuple[str, str], float]:
        """Absolute distance of each model/species mean to the overall species mean."""
        overall = OVERALL_SPECIES_MEANS
        per_model = self._model_species_means(date)
        return {
            (model, species): abs(value - overall[species])
            for (model, species), value in per_model.items()
        }

    def _max_distance(self, date: dt.date) -> float:
        """Maximum distance value for the given date (used for colour scaling)."""
        distances = self._distances(date).values()
        return max(distances) if distances else 0.0

    @override
    def color_for_date(self, date: dt.date) -> RichString:
        """Colour based on the relative distance of the worst‑case model/species.

        The percentage is ``max_distance / (max_distance + 1)`` to keep the value
        in the ``0‑1`` range; this mirrors the behaviour of the dims calendar.
        """
        max_dist = self._max_distance(date)
        # Avoid division by zero – treat zero distance as the middle of the scale
        pct = (max_dist / (max_dist + 1)) if max_dist else 0.5
        if pct < 0:
            return UNDER_0_COLOR
        if pct > 1:
            return OVER_1_COLOR
        return color_from_pct(pct, RDYLGN)

    @override
    def as_color_bar(self, size: int) -> list[RichString]:
        """Generate a colour bar from 0 % to 100 % distance.

        The bar simply interpolates the colour palette; the caller decides how it
        maps to actual data values.
        """
        return [color_from_pct(i / (size - 1), RDYLGN) for i in range(size)]

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
    start_app(
        inspector_cls=DistanceToMeanInspector,
        years=[2023, 2024, 2025, 2026],
        nb_processes=12,
    )
