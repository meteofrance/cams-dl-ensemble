import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats as sps
from tqdm import tqdm

from cams.dataset import get_run_dates
from cams.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from cams.types import MODELS_NAMES, STATISTICS_NAMES, SpeciesNames

# Path to the cached per-date, per-model stats (JSON)
OVERALL_MEANS_PATH = Path(__file__).with_name("stats.json")

# Species present in the raw data, expressed with the canonical species names
# from cams.types. Their NetCDF/GRIB variable name is "<species>.lower()_conc".
SPECIES: tuple[SpeciesNames, ...] = ("O3", "CO", "NO2", "PM10", "PM2P5", "SO2")

# Species read from the reanalysis monthly files (bare names, no _conc suffix).
REANALYSIS_VAR: dict[str, str] = {sp.lower(): f"{sp.lower()}_conc" for sp in SPECIES}

# Species appearing in the weighted-ensemble GRIB file names (PM2P5 is PM25).
ENSEMBLE_FILE_SPECIES: dict[str, str] = {
    "CO": "co_conc",
    "NO2": "no2_conc",
    "O3": "o3_conc",
    "PM10": "pm10_conc",
    "PM25": "pm2p5_conc",
    "SO2": "so2_conc",
}

# Model directory name for a given model key is its lowercase name.
MODEL_DIRS: tuple[str, ...] = tuple(name.lower() for name in MODELS_NAMES)


def _stats_from_array(
    values: np.ndarray, time_values: np.ndarray, time_axis: int
) -> dict[str, float | str]:
    """Compute all statistics over a numpy array of a species variable.

    Statistics are computed over the whole array. ``argmin`` and ``argmax``
    are the coordinate value of ``time_values`` at the location of the minimum
    and maximum of the array, along the ``time_axis`` dimension.

    Args:
        values: Data of a species variable.
        time_values: Coordinate values along the time axis.
        time_axis: Index of the time dimension in ``values``.

    Returns:
        dict: Statistics keyed by the names in :data:`STATISTICS_NAMES`.
    """
    flat = values.ravel()
    amin = float(np.min(flat))
    amax = float(np.max(flat))
    argmin_idx = np.unravel_index(int(np.argmin(flat)), values.shape)[time_axis]
    argmax_idx = np.unravel_index(int(np.argmax(flat)), values.shape)[time_axis]
    return dict(
        mean=float(np.mean(flat)),
        amin=amin,
        argmin=_coord_value(time_values[argmin_idx]),
        amax=amax,
        argmax=_coord_value(time_values[argmax_idx]),
        median=float(np.median(flat)),
        skew=float(sps.skew(flat)),
        kurtosis=float(sps.kurtosis(flat)),
        std=float(np.std(flat)),
    )


def _coord_value(value: Any) -> float | str:
    """Return a JSON-serialisable value for a coordinate scalar.

    Datetimes are converted to ISO strings so that the cached stats remain
    valid JSON.

    Args:
        value: Coordinate scalar (leadtime or datetime).

    Returns:
        float: A numeric coordinate value.
        str: An ISO string for a datetime coordinate value.
    """
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value).astype(str)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def _empty_stats() -> "Stats":
    """Return a Stats instance whose values indicate missing data (NaN)."""
    return _stats_from_dict({name: float("nan") for name in STATISTICS_NAMES})


@dataclass
class Stats:
    """Statistics computed over one species variable.

    The dataclass implements every statistic defined in
    :data:`cams.types.STATISTICS_NAMES` (mean, amin, argmin, amax, argmax,
    median, skew, kurtosis, std).

    Attributes:
        mean: Arithmetic mean.
        amin: Minimum value.
        argmin: Coordinate of the time axis at the minimum.
        amax: Maximum value.
        argmax: Coordinate of the time axis at the maximum.
        median: Median value.
        skew: Skewness.
        kurtosis: (Excess) kurtosis.
        std: Standard deviation.
    """

    mean: float
    amin: float
    argmin: float | str
    amax: float
    argmax: float | str
    median: float
    skew: float
    kurtosis: float
    std: float

    @classmethod
    def from_xarray(cls, data: xr.DataArray, time_dim: str) -> "Stats":
        """Compute the statistics of a data variable.

        Args:
            data: The data variable to summarise.
            time_dim: Name of the time axis used for ``argmin``/``argmax``.

        Returns:
            Stats: The statistics of ``data``.
        """
        time_axis = data.get_axis_num(time_dim)
        return _stats_from_dict(
            _stats_from_array(data.values, data.coords[time_dim].values, time_axis)
        )

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serializable dictionary of the statistics."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float | str]) -> "Stats":
        """Build a Stats instance from a dictionary.

        Args:
            data: Dictionary returned by :meth:`to_dict`.

        Returns:
            Stats: The reconstructed statistics.
        """
        return _stats_from_dict(data)


def _stats_from_dict(data: dict[str, float | str]) -> Stats:
    """Build a Stats instance from a dictionary of statistic values."""
    return Stats(**{name: data[name] for name in STATISTICS_NAMES})


class ModelStats(TypedDict):
    """Per-species statistics for a single model."""

    co_conc: Stats
    no2_conc: Stats
    o3_conc: Stats
    pm10_conc: Stats
    pm2p5_conc: Stats
    so2_conc: Stats


class DateStats(TypedDict):
    """Per-model statistics for a single date."""

    date: dt.date
    chimere: ModelStats
    dehm: ModelStats
    emep: ModelStats
    euradim: ModelStats
    gemaq: ModelStats
    lotos: ModelStats
    match: ModelStats
    minni: ModelStats
    mocage: ModelStats
    monarch: ModelStats
    reanalysis: ModelStats
    silam: ModelStats
    weighted_ensemble: ModelStats


def _empty_model_stats() -> ModelStats:
    """Return a ModelStats with missing (NaN) Stats for every species."""
    return ModelStats({f"{sp.lower()}_conc": _empty_stats() for sp in SPECIES})


def _species_stats_for_var(var: str) -> str:
    """Return the ModelStats key (``<species>_conc``) for a model data var."""
    return var if var.endswith("_conc") else f"{var}_conc"


def _compute_model_netcdf(
    model_dir: str, date: dt.date, stats: dict[str, ModelStats]
) -> None:
    """Compute stats for one model stored as a per-date NetCDF file."""
    paths = list(
        RAW_DATA_DIR.joinpath(model_dir).glob(f"{date.strftime(r'%Y_%m_%d')}-*.netcdf")
    )
    if not paths:
        return
    with xr.open_dataset(paths[0]) as ds:
        for var in ds.data_vars:
            key = _species_stats_for_var(var)
            if key in ModelStats.__annotations__:
                stats[model_dir][var] = Stats.from_xarray(ds[var], "time")


def _compute_reanalysis(date: dt.date, stats: dict[str, ModelStats]) -> None:
    """Compute stats for the reanalysis of a date from the monthly files."""
    for bare_species in REANALYSIS_VAR:
        path = (
            RAW_DATA_DIR
            / "reanalysis"
            / (f"cams.eaq.ira.ENSa.{bare_species}.l0.{date.strftime(r'%Y-%m')}.nc")
        )
        if not path.exists():
            continue
        key = REANALYSIS_VAR[bare_species]
        with xr.open_dataset(path) as ds:
            data = ds[bare_species].sel(time=(ds.time.dt.date == date))
            stats["reanalysis"][key] = Stats.from_xarray(data, "time")


def _compute_weighted_ensemble(date: dt.date, stats: dict[str, ModelStats]) -> None:
    """Compute stats for the weighted ensemble of a date from GRIB files."""
    import earthkit.data as ekd

    for file_species, key in ENSEMBLE_FILE_SPECIES.items():
        path = (
            RAW_DATA_DIR
            / "weighted_ensemble"
            / (f"{date.strftime(r'%Y_%m_%d')}-{file_species}-0m-0-96h.grib")
        )
        if not path.exists():
            continue
        data = ekd.from_source("file", str(path)).to_xarray().to_dataarray()
        data = data.isel(variable=0)
        # Expose the leadtime in hours instead of raw nanosecond deltas
        data = data.assign_coords(
            step=(data["step"] / np.timedelta64(1, "h")).astype(int)
        )
        stats["weighted_ensemble"][key] = Stats.from_xarray(data, "step")


def compute_stats_for_date(date: dt.date) -> DateStats:
    """Iterate over all raw files for the given date and compute stats
    value per model and per species.

    The 11 pollutant models are read from their per-date NetCDF files, the
    reanalysis from the monthly per-species NetCDF files and the weighted
    ensemble from the per-date, per-species GRIB files.

    Args:
        date: Date for which to compute stats.

    Returns:
        DateStats: Stats for the given date.
    """
    model_stats: dict[str, ModelStats] = {
        name: _empty_model_stats()
        for name in (*MODEL_DIRS, "reanalysis", "weighted_ensemble")
    }

    for model_dir in tqdm(MODEL_DIRS, desc="Compute stats", colour="#ff00ff"):
        _compute_model_netcdf(model_dir, date, model_stats)

    _compute_reanalysis(date, model_stats)
    _compute_weighted_ensemble(date, model_stats)

    return DateStats(date=date, **model_stats)


def compute_stats(n_jobs: int = -1) -> dict[dt.date, DateStats]:
    """Compute the stats for every run date available.

    Dates are processed in parallel using :func:`joblib.Parallel`.

    Args:
        n_jobs: Number of parallel jobs (``-1`` uses all available cores).

    Returns:
        dict: Statistics for each run date.
    """
    dates = get_run_dates(PROCESSED_DATA_DIR)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_stats_for_date)(date)
        for date in tqdm(dates, desc="Walk dates.")
    )

    return {date: result for date, result in zip(dates, results)}


def _stats_to_serializable(stats: dict[dt.date, DateStats]) -> dict[str, Any]:
    """Convert the stats to a plain JSON-serializable structure.

    Args:
        stats: Per-date stats to convert.

    Returns:
        dict: Nested dict with ISO date strings and plain scalar values.
    """
    serializable: dict[str, Any] = {}
    for date, date_stats in stats.items():
        date_serializable: dict[str, Any] = {"date": date.isoformat()}
        for model, model_stats in date_stats.items():
            if model == "date":
                continue
            date_serializable[model] = {
                species: species_stats.to_dict()
                for species, species_stats in model_stats.items()
            }
        serializable[date.isoformat()] = date_serializable
    return serializable


def _stats_from_serializable(data: dict[str, Any]) -> dict[dt.date, DateStats]:
    """Rebuild the stats from their JSON-serializable representation.

    Args:
        data: Nested dict produced by :func:`_stats_to_serializable`.

    Returns:
        dict: Reconstructed per-date stats.
    """
    stats: dict[dt.date, DateStats] = {}
    for date_str, date_data in data.items():
        date_stats: dict[str, Any] = {"date": dt.date.fromisoformat(date_str)}
        for model, model_data in date_data.items():
            if model == "date":
                continue
            date_stats[model] = ModelStats(
                {
                    species: Stats.from_dict(species_data)
                    for species, species_data in model_data.items()
                }
            )
        stats[dt.date.fromisoformat(date_str)] = DateStats(date_stats)
    return stats


def load_stats(n_jobs: int = -1) -> dict[dt.date, DateStats]:
    """Load the stats from disk, or compute and cache them if missing.

    Args:
        n_jobs: Number of parallel jobs (``-1`` uses all available cores).

    Returns:
        dict: Statistics for each run date.
    """
    if OVERALL_MEANS_PATH.exists():
        with open(OVERALL_MEANS_PATH, "r") as f:
            data = json.load(f)
        return _stats_from_serializable(data)

    stats = compute_stats(n_jobs=n_jobs)
    with open(OVERALL_MEANS_PATH, "w") as f:
        json.dump(_stats_to_serializable(stats), f, indent=4)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute and cache per-date stats.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 uses all cores).",
    )
    args = parser.parse_args()

    stats: dict[dt.date, DateStats] = load_stats(n_jobs=args.n_jobs)
    breakpoint()