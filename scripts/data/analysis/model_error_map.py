"""Computes and plots maps of the mean error of a composition model
compared to the TARGET reanalysis field.

The model is given as a command line argument and can be any of the 11
available composition models, or the "MEDIAN" keyword to use the median of
the 11 models.

Example:
    python scripts/data/analysis/model_error_map.py --model MOCAGE
    python scripts/data/analysis/model_error_map.py --model MEDIAN
"""

import argparse

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from cartopy.crs import PlateCarree
from joblib import Parallel, delayed
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR, SIZE_LAT, SIZE_LON
from cams.types import LEADTIMES, MODELS_NAMES, SpeciesNames

EXTENT = (-24.95, 44.95, 30.05, 71.95)


def compute_sample_error_map(sample: Sample, model: str) -> np.ndarray:
    """Compute the mean absolute error of a model for one sample.

    Args:
        sample: A sample of the dataset.
        model: Name of the model to analyse, or "MEDIAN".

    Returns:
        np.ndarray: Mean absolute error per species, averaged over time and
            level, with dimensions (species, latitude, longitude).
    """
    n_species = len(sample.species)
    try:
        data = sample.data
    except Exception as e:
        print(e)
        print(f"Could not load sample {sample}, skipping.")
        return np.zeros((n_species, SIZE_LAT, SIZE_LON))

    if model == "MEDIAN":
        model_vars = [v for v in data.data_vars if v != "TARGET"]
        field = xr.concat([data[v] for v in model_vars], dim="model").median(
            dim="model"
        )
    else:
        field = data[model]

    target = data["TARGET"]

    # Average over time and level for one day
    daily_error = abs((field - target).mean(dim=["time", "level"]))

    return daily_error.values


def compute_model_error_map(dataset: CAMSDataset, model: str) -> np.ndarray:
    """Compute the mean error map of a model compared to the TARGET field.

    Computation is done in parallel with joblib.

    Args:
        dataset: A cams dataset.
        model: Name of the model to analyse, or "MEDIAN".

    Returns:
        np.ndarray: Mean absolute error with dimensions
            (species, latitude, longitude).
    """
    res = Parallel(n_jobs=5)(
        delayed(compute_sample_error_map)(sample, model)
        for sample in tqdm(dataset.samples, desc="Computing model error map")
    )
    n_species = len(dataset.species)
    accumulated = np.zeros((n_species, SIZE_LAT, SIZE_LON))
    n_samples = 0

    for sample_error in res:
        if not isinstance(sample_error, np.ndarray):
            continue
        accumulated += sample_error
        n_samples += 1

    if n_samples == 0:
        raise ValueError("No valid samples found.")
    error_map = accumulated / n_samples

    return error_map


def plot_model_error_map(
    error_map: np.ndarray,
    model_name: str,
    species_list: list[SpeciesNames],
    cmap: str = "RdBu_r",
) -> None:
    """Plot error maps for each species.

    Args:
        error_map: Output from compute_model_error_map().
        model_name: Name of the model used for the title and the filename.
        species_list: The species to display.
        cmap: Matplotlib colormap.

    Returns:
        None: This function displays and saves a plot; it returns nothing.
    """
    sns.set_style("white")

    subplot_kw = {"projection": PlateCarree()}
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 7),
        constrained_layout=True,
        subplot_kw=subplot_kw,
    )

    axes = axes.flatten()

    for i, (ax, species) in enumerate(zip(axes, species_list)):
        field = error_map[i]

        vmax = np.abs(field).max()
        im = ax.imshow(
            field,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            extent=EXTENT,
        )

        ax.set_title(species)
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"), edgecolor="grey", linewidth=1
        )
        ax.coastlines(resolution="50m", color="black", linewidth=1)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean error")

    plt.suptitle(f"Mean {model_name} error versus VRA", fontsize=16)

    plt.savefig(f"model_error_map_{model_name}.png")


if __name__ == "__main__":
    import datetime as dt

    parser = argparse.ArgumentParser(
        description="Plot mean error map of a model, or the median of all models.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the model to plot, or 'MEDIAN' for the median of all models.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["NO2", "PM10", "PM2P5", "SO2", "O3", "CO"],
        help="Species to analyse.",
    )
    args = parser.parse_args()

    species: list[SpeciesNames] = args.species

    run_dates = get_run_dates(PROCESSED_DATA_DIR)

    # Dates when VRA is available
    run_dates = [date for date in run_dates if date < dt.datetime(2025, 1, 1)]

    dataset = CAMSDataset(
        run_dates=run_dates,
        lead_times=LEADTIMES,
        species=species,
        models=MODELS_NAMES if args.model == "MEDIAN" else [args.model],
        levels=[0],
    )

    error_map = compute_model_error_map(dataset, args.model)
    plot_model_error_map(error_map, args.model, species)
