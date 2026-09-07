"""Computes min/max of the different species on the Analysis data."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR
from cams.types import LEADTIMES, SPECIES_NAMES, SpeciesNames


def compute_hourly_concentrations(
    dataset: CAMSDataset, species: list[SpeciesNames] = SPECIES_NAMES
) -> dict[SpeciesNames, dict[int, float]]:
    """Computes max of hourly concentrations of pollutants over the reanalysis data.

    Args:
        dataset: A cams dataset.
        species: The list of species to compute the statistics on.

    Returns:
        dict[SpeciesNames, dict[int, float]]: Statistics dict of shape
            {species: {0: X, 1: X, ..., 23: X}}.
    """
    # Init min and max for all species
    stats: dict[SpeciesNames, dict[int, float]] = {
        spe: {hour: -np.inf for hour in range(24)} for spe in species
    }

    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing statistics"):
        try:
            target = sample.data["TARGET"]
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping to next sample.")
            continue
        max_values = target.max(dim=["level", "latitude", "longitude"])

        for spe in species:
            for hour in range(24):
                new_value = float(max_values.sel(species=spe).isel(time=hour).values)
                stats[spe][hour] = max(stats[spe][hour], new_value)
    return stats


def plot_pollutants_by_hour(
    data: dict[SpeciesNames, dict[int, float]], figsize: tuple[int, int] = (12, 6)
) -> None:
    """Plot maximum pollutant concentrations by hour of the day.

    Args:
        data: Dictionary with the structure
            {species: {hour: value, ...}, ...}.
        figsize: Matplotlib figure size.

    Returns:
        None: This function displays and saves a plot; it returns nothing.
    """
    # Clean plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    # Automatic color palette
    palette = sns.color_palette("tab10", n_colors=len(data))

    # Plot each pollutant
    for (pollutant, hourly_values), color in zip(data.items(), palette):
        hours = sorted(hourly_values.keys())
        values = [hourly_values[h] for h in hours]

        plt.plot(
            hours,
            values,
            marker="o",
            linewidth=2,
            markersize=5,
            label=pollutant,
            color=color,
        )

    # Figure formatting
    plt.xticks(range(24))
    plt.xlabel("Hour of the day")
    plt.ylabel("Maximum concentration")
    plt.title("Hourly evolution of maximum pollutant concentrations")
    plt.legend(title="Pollutants", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig("pollutant_max_concentrations.png")


def compute_target_climatology(dataset: CAMSDataset) -> xr.DataArray:
    """Compute climatological mean maps for each chemical species
    using the TARGET field from all samples.

    Args:
        dataset: Dataset containing samples accessible through
            `for sample in dataset.samples`.

    Returns:
        xr.DataArray: Climatological mean with dimensions
            (species, latitude, longitude).
    """
    accumulated: xr.DataArray | None = None
    n_samples = 0

    for sample in tqdm(dataset.samples, desc="Computing climatology"):
        try:
            target = sample.data["TARGET"]
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping.")
            continue

        # Average over time and level for one day
        daily_mean = target.mean(dim=["time", "level"])

        if accumulated is None:
            accumulated = daily_mean.copy()
        else:
            accumulated += daily_mean

        n_samples += 1

    if n_samples == 0:
        raise ValueError("No valid samples found.")

    climatology = accumulated / n_samples

    return climatology


def plot_species_climatology(climatology: xr.DataArray, cmap: str = "RdBu_r") -> None:
    """Plot climatology maps for each species.

    Args:
        climatology: Output from compute_target_climatology().
        cmap: Matplotlib colormap.

    Returns:
        None: This function displays and saves a plot; it returns nothing.
    """
    sns.set_style("white")

    species_list = climatology.species.values

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(18, 10), constrained_layout=True
    )

    axes = axes.flatten()

    for ax, species in zip(axes, species_list):
        field = climatology.sel(species=species)

        im = ax.pcolormesh(
            climatology.longitude,
            climatology.latitude,
            field,
            shading="auto",
            cmap=cmap,
        )

        ax.set_title(species)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean concentration")

    plt.suptitle("Climatology of pollutant concentrations", fontsize=16)

    plt.savefig("climato.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analysis data.",
    )
    species: list[SpeciesNames] = ["NO2", "PM10", "PM2P5", "SO2", "O3"]

    dataset = CAMSDataset(
        run_dates=get_run_dates(PROCESSED_DATA_DIR),
        models=["MOCAGE"],
        # We compute the stats on the reanalysis,
        # so we only need the first 24h of a sample
        # Else we will have overlaps with next sample, and compute some stats twice
        lead_times=LEADTIMES[:24],
        species=species,
        levels=[0],
    )

    stats = compute_hourly_concentrations(dataset, species)
    plot_pollutants_by_hour(stats)

    # Compute climatology
    climatology = compute_target_climatology(dataset)

    # Plot maps
    plot_species_climatology(climatology)
