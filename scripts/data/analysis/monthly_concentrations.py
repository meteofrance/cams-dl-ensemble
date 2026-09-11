"""Computes and plots mean concentration per hour of the different species
on the reanalysis data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR
from cams.types import LEADTIMES, SPECIES_NAMES, SpeciesNames


def compute_monthly_concentrations(
    dataset: CAMSDataset, species: list[SpeciesNames] = SPECIES_NAMES
) -> dict[SpeciesNames, dict[int, float]]:
    """Computes mean of hourly concentrations of pollutants over the reanalysis data.

    Args:
        dataset: A cams dataset.
        species: The list of species to compute the statistics on.

    Returns:
        dict[SpeciesNames, dict[int, float]]: Statistics dict of shape
            {species: {0: X, 1: X, ..., 12: X}}.
    """
    # Init sum and count for all species
    stats: dict[SpeciesNames, dict[int, float]] = {
        spe: {month: 0.0 for month in range(1, 13)} for spe in species
    }
    counts: dict[SpeciesNames, dict[int, int]] = {
        spe: {month: 0 for month in range(1, 13)} for spe in species
    }

    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing statistics"):
        try:
            target = sample.data["TARGET"]
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping to next sample.")
            continue
        month = sample.date_run.month
        for spe in species:
            value = float(target.sel(species=spe).isel(time=hour).mean())
            stats[spe][month] += value
            counts[spe][month] += 1

    for spe in species:
        for month in range(24):
            stats[spe][month] /= counts[spe][month]
    return stats


def plot_pollutants_by_month(
    data: dict[SpeciesNames, dict[int, float]], figsize: tuple[int, int] = (12, 6)
) -> None:
    """Plot mean pollutant concentrations by hour of the day.

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
    plt.ylim(bottom=1)
    plt.xlabel("Hour of the day")
    plt.ylabel("Mean concentration")
    plt.title("Hourly evolution of mean pollutant concentrations")
    plt.legend(title="Pollutants", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    species_str = "-".join(data.keys())
    plt.savefig(f"pollutant_mean_concentrations_{species_str}.png")


if __name__ == "__main__":
    # species: list[SpeciesNames] = ["NO2", "PM10", "PM2P5", "SO2"]
    species: list[SpeciesNames] = ["CO"]

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

    stats = compute_monthly_concentrations(dataset, species)
    plot_pollutants_by_month(stats)
