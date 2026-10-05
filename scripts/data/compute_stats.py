"""Computes min/max of the different species on the Analysis data."""

import json
from typing import Any

import numpy as np
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR, STATS_PATH
from cams.types import LEADTIMES, MODELS_NAMES, SpeciesNames


class WelfordVariance:
    """Implements the Welford Algorithm to compute variance online.
    It allows to compute the variance of a dataset with only on pass on the data.
    See:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    """

    def __init__(self) -> None:
        """Implements the Welford Algorithm to compute variance online."""
        self.mean = 0.0
        self.count = 0
        self.m2 = 0.0

    def add_batch(self, x: np.ndarray):
        """Adds one batch of data to the intermediate computations."""
        self.count += x.size

        old_mean = self.mean
        delta = x - old_mean
        self.mean += np.sum(delta / self.count)

        delta2 = x - self.mean
        self.m2 += np.sum(delta * delta2)

    def compute(self) -> tuple[float, float]:
        """Returns mean and variance."""
        return self.mean, self.m2 / self.count


def compute_stats(dataset: CAMSDataset, species: list[SpeciesNames]) -> dict[str, Any]:
    """Computes min, max, mean, std over the reanalysis data.
    We compute the mean and std with the Welford algorithm to avoid 2 passes
    over the data.

    Args:
        dataset: A cams dataset.
        species: The list of species to compute the statistics on.

    Returns:
        dict: Statistics dict of shape {species: {min: min, max: max}}.
    """
    # Init stats for all species
    stats = {spe: {"min": np.inf, "max": -np.inf} for spe in species}
    mean_var_computer = {spe: WelfordVariance() for spe in species}

    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing statistics"):
        try:
            target = sample.data["TARGET"]
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping to next sample.")
            continue

        for spe in species:
            target_spe = target.sel(species=spe)

            current_min = float(
                target_spe.min(dim=["time", "level", "latitude", "longitude"])
            )
            current_max = float(
                target_spe.max(dim=["time", "level", "latitude", "longitude"])
            )
            stats[spe]["min"] = min(stats[spe]["min"], current_min)
            stats[spe]["max"] = max(stats[spe]["max"], current_max)

            mean_var_computer[spe].add_batch(target_spe.values)

    for spe in species:
        mean, var = mean_var_computer[spe].compute()
        stats[spe]["mean"] = float(mean)
        stats[spe]["std"] = float(np.sqrt(var))

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Computes min/max of the different species on the Analysis data.",
    )

    species: list[SpeciesNames] = ["CO", "NO2", "PM10", "PM2P5", "SO2", "O3"]

    # Compute stats
    stats = compute_stats(
        dataset=CAMSDataset(
            run_dates=get_run_dates(PROCESSED_DATA_DIR),
            models=[MODELS_NAMES[0]],
            # We compute the stats on the reanalysis,
            # so we only need the first 24h of a sample
            # Else we will have overlaps with next sample, and compute some stats twice
            lead_times=LEADTIMES[:24],
            species=species,
            levels=[0],
        ),
        species=species,
    )
    for k, v in stats.items():
        print(k, v)

    # Save stats as json
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved in {STATS_PATH}!")
