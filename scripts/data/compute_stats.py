"""Computes min/max of the different species on the Analysis data."""

import json
from typing import Any

import numpy as np
from math import prod
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR, STATS_PATH
from cams.types import LEADTIMES, MODELS_NAMES, SpeciesNames


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
    stats = {spe: {"min": np.inf, "max": -np.inf, "mean": 0, "m2": 0, "n": 0} for spe in species}

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

            current_min = float(target_spe.min(dim=["time", "level", "latitude", "longitude"]))
            current_max = float(target_spe.max(dim=["time", "level", "latitude", "longitude"]))

            stats[spe]["min"] = min(stats[spe]["min"], current_min)
            stats[spe]["max"] = max(stats[spe]["max"], current_max)

            size = target_spe.size
            n = stats[spe]["n"]
            new_n = n + size

            mean_bloc = float(target_spe.mean())
            m2_bloc = float(((target_spe - mean_bloc) ** 2).sum())

            delta_mean = mean_bloc - stats[spe]["mean"]

            stats[spe]["mean"] += delta_mean * (size / new_n)

            stats[spe]["m2"] += m2_bloc + (delta_mean ** 2) * (n * size / new_n)
            stats[spe]["n"] = new_n

    for spe in species:
        var = stats[spe]["m2"] / stats[spe]["n"] if stats[spe]["n"] > 0 else 0
        stats[spe]["std"] = float(np.sqrt(var))
        del stats[spe]["m2"]
        del stats[spe]["n"]

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
