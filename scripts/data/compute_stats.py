"""Computes min/max of the different species on the Analysis data."""

import json
from typing import Any

import numpy as np
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR, STATS_PATH, AVAILABLE_SPECIES, MODEL_NAMES


def compute_stats(dataset: CAMSDataset) -> dict[str, Any]:
    """Computes min/max over the reanalysis data.

    Args:
        dataset: A cams dataset.

    Returns:
        dict: Statistics dict of shape {species: {min: min, max: max}}.
    """
    # Init min and max for all species
    stats = {species: {"min": np.inf, "max": -np.inf} for species in AVAILABLE_SPECIES}

    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing statistics"):
        try:
            target = sample.data["target"]
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping to next sample.")
            continue
        min_values = target.min(dim=["time", "level", "latitude", "longitude"])
        max_values = target.max(dim=["time", "level", "latitude", "longitude"])

        for species in AVAILABLE_SPECIES:
            stats[species]["min"] = min(
                stats[species]["min"], float(min_values.sel(species=species).values)
            )
            stats[species]["max"] = max(
                stats[species]["max"], float(max_values.sel(species=species).values)
            )
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Computes min/max of the different species on the Analysis data.",
    )

    # Compute stats
    stats = compute_stats(
        dataset=CAMSDataset(
            run_dates=get_run_dates(PROCESSED_DATA_DIR),
            models=[model.lower() for model in MODEL_NAMES],
            # We compute the stats on the reanalysis, so we only need the first 24h of a sample
            # Else we will have overlaps with next sample, and compute some stats twice
            lead_times=[i for i in range(0, 24)],
            species=AVAILABLE_SPECIES,
            levels=[0],
        )
    )
    for k, v in stats.items():
        print(k, v)

    # Save stats as json
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved in {STATS_PATH}!")
