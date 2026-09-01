"""Computes min/max of the different species on the Analysis data."""

import json
from typing import Any

import numpy as np
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR, STATS_PATH
from cams.types import MODELS_NAMES, SPECIES_NAMES, SpeciesNames


def compute_stats(
    dataset: CAMSDataset, species: list[SpeciesNames] = SPECIES_NAMES
) -> dict[str, Any]:
    """Computes min/max over the reanalysis data.

    Args:
        dataset: A cams dataset.
        species: The list of species to compute the statistics on.

    Returns:
        dict: Statistics dict of shape {species: {min: min, max: max}}.
    """
    # Init min and max for all species
    stats = {spe: {"min": np.inf, "max": -np.inf} for spe in species}

    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing statistics"):
        print(sample.data)
        try:
            target = sample.data["TARGET"]
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping to next sample.")
            continue
        min_values = target.min(dim=["time", "level", "latitude", "longitude"])
        max_values = target.max(dim=["time", "level", "latitude", "longitude"])

        for spe in species:
            stats[spe]["min"] = min(
                stats[spe]["min"], float(min_values.sel(species=spe).values)
            )
            stats[spe]["max"] = max(
                stats[spe]["max"], float(max_values.sel(species=spe).values)
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
            models=MODELS_NAMES,
            # We compute the stats on the reanalysis,
            # so we only need the first 24h of a sample
            # Else we will have overlaps with next sample, and compute some stats twice
            lead_times=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
            ],
            species=SPECIES_NAMES,
            levels=[0],
        )
    )
    for k, v in stats.items():
        print(k, v)

    # Save stats as json
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved in {STATS_PATH}!")
