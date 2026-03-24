"""Computes min/max of the different species on the Analysis data."""

import json
from typing import Any

import numpy as np
from tqdm import tqdm

from cams.dataset import CAMSDataset
from cams.sample import Sample
from cams.settings import STATS_PATH


def compute_stats(dataset: CAMSDataset) -> dict[str, Any]:
    """Computes min/max over the reanalysis data.

    Args:
        dataset: A cams dataset.

    Returns:
        dict: Statistics dict of shape {species: {min: min, max: max}}.
    """
    # Init min and max
    vmin, vmax = np.inf, -np.inf

    # Update min and max
    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing statistics"):
        target_data = sample.target_data.tensor[0].numpy()
        new_min, new_max = np.min(target_data), np.max(target_data)
        vmin = min(vmin, new_min)
        vmax = max(vmax, new_max)

    # Return values
    stats = {"O3": {"min": float(vmin), "max": float(vmax)}}
    return stats


if __name__ == "__main__":
    # Compute stats
    stats = compute_stats(dataset=CAMSDataset())

    # Save stats as json
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
