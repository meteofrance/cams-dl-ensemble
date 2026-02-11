"""Computes min/max of the different species on the Analysis data."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from cams.dataset import get_run_dates
from cams.sample import Sample
from cams.settings import CAMS_DATASET_DIR, STATS_PATH


def get_list_samples(data_dir: Path) -> list[Sample]:
    """Returns the list of valid samples available in data_dir."""
    run_dates = get_run_dates(data_dir)
    samples = [Sample(date, 15, data_dir) for date in run_dates]
    samples = [sample for sample in samples if sample.is_valid]
    return samples


def compute_stats(samples: list[Sample]) -> dict[str, Any]:
    """Computes min/max over the reanalysis data.

    Args:
        samples (list[Sample]): A list of valid Samples

    Returns:
        dict: A dict of the statistics
    """
    vmin, vmax = np.inf, -np.inf  # Init min and max
    for sample in tqdm(samples, desc="Computing statistics"):
        data = sample.target_data.tensor[0].numpy()
        new_min, new_max = np.min(data), np.max(data)
        vmin = min(vmin, new_min)
        vmax = max(vmax, new_max)
    stats = {"O3": {"min": float(vmin), "max": float(vmax)}}
    return stats


if __name__ == "__main__":
    samples = get_list_samples(CAMS_DATASET_DIR)
    stats = compute_stats(samples)

    # Save stats as json
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
