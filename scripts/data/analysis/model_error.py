"""Computes the error of the composition models (MAE and bias) per model,
lead time and species, compared to the TARGET reanalysis field, and plots it.

Example:
    python scripts/data/analysis/model_error.py
    python scripts/data/analysis/model_error.py --metric bias --species O3 NO2
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from cams.dataset import CAMSDataset, get_run_dates
from cams.sample import Sample
from cams.settings import PROCESSED_DATA_DIR
from cams.types import (
    LEADTIMES,
    MODELS_NAMES,
    Leadtimes,
    ModelsNames,
    SpeciesNames,
)


def compute_sample_error(sample: Sample) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute the models MEA and Bias for one sample of the dataset."""
    n_models = len(sample.models)
    n_species = len(sample.species)
    n_lt = len(sample.lead_times)
    try:
        data = sample.data
    except Exception as e:
        print(e)
        print(f"Could not load sample {sample}, skipping to next sample.")
        return (
            np.zeros((n_models, n_species, n_lt)),
            np.zeros((n_models, n_species, n_lt)),
            0,
        )
    target = data["TARGET"]
    models_mae, models_bias = [], []
    for m, model in enumerate(dataset.models):
        diff = data[model] - target
        spatial_dims = [d for d in diff.dims if d not in ["species", "time"]]
        mae = np.abs(diff).mean(dim=spatial_dims).values
        bias = diff.mean(dim=spatial_dims).values
        models_mae.append(mae)
        models_bias.append(bias)
    return np.stack(models_mae), np.stack(models_bias), 1


def compute_model_error(
    dataset: CAMSDataset,
) -> dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]]:
    """Compute mean absolute error and mean bias of each model compared to
    the TARGET reanalysis, per species and per lead time.
    Computation is done in parallel with joblib.

    Args:
        dataset: A cams dataset.

    Returns:
        dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]]:
            Statistics dict of shape
            {model: {species: {lead_time: {"mae": X, "bias": Y}}}}.
    """
    res = Parallel(n_jobs=15)(
        delayed(compute_sample_error)(sample)
        for sample in tqdm(dataset.samples, desc="Computing model error")
    )
    all_mae, all_bias, all_counts = zip(*res)
    mae_sums = np.sum(all_mae, axis=0)
    bias_sums = np.sum(all_bias, axis=0)
    count = sum(all_counts)

    error: dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]] = {
        model: {
            spe: {
                lt: {
                    "mae": mae_sums[k][j][i] / count,
                    "bias": bias_sums[k][j][i] / count,
                }
                for i, lt in enumerate(dataset.lead_times)
            }
            for j, spe in enumerate(dataset.species)
        }
        for k, model in enumerate(dataset.models)
    }
    return error


def plot_model_error(
    error: dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]],
    metric: str,
    lead_times: list[Leadtimes],
    species: list[SpeciesNames],
    figsize: tuple[int, int] = (10, 20),
) -> None:
    """Plot the chosen error metric of each model as a function of lead time,
    for each species.

    Args:
        error: Output from compute_model_error().
        metric: The metric to plot, either "mae" or "bias".
        lead_times: The lead times to display on the x-axis.
        species: The species to display.
        figsize: Matplotlib figure size.

    Returns:
        None: This function displays and saves a plot; it returns nothing.
    """
    n_species = len(species)
    fig, axes = plt.subplots(
        nrows=n_species,
        ncols=1,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
    )
    if n_species == 1:
        axes = [axes]
    sns.set_style("whitegrid")

    for ax, spe in zip(axes, species):
        for model in error:
            values = [error[model][spe][lt][metric] for lt in lead_times]
            ax.plot(lead_times, values, linewidth=2, markersize=4, label=model)
        ax.set_title(spe)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Models", loc="lower right", frameon=True)

    fig.suptitle(f"Model {metric} versus VRA", fontsize=16)
    plt.savefig(f"model_{metric}.png")


if __name__ == "__main__":
    import argparse
    import datetime as dt

    parser = argparse.ArgumentParser(description="Model error analysis.")
    parser.add_argument(
        "--species",
        nargs="+",
        default=["NO2", "PM10", "PM2P5", "SO2", "O3", "CO"],
        help="Species to analyse.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_NAMES,
        help="Models to analyse.",
    )
    args = parser.parse_args()

    species: list[SpeciesNames] = args.species
    models: list[ModelsNames] = args.models

    run_dates = get_run_dates(PROCESSED_DATA_DIR)
    run_dates = [date for date in run_dates if date < dt.datetime(2025, 1, 1)]
    # run_dates = [date for date in run_dates if date < dt.datetime(2023, 9, 4)]

    dataset = CAMSDataset(
        run_dates=run_dates,
        models=models,
        lead_times=LEADTIMES,
        species=species,
        levels=[0],
    )
    print("Len dataset:", len(dataset))

    error = compute_model_error(dataset)
    plot_model_error(error, "mae", dataset.lead_times, species)
    plot_model_error(error, "bias", dataset.lead_times, species)
