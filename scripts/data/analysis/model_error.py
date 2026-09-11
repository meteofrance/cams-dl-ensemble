"""Computes the error of the composition models (MAE and bias) per model,
lead time and species, compared to the TARGET reanalysis field, and plots it.

Example:
    python scripts/data/analysis/model_error.py
    python scripts/data/analysis/model_error.py --metric bias --species O3 NO2
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
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


def compute_model_error(
    dataset: CAMSDataset,
) -> dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]]:
    """Compute mean absolute error and mean bias of each model compared to
    the TARGET reanalysis, per species and per lead time.

    Args:
        dataset: A cams dataset.

    Returns:
        dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]]:
            Statistics dict of shape
            {model: {species: {lead_time: {"mae": X, "bias": Y}}}}.
    """
    mae_sums: defaultdict[ModelsNames, dict[SpeciesNames, dict[int, float]]] = (
        defaultdict(
            lambda: {
                spe: {lt: 0.0 for lt in dataset.lead_times} for spe in dataset.species
            }
        )
    )
    bias_sums: defaultdict[ModelsNames, dict[SpeciesNames, dict[int, float]]] = (
        defaultdict(
            lambda: {
                spe: {lt: 0.0 for lt in dataset.lead_times} for spe in dataset.species
            }
        )
    )
    counts: defaultdict[ModelsNames, dict[SpeciesNames, dict[int, int]]] = defaultdict(
        lambda: {spe: {lt: 0 for lt in dataset.lead_times} for spe in dataset.species}
    )

    sample: Sample
    for sample in tqdm(dataset.samples, desc="Computing model error"):
        try:
            data = sample.data
        except Exception as e:
            print(e)
            print(f"Could not load sample {sample}, skipping to next sample.")
            continue
        target = data["TARGET"]
        for model in dataset.models:
            forecast = data[model]
            diff = forecast - target
            for spe in dataset.species:
                for lt in dataset.lead_times:
                    values = diff.sel(species=spe, lead_time=lt).values
                    mae_sums[model][spe][lt] += float(abs(values).mean())
                    bias_sums[model][spe][lt] += float(values.mean())
                    counts[model][spe][lt] += 1

    error: dict[ModelsNames, dict[SpeciesNames, dict[int, dict[str, float]]]] = {
        model: {
            spe: {
                lt: {
                    "mae": mae_sums[model][spe][lt] / counts[model][spe][lt],
                    "bias": bias_sums[model][spe][lt] / counts[model][spe][lt],
                }
                for lt in dataset.lead_times
            }
            for spe in dataset.species
        }
        for model in dataset.models
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
            ax.plot(
                lead_times, values, marker="o", linewidth=2, markersize=4, label=model
            )
        ax.set_title(spe)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Models", loc="upper left", frameon=True)

    fig.suptitle(f"Model {metric} versus reanalysis", fontsize=16)
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
    parser.add_argument(
        "--metric",
        choices=["mae", "bias"],
        default="mae",
        help="Error metric to plot.",
    )
    args = parser.parse_args()

    species: list[SpeciesNames] = args.species
    models: list[ModelsNames] = args.models

    run_dates = get_run_dates(PROCESSED_DATA_DIR)
    # run_dates = [date for date in run_dates if date < dt.datetime(2025, 1, 1)]
    run_dates = [date for date in run_dates if date < dt.datetime(2023, 9, 4)]

    dataset = CAMSDataset(
        run_dates=run_dates,
        models=models,
        lead_times=LEADTIMES,
        species=species,
        levels=[0],
    )
    print("Len dataset:", len(dataset))

    error = compute_model_error(dataset)
    plot_model_error(error, args.metric, dataset.lead_times, species)
