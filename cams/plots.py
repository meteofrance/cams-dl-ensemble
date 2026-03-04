import json
import warnings
from pathlib import Path

import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import torch
from cartopy.crs import PlateCarree
from matplotlib.axes import Axes
from mfai.pytorch.namedtensor import NamedTensor

from cams.sample import Sample
from cams.settings import STATS_PATH

# Setup cache dir for cartopy to avoid downloading data each time
cartopy_cache_dir = Path("/scratch/shared/cartopy")
if cartopy_cache_dir.exists():
    cartopy.config["data_dir"] = str(cartopy_cache_dir)

# Constants
MOSAIC: list[list[str]] = [
    ["MATCH", "MINNI", "CHIMERE", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["MOCAGE", "MONARCH", "EURADIM", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["EMEP", "GEMAQ", "SILAM", "DEHM", "LOTOS", "TARGET", "TARGET", "TARGET"],
]
UNITS = {"O3": "Ozone (µg/m3)"}
CMAP = "turbo"
EXTENT = (-24.95, 44.95, 30.05, 71.95)


def get_vmin_vmax(species_name: str) -> tuple[float, float] | tuple[None, None]:
    """Retrieves vmin and vmax for one species. Returns None if stats file not found."""
    if STATS_PATH.exists():
        with open(STATS_PATH, "r") as file:
            STATS = json.load(file)
        vmin = STATS[species_name]["min"]
        vmax = STATS[species_name]["max"]
        return vmin, vmax
    else:
        warnings.warn(
            f"Statistics file not found: {STATS_PATH}. "
            "Please run `scripts/compute_stats.py`. "
            "Using vmin, vmax = None, None in plots instead."
        )
        return None, None


def format_axis(ax: Axes, title: str) -> None:
    """Formats a given plot axis with title, labels, ticks and coastlines.

    Args:
        ax: A matplotlib Axes.
        title: The title of this axes.
    """
    ax.set_title(title)
    ax.set(xticklabels=[], yticklabels=[])
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect(1.8)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="grey", linewidth=1)  # type: ignore[reportAttributeAccessIssue]
    ax.coastlines(resolution="50m", color="black", linewidth=1)  # type: ignore[reportAttributeAccessIssue]


def plot_sample(sample: Sample, save_path: Path, species_name: str = "O3") -> None:
    """Plots a sample's input and target data for one parameter only.

    Args:
        sample: Sample we want to plot.
        save_path: The path where the plot will be saved.
        species_name: The name of the species to plot.
    """
    x, y = sample.input_data, sample.target_data
    median = torch.median(x.tensor, dim=0).values
    vmin, vmax = get_vmin_vmax(species_name)

    # Create the different subfigures
    scale = 2.5
    subplot_kw = {"projection": PlateCarree()}
    fig, axs = plt.subplot_mosaic(
        mosaic=MOSAIC,  # type: ignore[reportArgumentType]
        layout="constrained",
        figsize=(8 * scale, 3.2 * scale),
        subplot_kw=subplot_kw,
    )

    # Render the 11 models to their corresponding plot cell
    cell_name: str
    ax: Axes
    for cell_name, ax in axs.items():
        if cell_name in ["MEDIAN", "TARGET"]:
            continue
        ax.imshow(x[cell_name][0], cmap=CMAP, vmin=vmin, vmax=vmax, extent=EXTENT)
        format_axis(ax, cell_name)

    # Render the median to its corresponding plot cell
    axs["MEDIAN"].imshow(median, cmap=CMAP, vmin=vmin, vmax=vmax, extent=EXTENT)
    format_axis(axs["MEDIAN"], "Median Ensemble = Baseline")

    # Render the target to its corresponding plot cell
    img = axs["TARGET"].imshow(
        y["Analysis"][0], cmap=CMAP, vmin=vmin, vmax=vmax, extent=EXTENT
    )
    format_axis(axs["TARGET"], "Analysis = Target")

    # Add Colorbar
    cbar = fig.colorbar(img, ax=axs["TARGET"])
    cbar.set_label(UNITS[species_name], size=13)

    # Add the plot's title
    run_str = sample.date_run.strftime(r"%Y-%m-%d %Hh")
    title = f"{species_name} - Run {run_str} - Leadtime +{sample.lead_time}h"
    fig.suptitle(title, size=16)

    plt.savefig(save_path)


def plot_y_vs_yhat(
    y: NamedTensor, y_hat: NamedTensor, save_path: Path, title: str = ""
) -> None:
    """Plots the ground truth VS the prediction from a model."""
    subplot_kw = {"projection": PlateCarree()}
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 5), subplot_kw=subplot_kw)
    axs = axs.flatten()

    vmin, vmax = get_vmin_vmax("O3")
    plot_kwargs = {"cmap": CMAP, "vmin": vmin, "vmax": vmax, "extent": EXTENT}

    axs[0].imshow(y.tensor[0].cpu(), **plot_kwargs)
    format_axis(axs[0], "Ground Truth = Analysis")
    img = axs[1].imshow(y_hat.tensor[0].cpu(), **plot_kwargs)
    format_axis(axs[1], "Prediction")

    cbar = fig.colorbar(img, ax=axs, fraction=0.023)
    cbar.set_label(UNITS["O3"], size=13)

    fig.suptitle(title, size=18)
    plt.savefig(save_path)
