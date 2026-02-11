import datetime as dt
import json
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import torch
from cartopy.crs import PlateCarree
from matplotlib.axes import Axes

from cams.sample import Sample
from cams.settings import STATS_PATH

# Constants
MOSAIC: list[list[str]] = [
    ["MATCH", "MINNI", "CHIMERE", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["MOCAGE", "MONARCH", "EURADIM", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["EMEP", "GEMAQ", "SILAM", "DEHM", "LOTOS", "TARGET", "TARGET", "TARGET"],
]
UNITS = {"O3": "Ozone (µg/m3)"}
CMAP = "turbo"
EXTENT = (-24.95, 44.95, 30.05, 71.95)
with open(STATS_PATH, "r") as file:
    STATS = json.load(file)


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
    vmin = STATS[species_name]["min"]
    vmax = STATS[species_name]["max"]

    # Create the different subfigures
    scale = 2.5
    subplot_kw = {"projection": PlateCarree()}
    fig, axs = plt.subplot_mosaic(  # type: ignore
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


if __name__ == "__main__":
    date = dt.datetime(2024, 7, 30)
    sample = Sample(date, 15)
    plot_sample(sample, Path(f"{date.strftime('%Y-%m-%d_O3')}.png"))
