import json
import math
import warnings
from pathlib import Path

import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.typing import HashableList
from mfai.pytorch.namedtensor import NamedTensor

from cams.sample import Sample
from cams.settings import STATS_PATH

# Setup cache dir for cartopy to avoid downloading data each time
cartopy_cache_dir = Path("/scratch/shared/cartopy")
if cartopy_cache_dir.exists():
    cartopy.config["data_dir"] = str(cartopy_cache_dir)

# Constants
MOSAIC: list[HashableList[str]] = [
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


def format_axis(ax: GeoAxes | Axes, title: str) -> None:
    """Formats a given plot axis with title, labels, ticks and coastlines.

    Args:
        ax: A matplotlib Axes.
        title: The title of this axes.
    """
    ax.set_title(title)
    ax.set(xticklabels=[], yticklabels=[])
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect(1.8)

    if isinstance(ax, GeoAxes):
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"), edgecolor="grey", linewidth=1
        )
        ax.coastlines(resolution="50m", color="black", linewidth=1)


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
        mosaic=MOSAIC,
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
    plt.close()


def plot_y_vs_yhat(
    y: NamedTensor, y_hat: NamedTensor, save_path: Path, title: str = ""
) -> None:
    """Plots the ground truth VS the prediction from a model."""
    subplot_kw = {"projection": PlateCarree()}
    fig = plt.figure(constrained_layout=True, figsize=(9, 8))
    subfig: np.typing.NDArray = fig.subfigures(nrows=2, ncols=1)  # type: ignore [reportAssignmentType]

    # Plot maps of species
    axes = subfig[0].subplots(nrows=1, ncols=2, subplot_kw=subplot_kw)
    axs = axes.flat
    vmin, vmax = get_vmin_vmax("O3")
    plot_kwargs = {"cmap": CMAP, "vmin": vmin, "vmax": vmax, "extent": EXTENT}
    axs[0].imshow(y.tensor[0].cpu(), **plot_kwargs)
    format_axis(axs[0], "Ground Truth = Analysis")
    img = axs[1].imshow(y_hat.tensor[0].cpu(), **plot_kwargs)
    format_axis(axs[1], "Prediction")
    cbar = subfig[0].colorbar(img, ax=axes, fraction=0.023)
    cbar.set_label(UNITS["O3"], size=13)

    # Plot difference btw y and y_hat
    ax = subfig[1].subplots(nrows=1, ncols=1, subplot_kw=subplot_kw)
    diff = y_hat.tensor[0].cpu() - y.tensor[0].cpu()
    img = ax.imshow(diff, cmap="RdBu_r", extent=EXTENT, vmin=-50, vmax=50)
    format_axis(ax, "Difference")
    cbar = subfig[1].colorbar(img, ax=ax, fraction=0.023)

    fig.suptitle(title, size=18)
    plt.savefig(save_path)
    plt.close()


def plot_y_vs_yhat_vs_median(
    x: NamedTensor, y: NamedTensor, y_hat: NamedTensor, save_path: Path, title: str = ""
) -> None:
    """Plots the ground truth, prediction, and median of inputs in three rows."""
    subplot_kw = {"projection": PlateCarree()}
    fig = plt.figure(constrained_layout=True, figsize=(9, 12))
    subfigs: np.typing.NDArray = fig.subfigures(nrows=3, ncols=1)  # type: ignore [reportAssignmentType]

    # Plot ground truth (full size)
    ax_gt: GeoAxes = subfigs[0].subplots(nrows=1, ncols=1, subplot_kw=subplot_kw)
    vmin, vmax = get_vmin_vmax("O3")
    plot_kwargs = {"cmap": CMAP, "vmin": vmin, "vmax": vmax, "extent": EXTENT}
    img_gt = ax_gt.imshow(y.tensor[0].cpu(), **plot_kwargs)
    format_axis(ax_gt, "Ground Truth = Analysis")
    cbar_gt = subfigs[0].colorbar(img_gt, ax=ax_gt, fraction=0.023)
    cbar_gt.set_label(UNITS["O3"], size=13)

    # Plot prediction and median side by side
    axes_pred_med: np.ndarray = subfigs[1].subplots(
        nrows=1, ncols=2, subplot_kw=subplot_kw
    )
    axs_pred_med = axes_pred_med.flat
    img_pred = axs_pred_med[0].imshow(y_hat.tensor[0].cpu(), **plot_kwargs)
    format_axis(axs_pred_med[0], "Prediction")
    axs_pred_med[1].imshow(x.tensor.cpu().median(dim=0).values, **plot_kwargs)
    format_axis(axs_pred_med[1], "Median of Inputs")
    cbar_pred = subfigs[1].colorbar(img_pred, ax=axes_pred_med, fraction=0.023)
    cbar_pred.set_label(UNITS["O3"], size=13)

    # Plot differences
    axes_diff = subfigs[2].subplots(nrows=1, ncols=2, subplot_kw=subplot_kw)
    axs_diff = axes_diff.flat
    diff_pred = y_hat.tensor[0].cpu() - y.tensor[0].cpu()
    img_diff_pred = axs_diff[0].imshow(
        diff_pred, cmap="RdBu_r", extent=EXTENT, vmin=-50, vmax=50
    )
    format_axis(axs_diff[0], "Difference (AI Prediction)")
    diff_med = x.tensor.cpu().median(dim=0).values - y.tensor[0].cpu()
    axs_diff[1].imshow(diff_med, cmap="RdBu_r", extent=EXTENT, vmin=-50, vmax=50)
    format_axis(axs_diff[1], "Difference (Median of Inputs)")
    subfigs[2].colorbar(img_diff_pred, ax=axes_diff, fraction=0.023)

    fig.suptitle(title, size=18)
    plt.savefig(save_path)
    plt.close()


def plot_named_tensor(
    nt: NamedTensor, species_name: str, save_path: Path, title: str = ""
) -> None:
    """Plots a NamedTensor where all features are from the same species."""
    num_plots = len(nt.feature_names)
    nrows = int(math.sqrt(num_plots))
    ncols = math.ceil(num_plots / nrows)
    subplot_kw = {"projection": PlateCarree()}
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), subplot_kw=subplot_kw
    )
    axs = axs.flatten()
    vmin, vmax = get_vmin_vmax(species_name)

    for i, ax in enumerate(axs):
        name = nt.feature_names[i]
        plot_kwargs = {"cmap": CMAP, "extent": EXTENT}
        if name not in ["argmin", "argmax", "skew", "kurtosis"]:
            plot_kwargs["vmin"] = vmin
            plot_kwargs["vmax"] = vmax
        ax.imshow(nt[name][0], **plot_kwargs)
        format_axis(ax, name)

    fig.suptitle(title, size=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
