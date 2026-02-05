import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from cams.sample import Sample
from mfai.pytorch.namedtensor import NamedTensor
import torch

# Constants
MOSAIC: list[list[str]] = [
    ["MATCH", "MINNI", "CHIMERE", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["MOCAGE", "MONARCH", "EURADIM", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["EMEP", "GEMAQ", "SILAM", "DEHM", "LOTOS", "TARGET", "TARGET", "TARGET"],
]
UNITS = {
    "O3": "Ozone (µg/m3)"
}
CMAP = "terrain"  # seismic, tab20c, terrain


def plot_sample(sample: Sample, save_path:Path, species_name: str = "O3") -> None:
    """Plots a sample's input and target data for one parameter only.

    Args:
        sample: Sample we want to plot.
        species_name: The species name to plot.
    """
    x, y = sample.input_data, sample.target_data
    nt = NamedTensor.concat([x, y])
    median = torch.median(x.tensor, dim=0).values
    vmin = torch.min(nt.tensor).item()
    vmax = torch.max(nt.tensor).item()

    # Create the different subfigures
    scale = 2.5
    fig, axs = plt.subplot_mosaic(  # type: ignore
        mosaic=MOSAIC,  # type: ignore[reportArgumentType]
        layout="constrained",
        figsize=(8 * scale, 3.2 * scale),
    )

    # Render the 11 models to their corresponding plot cell
    cell_name: str
    ax: Axes
    for cell_name, ax in axs.items():
        if cell_name in ["MEDIAN", "TARGET"]:
            continue
        ax.imshow(x[cell_name][0], cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(cell_name)
        ax.set(xticklabels=[], yticklabels=[])
        ax.tick_params(bottom=False, left=False)
        ax.set_aspect(1.8)

    # Render the median to its corresponding plot cell
    axs["MEDIAN"].imshow(median, cmap=CMAP, vmin=vmin, vmax=vmax)
    axs["MEDIAN"].set_title("Median Ensemble = Baseline")
    axs["MEDIAN"].set(xticklabels=[], yticklabels=[])
    axs["MEDIAN"].tick_params(bottom=False, left=False)
    axs["MEDIAN"].set_aspect(1.8)

    # Render the target to its corresponding plot cell
    img = axs["TARGET"].imshow(y["Analysis"][0], cmap=CMAP, vmin=vmin, vmax=vmax)
    axs["TARGET"].set_title("Analysis = Target")
    axs["TARGET"].set(xticklabels=[], yticklabels=[])
    axs["TARGET"].tick_params(bottom=False, left=False)
    axs["TARGET"].set_aspect(1.8)

    # Add Colorbar
    cbar = fig.colorbar(img, ax=axs["TARGET"])
    cbar.set_label(UNITS[species_name], size=13)

    # Add the plot's title
    date = sample.date_run
    fig.suptitle(f"{species_name} - {date.strftime(r'%Y-%m-%d %Hh00')}", size=16)

    plt.savefig(save_path)


if __name__=="__main__":
    sample = Sample(dt.datetime(2022, 7, 22, 15), 6)
    plot_sample(sample, Path("test.png"))

    # TODO : try to add map