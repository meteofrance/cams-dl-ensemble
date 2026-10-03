"""Plots the GPU VRAM usage of several mfai models versus their input channels.

For each model in a fixed selection, ``in_channels`` grows until the GPU runs
out of memory. At each value the model is built, moved to GPU and a full forward
+ backward pass with an AdamW optimizer step is run on a fixed-size input
tensor. The model output has ``in_channels // nb_models`` channels (the number
of species, levels and leadtimes to predict). The peak VRAM (maximum memory
allocated by the PyTorch caching allocator) is recorded, and the largest
``in_channels`` that fits on the card is the end of the curve. One curve per
model is plotted on a single PNG.

The models swept are: DeepLabV3Plus, HalfUNet, Segformer, SwinUNetR, UNet and
UNetRPP.

usage: vram_usage_plots.py [-h] [--save_dir SAVE_DIR] [--batch_size N]
                           [--nb_models N] [--nb_species N]
                           [--nb_levels N] [--height N] [--width N]

options:
  --save_dir SAVE_DIR       Directory where the plot will be saved
  --batch_size N            Fixed batch size used for every measurement
  --nb_models Size of the model dimension
  --nb_species Size of the species dimension
  --nb_levels Size of the levels dimension
  --height N                Fixed input height used for every measurement
  --width N                 Fixed input width used for every measurement
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch.nn as nn
from mfai.pytorch.models.deeplabv3 import DeepLabV3Plus, DeepLabV3PlusSettings
from mfai.pytorch.models.half_unet import HalfUNet, HalfUNetSettings
from mfai.pytorch.models.segformer import Segformer, SegformerSettings
from mfai.pytorch.models.swinunetr import SwinUNetR, SwinUNetRSettings
from mfai.pytorch.models.unet import UNet, UNetSettings
from mfai.pytorch.models.unetrpp import UNetRPP, UNetRPPSettings

ModelBuilder = Callable[[int, int, tuple[int, int]], nn.Module]


def _deeplabv3_builder(
    in_channels: int, out_channels: int, input_shape: tuple[int, int]
) -> DeepLabV3Plus:
    """Build a DeepLabV3Plus without pretrained encoder weights.

    The pretrained weights are disabled to keep the measurement deterministic
    and usable on nodes without internet access. Autopadding is enabled so that
    any input spatial shape is accepted.

    Args:
        in_channels: Number of input channels of the model.
        out_channels: Number of output channels of the model.
        input_shape: Spatial ``(height, width)`` shape of the model input.

    Returns:
        DeepLabV3Plus: A freshly built model instance.
    """
    settings = DeepLabV3PlusSettings(encoder_weights=False, autopad_enabled=True)
    return DeepLabV3Plus(
        in_channels=in_channels,
        out_channels=out_channels,
        input_shape=input_shape,
        settings=settings,
    )


def _settings_builder(model_kls: type[nn.Module], settings_kls: type) -> ModelBuilder:
    """Return a builder for a model using a settings dataclass.

    The builder enables autopadding so that any input spatial shape is accepted,
    which some architectures need (their spatial dimensions must be multiples of
    a downsampling factor).

    Args:
        model_kls: Model class sharing the settings-based constructor.
        settings_kls: Dataclass type used to configure the model.

    Returns:
        ModelBuilder: A builder taking the number of input channels, the number
            of output channels and the spatial input shape.
    """

    def _build(
        in_channels: int, out_channels: int, input_shape: tuple[int, int]
    ) -> nn.Module:
        settings = settings_kls(autopad_enabled=True)
        return model_kls(
            in_channels=in_channels,
            out_channels=out_channels,
            input_shape=input_shape,
            settings=settings,
        )

    return _build


MODELS: dict[str, ModelBuilder] = {
    "DeepLabV3Plus": _deeplabv3_builder,
    "HalfUNet": _settings_builder(HalfUNet, HalfUNetSettings),
    "Segformer": _settings_builder(Segformer, SegformerSettings),
    "SwinUNetR": _settings_builder(SwinUNetR, SwinUNetRSettings),
    "UNet": _settings_builder(UNet, UNetSettings),
    "UNetRPP": _settings_builder(UNetRPP, UNetRPPSettings),
}


def measure_vram(
    builder: ModelBuilder,
    *,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    height: int,
    width: int,
) -> int:
    """Peak VRAM (bytes) of a full training step of a model.

    The model and an AdamW optimizer are built, moved to GPU and a random input
    tensor of fixed spatial size is used in train mode. A forward pass, backward
    pass and optimizer step are run, covering the weights, activations,
    gradients, batch data and optimizer state memory. The PyTorch caching
    allocator peak is reset before the pass and read after a synchronization.

    Args:
        builder: Callable building the model for a given number of channels and
            input shape.
        in_channels: Number of input channels of the model.
        out_channels: Number of output channels of the model.
        batch_size: Batch size of the synthetic input tensor.
        height: Spatial height of the synthetic input tensor.
        width: Spatial width of the synthetic input tensor.

    Returns:
        int: Peak number of bytes allocated on the GPU during the pass.
    """
    model = builder(in_channels, out_channels, (height, width)).train()
    model.cuda()

    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    y = torch.randn(batch_size, out_channels, height, width, device="cuda")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    model.cpu()
    torch.cuda.empty_cache()
    return peak


def measure_vram_curve(
    builder: ModelBuilder,
    *,
    nb_models: int,
    nb_species: int,
    nb_levels: int,
    batch_size: int,
    height: int,
    width: int,
) -> tuple[list[int], list[int]]:
    """Measure peak VRAM for increasing leadtimes until the GPU is full.

    ``nb_leadtimes`` starts at 1 and increases by 1 at each step. A single
    leadtime contributes ``channels_per_leadtime = nb_models * nb_species *
    nb_levels`` input channels, so ``in_channels = channels_per_leadtime *
    nb_leadtimes`` and ``out_channels = nb_species * nb_levels * nb_leadtimes``.
    The search stops as soon as a training step runs out of GPU memory; the last
    value that fit defines the end of the curve.

    Args:
        builder: Callable building the model for a given number of channels.
        nb_models: Size of the model dimension of the input data.
        nb_species: Size of the species dimension of the input data.
        nb_levels: Size of the levels dimension of the input data.
        batch_size: Fixed batch size used for every measurement.
        height: Fixed input height used for every measurement.
        width: Fixed input width used for every measurement.

    Returns:
        tuple[list[int], list[int]]: The successful ``in_channels`` values and
            the associated peak VRAM during a full training step in bytes.
    """
    channels_per_leadtime = nb_models * nb_species * nb_levels
    channels: list[int] = []
    vram_bytes: list[int] = []
    nb_leadtimes = 1
    while True:
        in_channels = channels_per_leadtime * nb_leadtimes
        out_channels = nb_species * nb_levels * nb_leadtimes
        try:
            peak = measure_vram(
                builder,
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch_size,
                height=height,
                width=width,
            )
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError):
            # Out-of-memory during backward can surface as a different CUDA
            # error (e.g. cudaErrorInvalidValue) instead of a clean
            # OutOfMemoryError, depending on where the failed allocation
            # happens. Both mean the GPU ran out of memory, i.e. the end of
            # the curve.
            torch.cuda.empty_cache()
            break
        channels.append(in_channels)
        vram_bytes.append(peak)
        print(f"  {in_channels} channels: {peak / 1024**2:.0f} MB", file=sys.stderr)
        nb_leadtimes += 1
    return channels, vram_bytes


def plot_vram_curves(
    measurements: dict[str, tuple[list[int], list[int]]],
    save_path: Path,
    *,
    batch_size: int,
    nb_models: int,
    nb_species: int,
    nb_levels: int,
    height: int,
    width: int,
) -> None:
    """Plot the VRAM usage curves of every model and save them in one image.

    Args:
        measurements: Mapping of model name to a ``(channels, vram_bytes)`` pair.
        save_path: Path where to save the PNG plot.
        batch_size: Batch size used for the measurements.
        nb_models: Size of the model dimension of the input data.
        nb_species: Size of the species dimension of the input data.
        nb_levels: Size of the levels dimension of the input data.
        height: Input height used for the measurements.
        width: Input width used for the measurements.
    """
    channels_per_leadtime = nb_models * nb_species * nb_levels

    all_channels = sorted(
        {c for channels, _ in measurements.values() for c in channels}
    )
    tick_step = max(1, len(all_channels) // 8)
    tick_channels = all_channels[::tick_step]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(
        "Peak VRAM usage for a full training step vs number of leadtimes\n"
        f"(batch={batch_size}, input_shape={height}x{width}, "
        f"{nb_models} models, {nb_species} species, {nb_levels} level)"
    )
    for name, (channels, vram_bytes) in measurements.items():
        vram_gb = [b / (1024**3) for b in vram_bytes]
        ax.plot(channels, vram_gb, marker="o", label=name)
    ax.set_xlabel("Number of input channels")
    ax.set_ylabel("Peak VRAM usage (GB)")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(tick_channels[0], tick_channels[-1])
    ax.set_xticks(tick_channels)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(tick_channels)
    ax_top.set_xticklabels([str(int(c / channels_per_leadtime)) for c in tick_channels])
    ax_top.set_xlabel(
        "Number of leadtimes "
        f"({nb_models} models, {nb_species} species, {nb_levels} level)"
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Entry point of the VRAM plotting script."""
    parser = argparse.ArgumentParser(description="Plots model VRAM usage.")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("output/vram"),
        dest="save_dir",
        help="Directory where the plot will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        dest="batch_size",
        help="Fixed batch size used for every measurement (must be > 1 for "
        "BatchNorm-based models)",
    )
    parser.add_argument(
        "--nb_models",
        type=int,
        default=11,
        dest="nb_models",
        help="Size of the model dimension",
    )
    parser.add_argument(
        "--nb_species",
        type=int,
        default=6,
        dest="nb_species",
        help="Size of the species dimension",
    )
    parser.add_argument(
        "--nb_levels",
        type=int,
        default=1,
        dest="nb_levels",
        help="Size of the levels dimension",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        dest="height",
        help="Fixed input height used for every measurement",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        dest="width",
        help="Fixed input width used for every measurement",
    )

    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. This script requires a GPU.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    measurements: dict[str, tuple[list[int], list[int]]] = {}
    for name, builder in MODELS.items():
        print(f"Measuring {name}...", file=sys.stderr)
        measurements[name] = measure_vram_curve(
            builder,
            nb_models=args.nb_models,
            nb_species=args.nb_species,
            nb_levels=args.nb_levels,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )
    save_path = args.save_dir / (
        f"{args.batch_size}_batch_size-"
        f"{args.nb_models}_models-"
        f"{args.nb_species}_species-"
        f"{args.nb_levels}_levels-"
        f"{args.height}x{args.width}_input_shape.png"
    )
    plot_vram_curves(
        measurements=measurements,
        save_path=save_path,
        batch_size=args.batch_size,
        nb_models=args.nb_models,
        nb_species=args.nb_species,
        nb_levels=args.nb_levels,
        height=args.height,
        width=args.width,
    )
    print(f"Plot saved at {save_path}")


if __name__ == "__main__":
    main()
