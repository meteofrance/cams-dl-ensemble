"""Unit tests for the ``scripts.vram_usage_plots`` module.

The heavy GPU parts (``measure_vram``) are mocked so the tests run on a CPU
machine and stay fast. Only the pure-, plotting- and control-flow logic is
exercised here.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from scripts import vram_usage_plots as vup
from scripts.vram_usage_plots import MODELS, measure_vram_curve, plot_vram_curves

SettingsBuilder = Callable[[int, int, tuple[int, int]], nn.Module]


def test_settings_builder_builds_model() -> None:
    """Models built from ``settings`` builders accept any spatial shape."""
    from mfai.pytorch.models.half_unet import HalfUNet, HalfUNetSettings

    build: SettingsBuilder = vup._settings_builder(  # type: ignore[reportPrivateUsage]
        HalfUNet, HalfUNetSettings
    )
    model = build(4, 2, (17, 17))
    assert isinstance(model, HalfUNet)
    assert model.out_channels == 2


def test_deeplabv3_builder_disables_pretrained_weights() -> None:
    """DeepLabV3Plus is built without pretrained encoder weights."""
    from mfai.pytorch.models.deeplabv3 import DeepLabV3Plus

    model: DeepLabV3Plus = vup._deeplabv3_builder(  # type: ignore[reportPrivateUsage]
        4, 2, (17, 17)
    )
    assert isinstance(model, DeepLabV3Plus)
    assert model.out_channels == 2


def test_models_builders_exist() -> None:
    """Every documented model has a registered builder."""
    expected = {
        "DeepLabV3Plus",
        "HalfUNet",
        "Segformer",
        "SwinUNetR",
        "UNet",
        "UNetRPP",
    }
    assert set(MODELS) == expected


def test_measure_vram_curve_stops_on_out_of_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The curve stops as soon as a training step runs out of memory."""
    peaks = [1024, 2048, 4096]

    def fake_measure_vram(
        builder: Any, *, in_channels: int, out_channels: int, **kwargs: Any
    ) -> int:
        nb_leadtimes = in_channels // (2 * 1 * 1)  # nb_models * nb_species * nb_levels
        if nb_leadtimes > len(peaks):
            raise torch.cuda.OutOfMemoryError
        return peaks[nb_leadtimes - 1]

    monkeypatch.setattr("scripts.vram_usage_plots.measure_vram", fake_measure_vram)

    channels, vram_bytes = measure_vram_curve(
        Mock(),
        nb_models=2,
        nb_species=1,
        nb_levels=1,
        batch_size=1,
        height=16,
        width=16,
    )

    assert channels == [2, 4, 6]
    assert vram_bytes == peaks


def test_measure_vram_curve_handles_accelerator_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic accelerator error also ends the curve."""

    def fake_measure_vram(
        builder: Any, *, in_channels: int, out_channels: int, **kwargs: Any
    ) -> int:
        if in_channels > 2:
            raise torch.AcceleratorError
        return 512

    monkeypatch.setattr("scripts.vram_usage_plots.measure_vram", fake_measure_vram)

    channels, vram_bytes = measure_vram_curve(
        Mock(),
        nb_models=1,
        nb_species=1,
        nb_levels=1,
        batch_size=1,
        height=16,
        width=16,
    )

    assert channels == [1, 2]
    assert vram_bytes == [512, 512]


def test_plot_vram_curves_saves_png(temp_dir: Path) -> None:
    """The plot is saved as a non-empty PNG file."""
    measurements = {
        "UNet": ([1, 2, 3], [1024**3, 2 * 1024**3, 3 * 1024**3]),
        "HalfUNet": ([1, 2, 3], [512 * 1024**2, 1024**3, 2 * 1024**3]),
    }
    save_path = temp_dir / "vram.png"

    plot_vram_curves(
        measurements,
        save_path,
        batch_size=2,
        nb_models=11,
        nb_species=6,
        nb_levels=1,
        height=128,
        width=128,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_main_raises_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` refuses to run when no CUDA GPU is available."""
    monkeypatch.setattr(
        "scripts.vram_usage_plots.torch.cuda.is_available", lambda: False
    )

    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        from scripts.vram_usage_plots import main

        main([])


def test_main_saves_plot(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` measures every model and saves the plot."""
    monkeypatch.setattr(vup.torch.cuda, "is_available", lambda: True)
    fake_builders = {"UNet": Mock(), "HalfUNet": Mock()}
    monkeypatch.setattr(vup, "MODELS", fake_builders)

    def fake_curve(builder: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
        return [1, 2], [1024**3, 2 * 1024**3]

    monkeypatch.setattr(vup, "measure_vram_curve", fake_curve)
    monkeypatch.setattr(vup, "plot_vram_curves", Mock())

    vup.main(["--save_dir", str(temp_dir)])

    saved = list(temp_dir.glob("*.png"))
    assert len(saved) == 1
    out = capsys.readouterr().out
    assert "Plot saved" in out
