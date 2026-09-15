from typing import cast

import pytest
import torch
from mfai.pytorch.models.half_unet import HalfUNet
from mfai.pytorch.namedtensor import NamedTensor
from torchmetrics import MetricCollection

from cams.plmodule import CAMSLightningModule


@pytest.fixture
def model() -> HalfUNet:
    """A small HalfUNet model usable to instantiate the lightning module."""
    return HalfUNet(in_channels=1, out_channels=1, input_shape=(64, 64))


@pytest.fixture
def loss() -> torch.nn.Module:
    """Loss function used to instantiate the lightning module."""
    return torch.nn.MSELoss()


def instantiate(
    model: HalfUNet,
    loss: torch.nn.Module,
    *,
    lead_times: list[object],
    species: list[object],
    levels: list[object],
    val_leadtimes: list[object],
) -> CAMSLightningModule:
    """Create a lightning module with the given data selection arguments."""
    return CAMSLightningModule(
        model=model,
        loss=loss,
        lead_times=cast(list, lead_times),
        species=cast(list, species),
        levels=cast(list, levels),
        val_leadtimes=cast(list, val_leadtimes),
    )


def metric_names(metrics: MetricCollection) -> list[str]:
    """Return the flattened metric names of a collection as plain strings."""
    return [str(name) for name in metrics.keys()]


def test_init_val_leadtimes_not_in_leadtimes(
    model: HalfUNet, loss: torch.nn.Module
) -> None:
    """Raises ValueError when a validation leadtime is not in the selected ones."""
    with pytest.raises(ValueError):
        instantiate(
            model,
            loss,
            lead_times=[15],
            species=["O3"],
            levels=[0],
            val_leadtimes=[15, 24],
        )


def test_init_val_leadtimes_empty(model: HalfUNet, loss: torch.nn.Module) -> None:
    """Accepts an empty validation leadtime list, a subset of the selected ones."""
    module = instantiate(
        model,
        loss,
        lead_times=[15, 24],
        species=["O3"],
        levels=[0],
        val_leadtimes=[],
    )
    assert module.val_leadtimes == []


def test_init_requires_data_selection_arguments(
    model: HalfUNet, loss: torch.nn.Module
) -> None:
    """Raises TypeError when lead_times, species or levels are missing."""
    with pytest.raises(TypeError):
        CAMSLightningModule(model=model, loss=loss)  # type: ignore[call-arg]


def test_get_metrics_names(model: HalfUNet, loss: torch.nn.Module) -> None:
    """Builds one metric collection per species and validation leadtime."""
    module = instantiate(
        model,
        loss,
        lead_times=[3, 15, 24],
        species=["O3", "NO2"],
        levels=[0],
        val_leadtimes=[15],
    )
    expected = {
        "O3-15h-0m/Accuracy_120",
        "O3-15h-0m/F1Score_120",
        "O3-15h-0m/FalseAlarmRate_120",
        "O3-15h-0m/FalsePositiveRate_120",
        "NO2-15h-0m/Accuracy_120",
        "NO2-15h-0m/F1Score_120",
        "NO2-15h-0m/FalseAlarmRate_120",
        "NO2-15h-0m/FalsePositiveRate_120",
    }
    names = set(metric_names(module.metrics))
    assert expected <= names


def test_get_metrics_skips_non_selected_leadtimes(
    model: HalfUNet, loss: torch.nn.Module
) -> None:
    """Only validation leadtimes selected appear in the metrics, not lead_times."""
    module = instantiate(
        model,
        loss,
        lead_times=[3, 15, 24],
        species=["O3"],
        levels=[0],
        val_leadtimes=[15],
    )
    names = set(metric_names(module.metrics))
    assert "O3-3h-0m/Accuracy_120" not in names
    assert "O3-24h-0m/Accuracy_120" not in names


def test_get_metrics_all_species_cross_product(
    model: HalfUNet, loss: torch.nn.Module
) -> None:
    """Number of metric collections is the species/leadtime cross product."""
    module = instantiate(
        model,
        loss,
        lead_times=[15, 24],
        species=["O3", "NO2"],
        levels=[0],
        val_leadtimes=[15, 24],
    )
    names = metric_names(module.metrics)
    accuracy_keys = [name for name in names if name.endswith("/Accuracy_120")]
    assert len(accuracy_keys) == 4


def test_get_metrics_compute_groups_do_not_share_state(
    model: HalfUNet, loss: torch.nn.Module
) -> None:
    """Metrics across species/leadtimes keep independent states and values.

    With the default ``compute_groups=True``, torchmetrics merges metrics whose
    binary states coincide into a single compute group, sharing their state by
    reference and only updating the group's first member. That silently forces
    every merged species/leadtime metric to report the same value. The metrics
    must be built with ``compute_groups=False`` to keep them independent.
    """

    module = instantiate(
        model,
        loss,
        lead_times=[3, 9],
        species=["O3", "NO2"],
        levels=[0],
        val_leadtimes=[3, 9],
    )
    metrics = module.get_metrics()
    assert metrics.compute_groups == {}

    features = [
        "TARGET - O3 - +3h - 0m",
        "TARGET - O3 - +9h - 0m",
        "TARGET - NO2 - +3h - 0m",
        "TARGET - NO2 - +9h - 0m",
    ]
    names = ["batch", "features", "x", "y"]

    def _update(
        preds: list[float],
        targets: list[float],
        subsample: int = 4,
    ) -> None:
        """Feed ``subsample`` batches of identical spatially uniform data."""
        spatial = torch.Size([subsample, 4, 4])

        def _build(values: list[float]) -> torch.Tensor:
            features_t = torch.stack(
                [torch.full(spatial, value, dtype=torch.float32) for value in values]
            )
            return features_t.permute(1, 0, 2, 3).contiguous()

        metrics.update(
            NamedTensor(_build(preds), names=names, feature_names=features),
            NamedTensor(_build(targets), names=names, feature_names=features),
        )

    # Phase 1: every feature has identical binary state. Under compute_groups=True
    # this is what makes torchmetrics merge the metrics into one compute group.
    _update([200.0] * 4, [200.0] * 4, subsample=2)
    # Phase 2: NO2 targets stay high while O3 targets drop, so NO2 accuracy must
    # not be equal to O3 accuracy.
    _update([200.0] * 4, [0.0, 0.0, 200.0, 200.0], subsample=2)

    output = metrics.compute()
    o3_acc = float(output["O3-3h-0m/Accuracy_120"])
    no2_acc = float(output["NO2-3h-0m/Accuracy_120"])
    assert no2_acc == 1.0
    assert o3_acc != no2_acc
