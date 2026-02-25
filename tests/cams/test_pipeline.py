

from scripts.fit_and_val import fit_and_val
from tests.conftest import CAMSDataModuleTest


def test_full_pipeline() -> None:
    """Test the full project life cycle.
    Test the cli interface entry points.

    - Training.
    - Retrain a checkpoint. X TODO
    - Checkpoint writting. X TODO
    - Predict from checkpoint. X TODO
    - Export to onnx. X TODO
    - Predict from onnx. X TODO
    """

    # Train with a test config.
    fit_and_val(
        datamodule_cls=CAMSDataModuleTest,
        args=["--config", "configs/test_config.yaml"],
    )


if __name__ == "__main__":
    test_full_pipeline()
