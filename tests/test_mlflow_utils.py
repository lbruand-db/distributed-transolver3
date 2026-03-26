"""Tests for transolver3.mlflow_utils — MLflow experiment tracking utilities."""

import tempfile

import pytest
import torch

mlflow = pytest.importorskip("mlflow", reason="mlflow required for mlflow_utils tests")

from transolver3.mlflow_utils import (  # noqa: E402
    log_training_run,
    log_model_with_signature,
    log_normalization_artifacts,
    load_normalization_artifacts,
    log_prediction_visualization,
)
from transolver3.normalizer import InputNormalizer, TargetNormalizer  # noqa: E402
from transolver3 import Transolver3  # noqa: E402


@pytest.fixture
def small_model():
    return Transolver3(space_dim=3, n_layers=2, n_hidden=64, n_head=4, out_dim=1, slice_num=8)


@pytest.fixture
def mlflow_run():
    """Start and yield an MLflow run, then clean up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}")
        mlflow.set_experiment("test-transolver3")
        with mlflow.start_run() as run:
            yield run


def test_log_training_run(small_model, mlflow_run):
    """log_training_run logs params and model info."""
    config = {"n_layers": 2, "n_hidden": 64, "lr": 1e-3}
    log_training_run(small_model, config)

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(mlflow_run.info.run_id).data
    assert run_data.params["n_layers"] == "2"
    assert run_data.params["lr"] == "0.001"
    assert "model_parameters" in run_data.params


def test_log_training_run_no_active_run(small_model):
    """log_training_run raises ValueError without an active run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}")
        with pytest.raises(ValueError, match="No active MLflow run"):
            log_training_run(small_model, {"lr": 1e-3})


def test_log_model_with_signature(small_model, mlflow_run):
    """log_model_with_signature logs model with inferred signature."""
    sample_input = torch.randn(1, 50, 3)
    info = log_model_with_signature(small_model, sample_input)
    assert info is not None
    assert info.artifact_path == "transolver3"


def test_normalization_artifacts_roundtrip(mlflow_run):
    """Normalizer state_dicts survive save/load via MLflow artifacts."""
    input_norm = InputNormalizer(scale=1000.0, per_sample=False)
    input_norm.fit(torch.randn(3, 100, 3))

    target_norm = TargetNormalizer(out_dim=4)
    target_norm.fit(torch.randn(3, 100, 4))

    log_normalization_artifacts(input_norm=input_norm, target_norm=target_norm)

    loaded_input, loaded_target = load_normalization_artifacts(mlflow_run.info.run_id)

    assert loaded_input is not None
    assert loaded_target is not None
    assert torch.allclose(input_norm.data_min, loaded_input.data_min, atol=1e-6)
    assert torch.allclose(target_norm.mean, loaded_target.mean, atol=1e-6)
    assert torch.allclose(target_norm.std, loaded_target.std, atol=1e-6)


def test_log_prediction_visualization(mlflow_run):
    """log_prediction_visualization creates and logs a PNG artifact."""
    pytest.importorskip("matplotlib", reason="matplotlib required")

    pred = torch.randn(200, 1)
    target = torch.randn(200, 1)
    coords = torch.randn(200, 3)

    log_prediction_visualization(pred, target, coords, step=42, title="Test Viz")

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(mlflow_run.info.run_id, path="predictions")
    artifact_names = [a.path for a in artifacts]
    assert any("prediction_step_42" in name for name in artifact_names)
