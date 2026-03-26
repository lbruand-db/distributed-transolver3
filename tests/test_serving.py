"""Tests for transolver3.serving — model serving with MLflow pyfunc."""

import json
import os
import tempfile

import pytest
import torch
import numpy as np

from transolver3 import Transolver3
from transolver3.normalizer import InputNormalizer, TargetNormalizer


class MockContext:
    """Mock MLflow PythonModelContext with artifacts."""

    def __init__(self, artifacts):
        self.artifacts = artifacts


def _save_model_artifacts(model, config, tmpdir, normalizers=None):
    """Helper to save model artifacts to a temp directory."""
    model_state_path = os.path.join(tmpdir, "model_state.pt")
    torch.save(model.state_dict(), model_state_path)

    config_path = os.path.join(tmpdir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    artifacts = {
        "model_state": model_state_path,
        "model_config": config_path,
    }

    if normalizers and "input" in normalizers:
        path = os.path.join(tmpdir, "input_normalizer.pt")
        torch.save(normalizers["input"].state_dict(), path)
        artifacts["input_normalizer"] = path

    if normalizers and "target" in normalizers:
        path = os.path.join(tmpdir, "target_normalizer.pt")
        torch.save(normalizers["target"].state_dict(), path)
        artifacts["target_normalizer"] = path

    return artifacts


@pytest.fixture
def model_config():
    return {
        "space_dim": 3,
        "n_layers": 2,
        "n_hidden": 64,
        "n_head": 4,
        "fun_dim": 0,
        "out_dim": 1,
        "slice_num": 8,
    }


@pytest.fixture
def small_model(model_config):
    return Transolver3(**model_config)


def test_pyfunc_predict_numpy(small_model, model_config):
    """TransolverPyfunc predict works with numpy array input."""
    from transolver3.serving import TransolverPyfunc

    pyfunc = TransolverPyfunc()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = _save_model_artifacts(small_model, model_config, tmpdir)
        context = MockContext(artifacts)
        pyfunc.load_context(context)

    coords = np.random.randn(1, 50, 3).astype(np.float32)
    result = pyfunc.predict(None, {"coordinates": coords})

    assert result.shape == (1, 50, 1)
    assert result.dtype == np.float32


def test_pyfunc_predict_with_normalizers(small_model, model_config):
    """TransolverPyfunc handles input/target normalization."""
    from transolver3.serving import TransolverPyfunc

    input_norm = InputNormalizer(scale=1000.0, per_sample=False)
    input_norm.fit(torch.randn(3, 100, 3))

    target_norm = TargetNormalizer(out_dim=1)
    target_norm.fit(torch.randn(3, 100, 1))

    pyfunc = TransolverPyfunc()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = _save_model_artifacts(
            small_model, model_config, tmpdir,
            normalizers={"input": input_norm, "target": target_norm},
        )
        context = MockContext(artifacts)
        pyfunc.load_context(context)

    assert pyfunc.input_norm is not None
    assert pyfunc.target_norm is not None

    coords = np.random.randn(1, 50, 3).astype(np.float32)
    result = pyfunc.predict(None, {"coordinates": coords})

    assert result.shape == (1, 50, 1)


def test_pyfunc_predict_2d_input(small_model, model_config):
    """TransolverPyfunc handles 2D (N, 3) input by adding batch dim."""
    from transolver3.serving import TransolverPyfunc

    pyfunc = TransolverPyfunc()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = _save_model_artifacts(small_model, model_config, tmpdir)
        context = MockContext(artifacts)
        pyfunc.load_context(context)

    coords = np.random.randn(50, 3).astype(np.float32)
    result = pyfunc.predict(None, {"coordinates": coords})

    assert result.shape == (1, 50, 1)


def test_register_serving_model():
    """register_serving_model requires mlflow (import guard test)."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow required")  # noqa: F841
    from transolver3.serving import register_serving_model
    assert callable(register_serving_model)
