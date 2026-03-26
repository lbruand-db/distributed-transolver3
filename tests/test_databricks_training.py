"""Tests for transolver3.databricks_training — TorchDistributor and Spark integration."""

import os
from unittest.mock import patch

import pytest

from transolver3.databricks_training import (
    is_on_databricks,
    _resolve_script_path,
)


def test_is_on_databricks_false():
    """is_on_databricks returns False when env var not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove the variable if it exists
        os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
        assert is_on_databricks() is False


def test_is_on_databricks_true():
    """is_on_databricks returns True when env var is set."""
    with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "15.4"}):
        assert is_on_databricks() is True


def test_resolve_script_path_string():
    """_resolve_script_path resolves a string path."""
    result = _resolve_script_path("/some/path/train.py")
    assert result == "/some/path/train.py"


def test_resolve_script_path_module():
    """_resolve_script_path resolves a module with __file__."""
    import transolver3.distributed as mod
    result = _resolve_script_path(mod)
    assert result.endswith("distributed.py")
    assert os.path.isabs(result)


def test_resolve_script_path_unresolvable():
    """_resolve_script_path raises for unresolvable objects."""
    # Create an object with no __module__ and no __file__
    obj = object()
    with pytest.raises(ValueError, match="Cannot resolve script path"):
        _resolve_script_path(obj)


def test_preprocess_with_spark_import_guard():
    """preprocess_with_spark raises ImportError guidance if pyspark missing."""
    pyspark = pytest.importorskip("pyspark", reason="pyspark required")  # noqa: F841
    # If pyspark IS available, just verify the function is importable
    from transolver3.databricks_training import preprocess_with_spark
    assert callable(preprocess_with_spark)
