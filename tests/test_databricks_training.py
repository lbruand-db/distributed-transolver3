# Copyright 2026 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for transolver3.databricks_training — TorchDistributor and Spark integration."""

import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from transolver3.databricks_training import (
    is_on_databricks,
    _resolve_script_path,
    _propagate_databricks_auth_env,
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


# --- _propagate_databricks_auth_env tests ---


def _install_fake_module(name, attrs):
    """Inject a fake module into sys.modules for the duration of a test.

    Works even when the real package isn't installed. Handles dotted names
    by creating parent modules as needed.
    """
    import sys

    parts = name.split(".")
    for i in range(len(parts)):
        parent = ".".join(parts[: i + 1])
        if parent not in sys.modules:
            sys.modules[parent] = MagicMock()
    # Set the final module's attributes
    mod = sys.modules[name]
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture(autouse=True)
def _clean_auth_env():
    """Remove auth env vars before and after each test."""
    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        os.environ.pop(key, None)
    yield
    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        os.environ.pop(key, None)


class TestPropagateAuth:
    """Tests for _propagate_databricks_auth_env."""

    def test_skips_when_env_already_set(self):
        """Does nothing if DATABRICKS_HOST and DATABRICKS_TOKEN are already set."""
        os.environ["DATABRICKS_HOST"] = "https://existing.cloud.databricks.com"
        os.environ["DATABRICKS_TOKEN"] = "existing-token"
        _propagate_databricks_auth_env()
        assert os.environ["DATABRICKS_HOST"] == "https://existing.cloud.databricks.com"
        assert os.environ["DATABRICKS_TOKEN"] == "existing-token"

    def test_extracts_from_mlflow_creds(self):
        """Extracts host/token from MLflow's get_databricks_host_creds."""
        import sys

        mock_creds = SimpleNamespace(host="https://test.cloud.databricks.com", token="mlflow-token-123")
        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("mlflow"):
                saved[mod_name] = sys.modules.pop(mod_name)
        try:
            _install_fake_module(
                "mlflow.utils.databricks_utils",
                {"get_databricks_host_creds": lambda: mock_creds},
            )
            _propagate_databricks_auth_env()
            assert os.environ["DATABRICKS_HOST"] == "https://test.cloud.databricks.com"
            assert os.environ["DATABRICKS_TOKEN"] == "mlflow-token-123"
        finally:
            for mod_name in list(sys.modules):
                if mod_name.startswith("mlflow"):
                    del sys.modules[mod_name]
            sys.modules.update(saved)

    def test_falls_back_to_databricks_sdk(self):
        """Falls back to Databricks SDK when MLflow creds fail."""
        import sys

        mock_config = SimpleNamespace(host="https://sdk.cloud.databricks.com", token="sdk-token-456")
        mock_client = MagicMock()
        mock_client.config = mock_config

        saved_mlflow = {}
        saved_db = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("mlflow"):
                saved_mlflow[mod_name] = sys.modules.pop(mod_name)
            if mod_name.startswith("databricks"):
                saved_db[mod_name] = sys.modules.pop(mod_name)
        try:
            # MLflow fails
            _install_fake_module(
                "mlflow.utils.databricks_utils",
                {"get_databricks_host_creds": MagicMock(side_effect=Exception("no auth"))},
            )
            # SDK succeeds
            _install_fake_module(
                "databricks.sdk",
                {"WorkspaceClient": lambda: mock_client},
            )
            _propagate_databricks_auth_env()
            assert os.environ["DATABRICKS_HOST"] == "https://sdk.cloud.databricks.com"
            assert os.environ["DATABRICKS_TOKEN"] == "sdk-token-456"
        finally:
            for mod_name in list(sys.modules):
                if mod_name.startswith("mlflow") or mod_name.startswith("databricks"):
                    del sys.modules[mod_name]
            sys.modules.update(saved_mlflow)
            sys.modules.update(saved_db)

    def test_no_op_when_all_approaches_fail(self):
        """Does nothing if all credential sources fail."""
        import sys

        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("mlflow") or mod_name.startswith("databricks"):
                saved[mod_name] = sys.modules.pop(mod_name)
        try:
            _install_fake_module(
                "mlflow.utils.databricks_utils",
                {"get_databricks_host_creds": MagicMock(side_effect=Exception("nope"))},
            )
            _install_fake_module(
                "databricks.sdk",
                {"WorkspaceClient": MagicMock(side_effect=Exception("nope"))},
            )
            _propagate_databricks_auth_env()
            assert "DATABRICKS_HOST" not in os.environ
            assert "DATABRICKS_TOKEN" not in os.environ
        finally:
            for mod_name in list(sys.modules):
                if mod_name.startswith("mlflow") or mod_name.startswith("databricks"):
                    del sys.modules[mod_name]
            sys.modules.update(saved)

    def test_mlflow_creds_with_none_token_falls_through(self):
        """Falls through to SDK if MLflow creds return None token."""
        import sys

        mock_creds_no_token = SimpleNamespace(host="https://test.cloud.databricks.com", token=None)
        mock_config = SimpleNamespace(host="https://sdk.cloud.databricks.com", token="sdk-token")
        mock_client = MagicMock()
        mock_client.config = mock_config

        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("mlflow") or mod_name.startswith("databricks"):
                saved[mod_name] = sys.modules.pop(mod_name)
        try:
            _install_fake_module(
                "mlflow.utils.databricks_utils",
                {"get_databricks_host_creds": lambda: mock_creds_no_token},
            )
            _install_fake_module(
                "databricks.sdk",
                {"WorkspaceClient": lambda: mock_client},
            )
            _propagate_databricks_auth_env()
            assert os.environ["DATABRICKS_HOST"] == "https://sdk.cloud.databricks.com"
            assert os.environ["DATABRICKS_TOKEN"] == "sdk-token"
        finally:
            for mod_name in list(sys.modules):
                if mod_name.startswith("mlflow") or mod_name.startswith("databricks"):
                    del sys.modules[mod_name]
            sys.modules.update(saved)

    def test_only_host_set_still_tries_extraction(self):
        """Tries extraction if only one of HOST/TOKEN is set."""
        import sys

        os.environ["DATABRICKS_HOST"] = "https://partial.com"
        mock_creds = SimpleNamespace(host="https://new.cloud.databricks.com", token="new-token")

        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("mlflow"):
                saved[mod_name] = sys.modules.pop(mod_name)
        try:
            _install_fake_module(
                "mlflow.utils.databricks_utils",
                {"get_databricks_host_creds": lambda: mock_creds},
            )
            _propagate_databricks_auth_env()
            assert os.environ["DATABRICKS_HOST"] == "https://new.cloud.databricks.com"
            assert os.environ["DATABRICKS_TOKEN"] == "new-token"
        finally:
            for mod_name in list(sys.modules):
                if mod_name.startswith("mlflow"):
                    del sys.modules[mod_name]
            sys.modules.update(saved)
