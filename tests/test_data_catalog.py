"""Tests for transolver3.data_catalog — Delta Lake mesh metadata and normalizer stats."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Skip entire module if pyspark is not installed
# ---------------------------------------------------------------------------
pyspark = pytest.importorskip("pyspark", reason="pyspark required for data_catalog tests")

from pyspark.sql import SparkSession  # noqa: E402

from transolver3.data_catalog import (  # noqa: E402
    register_mesh_metadata,
    get_mesh_metadata,
    log_normalization_stats,
    load_normalization_stats,
)
from transolver3.normalizer import InputNormalizer, TargetNormalizer  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    """Create a local SparkSession for testing (reused across tests)."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("transolver3-test")
        .config("spark.sql.warehouse.dir", tempfile.mkdtemp())
        .config("spark.driver.extraJavaOptions", "-Dderby.system.home=" + tempfile.mkdtemp())
        .enableHiveSupport()
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_npz_dir():
    """Create a temp directory with synthetic .npz mesh files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            n_surf = 100 + i * 10
            n_vol = 500 + i * 50
            np.savez(
                os.path.join(tmpdir, f"sample_{i:03d}.npz"),
                surface_coords=np.random.randn(n_surf, 3).astype(np.float32),
                surface_normals=np.random.randn(n_surf, 3).astype(np.float32),
                surface_pressure=np.random.randn(n_surf, 1).astype(np.float32),
                volume_coords=np.random.randn(n_vol, 3).astype(np.float32),
                volume_pressure=np.random.randn(n_vol, 1).astype(np.float32),
                params=np.random.randn(16).astype(np.float32),
            )
        yield tmpdir


# Use a unique database per test run to avoid conflicts
DB = "default"


def test_register_mesh_metadata(spark, sample_npz_dir):
    """register_mesh_metadata scans .npz files and writes a Delta table."""
    df = register_mesh_metadata(spark, DB, DB, "test_mesh_meta", sample_npz_dir)
    assert df.count() == 3

    row = df.filter("sample_id = 'sample_000'").first()
    assert row.surface_cells == 100
    assert row.volume_cells == 500
    assert row.total_cells == 600
    assert row.params_dim == 16
    assert row.file_size_bytes > 0
    assert "surface_coords" in json.loads(row.available_keys)


def test_get_mesh_metadata(spark, sample_npz_dir):
    """get_mesh_metadata reads back the registered metadata."""
    register_mesh_metadata(spark, DB, DB, "test_mesh_meta_read", sample_npz_dir)
    df = get_mesh_metadata(spark, DB, DB, "test_mesh_meta_read")
    assert df.count() == 3
    assert "sample_id" in df.columns


def test_log_and_load_input_normalizer(spark):
    """Round-trip InputNormalizer stats through Delta."""
    norm = InputNormalizer(scale=1000.0, per_sample=False)
    coords = torch.randn(5, 100, 3)
    norm.fit(coords)

    log_normalization_stats(spark, DB, DB, "test_norm_stats", norm, "input_coords")

    state = load_normalization_stats(spark, DB, DB, "test_norm_stats", "InputNormalizer")

    norm2 = InputNormalizer(scale=1000.0, per_sample=False)
    norm2.load_state_dict(state)

    assert torch.allclose(norm.data_min, norm2.data_min, atol=1e-6)
    assert torch.allclose(norm.data_max, norm2.data_max, atol=1e-6)


def test_log_and_load_target_normalizer(spark):
    """Round-trip TargetNormalizer stats through Delta."""
    norm = TargetNormalizer(out_dim=4)
    targets = torch.randn(5, 100, 4)
    norm.fit(targets)

    log_normalization_stats(spark, DB, DB, "test_target_norm_stats", norm, "pressure_target")

    state = load_normalization_stats(spark, DB, DB, "test_target_norm_stats", "TargetNormalizer")

    norm2 = TargetNormalizer(out_dim=4)
    norm2.load_state_dict(state)

    assert torch.allclose(norm.mean, norm2.mean, atol=1e-6)
    assert torch.allclose(norm.std, norm2.std, atol=1e-6)


def test_load_missing_normalizer_raises(spark):
    """load_normalization_stats raises ValueError when type not found."""
    # Write an InputNormalizer to a fresh table
    norm = InputNormalizer()
    log_normalization_stats(spark, DB, DB, "test_norm_missing", norm)

    with pytest.raises(ValueError, match="No TargetNormalizer"):
        load_normalization_stats(spark, DB, DB, "test_norm_missing", "TargetNormalizer")
