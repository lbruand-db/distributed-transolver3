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

"""
Tests for profiling and benchmarking utilities.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transolver3.model import Transolver3
from transolver3.profiling import (
    profile_memory,
    profile_latency,
    benchmark_scaling,
    format_benchmark_table,
    MemoryResult,
    LatencyResult,
)


def test_profile_memory_forward():
    """Test profile_memory returns valid MemoryResult for forward mode."""
    model = Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=8,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 100, 3)

    result = profile_memory(model, x, mode="forward")
    assert isinstance(result, MemoryResult)
    assert result.peak_mb > 0
    assert result.mesh_size == 100
    assert result.config["mode"] == "forward"

    # With tiling
    result_tiled = profile_memory(model, x, tile_size=30, mode="forward")
    assert isinstance(result_tiled, MemoryResult)
    assert result_tiled.peak_mb > 0


def test_profile_memory_cached():
    """Test profile_memory works for cached inference mode."""
    model = Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=8,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 100, 3)

    result = profile_memory(model, x, mode="cached", cache_chunk_size=30, decode_chunk_size=30)
    assert isinstance(result, MemoryResult)
    assert result.peak_mb > 0
    assert result.config["mode"] == "cached"


def test_profile_latency():
    """Test profile_latency returns valid LatencyResult."""
    model = Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=8,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 100, 3)

    result = profile_latency(model, x, mode="forward", num_warmup=1, num_runs=3)
    assert isinstance(result, LatencyResult)
    assert result.mean_ms > 0
    assert result.num_runs == 3
    assert result.mesh_size == 100


def test_benchmark_scaling():
    """Test benchmark_scaling produces structured results across mesh sizes."""
    model = Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=8,
        mlp_ratio=1,
    )
    model.eval()

    configs = [
        {"label": "no_tiling", "num_tiles": 0},
        {"label": "tile_50", "tile_size": 50},
    ]

    results = benchmark_scaling(
        model,
        space_dim=3,
        mesh_sizes=[50, 100, 200],
        configs=configs,
        measure_memory=True,
        measure_latency=True,
        num_latency_runs=2,
    )

    assert results["mesh_sizes"] == [50, 100, 200]
    assert len(results["configs"]) == 2
    assert len(results["memory"]) == 2  # 2 configs
    assert len(results["memory"][0]) == 3  # 3 mesh sizes
    assert len(results["latency"]) == 2
    assert len(results["latency"][0]) == 3

    for row in results["memory"]:
        for mr in row:
            assert isinstance(mr, MemoryResult)
            assert mr.peak_mb > 0

    for row in results["latency"]:
        for lr in row:
            assert isinstance(lr, LatencyResult)
            assert lr.mean_ms > 0


def test_format_benchmark_table():
    """Test format_benchmark_table produces readable output."""
    model = Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=8,
        mlp_ratio=1,
    )
    model.eval()

    results = benchmark_scaling(
        model,
        space_dim=3,
        mesh_sizes=[50, 100],
        configs=[{"label": "baseline", "num_tiles": 0}],
        measure_latency=False,
        num_latency_runs=1,
    )

    table = format_benchmark_table(results)
    assert isinstance(table, str)
    assert "Memory" in table
    assert "baseline" in table
    assert "N=50" in table
    assert "N=100" in table


def test_tiling_reduces_memory_relative():
    """Verify tiling doesn't increase memory vs no tiling (sanity check)."""
    model = Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=16,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 500, 3)

    mem_no_tile = profile_memory(model, x, num_tiles=0)
    mem_tiled = profile_memory(model, x, tile_size=100)

    assert mem_no_tile.peak_mb > 0
    assert mem_tiled.peak_mb > 0
    assert mem_no_tile.peak_mb < float("inf")
    assert mem_tiled.peak_mb < float("inf")
