# Copyright 2024 Databricks, Inc.
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

"""Tests for transolver3.profiling — memory and latency profiling (CPU paths)."""

import torch

from transolver3.model import Transolver3
from transolver3.profiling import (
    MemoryResult,
    LatencyResult,
    _track_memory_cpu,
    profile_memory,
    profile_latency,
    benchmark_scaling,
    format_benchmark_table,
)


def _small_model():
    return Transolver3(
        space_dim=3,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=1,
        slice_num=8,
        mlp_ratio=1,
    ).eval()


class TestTrackMemoryCpu:
    def test_tracks_allocation(self):
        """_track_memory_cpu records positive peak memory."""
        with _track_memory_cpu():
            _ = [0] * 100_000  # allocate some memory
        assert _track_memory_cpu._peak > 0


class TestProfileMemory:
    def test_forward_mode_cpu(self):
        """profile_memory in forward mode returns valid MemoryResult."""
        model = _small_model()
        x = torch.randn(1, 100, 3)
        result = profile_memory(model, x, mode="forward")
        assert isinstance(result, MemoryResult)
        assert result.peak_mb > 0
        assert result.mesh_size == 100
        assert result.backend == "cpu_tracemalloc"

    def test_cached_mode_cpu(self):
        """profile_memory in cached mode returns valid MemoryResult."""
        model = _small_model()
        x = torch.randn(1, 100, 3)
        result = profile_memory(model, x, mode="cached", cache_chunk_size=50, decode_chunk_size=50)
        assert isinstance(result, MemoryResult)
        assert result.peak_mb > 0
        assert result.config["mode"] == "cached"


class TestProfileLatency:
    def test_latency_cpu(self):
        """profile_latency returns valid LatencyResult on CPU."""
        model = _small_model()
        x = torch.randn(1, 100, 3)
        result = profile_latency(model, x, num_warmup=1, num_runs=3)
        assert isinstance(result, LatencyResult)
        assert result.mean_ms > 0
        assert result.num_runs == 3
        assert result.mesh_size == 100

    def test_cached_latency_cpu(self):
        """profile_latency in cached mode works on CPU."""
        model = _small_model()
        x = torch.randn(1, 100, 3)
        result = profile_latency(
            model,
            x,
            mode="cached",
            num_warmup=1,
            num_runs=2,
            cache_chunk_size=50,
            decode_chunk_size=50,
        )
        assert isinstance(result, LatencyResult)
        assert result.mean_ms > 0


class TestBenchmarkScaling:
    def test_scaling_cpu(self):
        """benchmark_scaling produces structured results on CPU."""
        model = _small_model()
        results = benchmark_scaling(
            model,
            space_dim=3,
            mesh_sizes=[50, 100],
            configs=[{"label": "no_tiling", "num_tiles": 0, "tile_size": 0}],
            measure_memory=True,
            measure_latency=True,
            num_latency_runs=2,
        )
        assert results["mesh_sizes"] == [50, 100]
        assert len(results["memory"]) == 1  # 1 config
        assert len(results["memory"][0]) == 2  # 2 mesh sizes
        assert len(results["latency"]) == 1
        assert len(results["latency"][0]) == 2

    def test_format_benchmark_table_output(self):
        """format_benchmark_table produces non-empty string."""
        model = _small_model()
        results = benchmark_scaling(
            model,
            space_dim=3,
            mesh_sizes=[50],
            configs=[{"label": "test", "num_tiles": 0, "tile_size": 0}],
            num_latency_runs=2,
        )
        table = format_benchmark_table(results)
        assert "Memory" in table
        assert "Latency" in table
        assert "test" in table
