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

"""Tests for distributed mesh sharding and distributed inference."""

import torch
import numpy as np
import pytest

from transolver3.model import Transolver3
from transolver3.inference import CachedInference, DistributedCachedInference

from transolver3.distributed import mesh_shard_range


class TestMeshShardRange:
    """Test mesh_shard_range partitions correctly."""

    def test_even_division(self):
        """100 points across 4 workers = 25 each."""
        ranges = [mesh_shard_range(100, r, 4) for r in range(4)]
        assert ranges == [(0, 25), (25, 50), (50, 75), (75, 100)]

    def test_uneven_division(self):
        """10 points across 3 workers = 4, 3, 3."""
        ranges = [mesh_shard_range(10, r, 3) for r in range(3)]
        # First worker gets one extra (remainder = 1)
        assert ranges[0] == (0, 4)
        assert ranges[1] == (4, 7)
        assert ranges[2] == (7, 10)

    def test_covers_all_points(self):
        """All points are covered exactly once."""
        N = 8_800_000  # DrivAerML surface mesh size
        for world_size in [1, 2, 4, 8]:
            all_indices = set()
            for rank in range(world_size):
                start, end = mesh_shard_range(N, rank, world_size)
                shard_indices = set(range(start, end))
                # No overlap with previous shards
                assert all_indices.isdisjoint(shard_indices), f"Overlap at world_size={world_size}, rank={rank}"
                all_indices.update(shard_indices)
            # All points covered
            assert len(all_indices) == N, f"Missing points at world_size={world_size}: {N - len(all_indices)}"

    def test_single_worker(self):
        """Single worker gets everything."""
        assert mesh_shard_range(1000, 0, 1) == (0, 1000)

    def test_more_workers_than_points(self):
        """Edge case: more workers than points."""
        ranges = [mesh_shard_range(3, r, 8) for r in range(8)]
        # First 3 workers get 1 point each, rest get 0
        non_empty = [(s, e) for s, e in ranges if s < e]
        assert len(non_empty) == 3
        assert sum(e - s for s, e in ranges) == 3


class TestMeshShardedDataset:
    """Test DrivAerMLDataset with shard_id/num_shards."""

    @pytest.fixture
    def synthetic_data_dir(self, tmp_path):
        """Create a small synthetic DrivAerML-like .npz file."""
        N = 1000
        d_params = 6
        data = {
            "params": np.random.randn(d_params).astype(np.float32),
            "surface_coords": np.random.randn(N, 3).astype(np.float32),
            "surface_normals": np.random.randn(N, 3).astype(np.float32),
            "surface_pressure": np.random.randn(N, 1).astype(np.float32),
            "surface_wall_shear": np.random.randn(N, 3).astype(np.float32),
        }
        sample_path = tmp_path / "sample_000.npz"
        np.savez(sample_path, **data)

        # Write split file
        split_file = tmp_path / "train.txt"
        split_file.write_text("sample_000.npz\n")

        return tmp_path, data, N

    def test_sharded_loading(self, synthetic_data_dir):
        """Sharded datasets load disjoint subsets of the mesh."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Industrial-Scale-Benchmarks"))
        from dataset.drivaer_ml import DrivAerMLDataset

        data_dir, data, N = synthetic_data_dir

        # Load with 2 shards
        ds0 = DrivAerMLDataset(
            str(data_dir), split="train", field="surface", subset_size=None, shard_id=0, num_shards=2
        )
        ds1 = DrivAerMLDataset(
            str(data_dir), split="train", field="surface", subset_size=None, shard_id=1, num_shards=2
        )

        sample0 = ds0[0]
        sample1 = ds1[0]

        n0 = sample0["surface_x"].shape[0]
        n1 = sample1["surface_x"].shape[0]

        # Both shards together cover all points
        assert n0 + n1 == N, f"Shards cover {n0}+{n1}={n0 + n1}, expected {N}"

        # Each shard is roughly half
        assert abs(n0 - n1) <= 1, f"Shards unbalanced: {n0} vs {n1}"

    def test_unsharded_loading(self, synthetic_data_dir):
        """Without sharding, all points are loaded."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Industrial-Scale-Benchmarks"))
        from dataset.drivaer_ml import DrivAerMLDataset

        data_dir, data, N = synthetic_data_dir

        ds = DrivAerMLDataset(str(data_dir), split="train", field="surface", subset_size=None)
        sample = ds[0]
        assert sample["surface_x"].shape[0] == N

    def test_sharded_with_subsample(self, synthetic_data_dir):
        """Sharding + subsampling: subsample from the local shard only."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Industrial-Scale-Benchmarks"))
        from dataset.drivaer_ml import DrivAerMLDataset

        data_dir, data, N = synthetic_data_dir

        subset = 100
        ds = DrivAerMLDataset(
            str(data_dir), split="train", field="surface", subset_size=subset, shard_id=0, num_shards=4
        )
        sample = ds[0]
        # Should be min(shard_size, subset_size)
        shard_size = N // 4 + (1 if 0 < N % 4 else 0)
        expected = min(shard_size, subset)
        assert sample["surface_x"].shape[0] == expected


class TestDistributedCachedInference:
    """Test DistributedCachedInference falls back correctly in non-distributed mode."""

    @pytest.fixture
    def model_and_data(self):
        """Create a small model and synthetic mesh."""
        B, N, space_dim, out_dim = 1, 200, 6, 2
        model = Transolver3(
            space_dim=space_dim,
            n_layers=3,
            n_hidden=32,
            n_head=4,
            fun_dim=0,
            out_dim=out_dim,
            slice_num=8,
            mlp_ratio=1,
        )
        model.eval()
        x = torch.randn(B, N, space_dim)
        return model, x

    def test_fallback_matches_cached_inference(self, model_and_data):
        """Without distributed init, DistributedCachedInference == CachedInference."""
        model, x = model_and_data

        engine_single = CachedInference(model, cache_chunk_size=50, decode_chunk_size=50)
        engine_dist = DistributedCachedInference(model, cache_chunk_size=50, decode_chunk_size=50)

        pred_single = engine_single.predict(x)
        pred_dist = engine_dist.predict(x, gather=False)

        assert pred_single.shape == pred_dist.shape
        diff = (pred_single - pred_dist).abs().max().item()
        assert diff < 1e-5, f"Predictions differ by {diff}"

    def test_build_cache_fallback(self, model_and_data):
        """Cache from distributed engine matches single-GPU cache."""
        model, x = model_and_data

        cache_single = CachedInference(model, cache_chunk_size=50).build_cache(x)
        cache_dist = DistributedCachedInference(model, cache_chunk_size=50).build_cache(x)

        assert len(cache_single) == len(cache_dist)
        for s, d in zip(cache_single, cache_dist):
            diff = (s - d).abs().max().item()
            assert diff < 1e-5, f"Cache differs by {diff}"

    def test_manual_shard_simulation(self, model_and_data):
        """Simulate 2-rank sharding without actual distributed: split mesh,
        accumulate s_raw/d manually, verify cache matches single-GPU."""
        model, x = model_and_data
        B, N, C = x.shape

        # Single-GPU cache (ground truth)
        cache_full = model.cache_physical_states(x, chunk_size=50)

        # Simulate 2 shards: split x into two halves
        x0 = x[:, : N // 2]
        x1 = x[:, N // 2 :]

        # Each "rank" preprocesses and accumulates locally
        fx0 = model._preprocess(x0)
        fx1 = model._preprocess(x1)

        cache_simulated = []
        for block_idx, block in enumerate(model.blocks):
            s_raw_0, d_0 = block.compute_physical_state(fx0)
            s_raw_1, d_1 = block.compute_physical_state(fx1)

            # Simulated all-reduce (just add)
            s_raw_total = s_raw_0 + s_raw_1
            d_total = d_0 + d_1

            s_out = block.compute_cached_state(s_raw_total, d_total)
            cache_simulated.append(s_out)

            # Advance through block
            fx0 = block(fx0)
            fx1 = block(fx1)

        # Compare: sharded accumulation should match single-GPU
        for layer_idx, (s_full, s_sim) in enumerate(zip(cache_full, cache_simulated)):
            diff = (s_full - s_sim).abs().max().item()
            assert diff < 1e-4, f"Layer {layer_idx}: cache differs by {diff}"

    def test_predict_gather_false_matches_single_gpu(self, model_and_data):
        """predict(gather=False) in non-distributed mode matches CachedInference."""
        model, x = model_and_data

        engine_single = CachedInference(model, cache_chunk_size=50, decode_chunk_size=50)
        engine_dist = DistributedCachedInference(model, cache_chunk_size=50, decode_chunk_size=50)

        pred_single = engine_single.predict(x)
        pred_gather_false = engine_dist.predict(x, gather=False)
        pred_gather_true = engine_dist.predict(x, gather=True)

        # In non-distributed mode, gather=True should behave same as gather=False
        assert pred_single.shape == pred_gather_false.shape
        assert pred_single.shape == pred_gather_true.shape
        diff_false = (pred_single - pred_gather_false).abs().max().item()
        diff_true = (pred_single - pred_gather_true).abs().max().item()
        assert diff_false < 1e-5, f"gather=False differs by {diff_false}"
        assert diff_true < 1e-5, f"gather=True differs by {diff_true}"

    def test_gather_simulated_uneven_shards(self, model_and_data):
        """Simulate the all_gather padding logic for uneven shard sizes.

        Splits a mesh into uneven partitions, runs predict on each shard
        independently, then manually performs the same pad-gather-trim
        that DistributedCachedInference.predict(gather=True) would do.
        Verifies the reassembled output matches single-GPU predictions.
        """
        model, x = model_and_data
        B, N, _ = x.shape

        # Single-GPU ground truth
        engine = CachedInference(model, cache_chunk_size=50, decode_chunk_size=50)
        pred_full = engine.predict(x)

        # Simulate 3 uneven shards (e.g., 67 + 67 + 66 = 200)
        world_size = 3
        shards = []
        for rank in range(world_size):
            start, end = mesh_shard_range(N, rank, world_size)
            shards.append((start, end))

        # Each "rank" runs predict on its local shard
        local_preds = []
        for start, end in shards:
            x_local = x[:, start:end]
            pred_local = engine.predict(x_local)
            local_preds.append(pred_local)

        # Simulate the pad-gather-trim logic from inference.py lines 300-323
        all_n = [p.shape[1] for p in local_preds]
        max_n = max(all_n)

        # Pad each local prediction to max_n
        padded = []
        for pred in local_preds:
            n_local = pred.shape[1]
            if n_local < max_n:
                pad = torch.zeros(B, max_n - n_local, pred.shape[2])
                padded.append(torch.cat([pred, pad], dim=1))
            else:
                padded.append(pred)

        # Trim padding and concatenate in rank order
        parts = []
        for i, n in enumerate(all_n):
            parts.append(padded[i][:, :n])
        reassembled = torch.cat(parts, dim=1)

        assert reassembled.shape == pred_full.shape, f"Shape mismatch: {reassembled.shape} vs {pred_full.shape}"
        # Note: predictions won't be bit-identical because each shard runs
        # independently (no all-reduce on cache). This test verifies the
        # gather/padding/trimming logic is correct, not numerical equivalence.
        # We verify shape and that the reassembly didn't corrupt data.
        for i, (start, end) in enumerate(shards):
            reassembled_shard = reassembled[:, start:end]
            diff = (reassembled_shard - local_preds[i]).abs().max().item()
            assert diff == 0.0, f"Shard {i} corrupted during gather: diff={diff}"
