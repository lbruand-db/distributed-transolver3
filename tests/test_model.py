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
Tests for Transolver3 model: forward pass, caching, chunking, mixed precision.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transolver3.transolver3_block import Transolver3Block, _pointwise_chunked
from transolver3.model import Transolver3
from transolver3.inference import CachedInference
from transolver3.amortized_training import relative_l2_loss


def test_block_forward():
    """Test Transolver3Block forward pass."""
    B, N, C = 2, 100, 64
    out_dim = 4

    block = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0, slice_num=16, last_layer=False)
    x = torch.randn(B, N, C)
    out = block(x)
    assert out.shape == (B, N, C), f"Non-last block: expected {(B, N, C)}, got {out.shape}"

    block_last = Transolver3Block(
        num_heads=4, hidden_dim=C, dropout=0.0, slice_num=16, last_layer=True, out_dim=out_dim
    )
    out = block_last(x)
    assert out.shape == (B, N, out_dim), f"Last block: expected {(B, N, out_dim)}, got {out.shape}"


def test_full_model():
    """Test Transolver3 end-to-end forward pass."""
    B, N = 2, 100
    space_dim = 3
    fun_dim = 2
    out_dim = 4

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=fun_dim,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
        ref=4,
        unified_pos=False,
    )

    x = torch.randn(B, N, space_dim)
    fx = torch.randn(B, N, fun_dim)

    out = model(x, fx=fx)
    assert out.shape == (B, N, out_dim), f"Expected {(B, N, out_dim)}, got {out.shape}"

    out_tiled = model(x, fx=fx, num_tiles=2)
    assert out_tiled.shape == (B, N, out_dim)


def test_amortized_training():
    """Test geometry amortized training with subset indices."""
    B, N = 1, 500
    space_dim = 3
    out_dim = 2
    subset_size = 100

    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=8,
        mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim)

    indices = torch.randperm(N)[:subset_size]
    out = model(x, subset_indices=indices)
    assert out.shape == (B, subset_size, out_dim), f"Expected {(B, subset_size, out_dim)}, got {out.shape}"

    loss = relative_l2_loss(out, target[:, indices])
    loss.backward()


def test_cached_model_inference():
    """Test full model cached inference pipeline."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(B, N, space_dim)

    with torch.no_grad():
        out_direct = model(x)
        engine = CachedInference(model, cache_chunk_size=50, decode_chunk_size=50)
        out_cached = engine.predict(x)

    assert out_cached.shape == out_direct.shape
    diff = (out_direct - out_cached).abs().max().item()
    assert diff < 1e-3, f"Cached model inference max diff: {diff}"


def test_chunked_decode_matches_unchunked():
    """Chunked decode (small decode_chunk_size) matches unchunked decode."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
    )
    model.eval()
    x = torch.randn(B, N, space_dim)

    with torch.no_grad():
        # Unchunked: decode_chunk_size >= N so no chunking
        engine_full = CachedInference(model, cache_chunk_size=50, decode_chunk_size=N)
        cache = engine_full.build_cache(x)
        pred_full = engine_full.decode(x, cache)

        # Chunked: small decode_chunk_size forces the loop path
        engine_chunked = CachedInference(model, cache_chunk_size=50, decode_chunk_size=37)
        pred_chunked = engine_chunked.decode(x, cache)

    assert pred_full.shape == pred_chunked.shape
    diff = (pred_full - pred_chunked).abs().max().item()
    assert diff < 1e-5, f"Chunked decode differs from unchunked by {diff}"


def test_fix3_mlp_chunking():
    """Fix #3: MLP chunking produces identical results to non-chunked processing."""
    B, N, C = 2, 200, 64
    out_dim = 4

    # Test with non-last block
    block = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0, slice_num=16, last_layer=False, mlp_chunk_size=50)
    block_ref = Transolver3Block(
        num_heads=4, hidden_dim=C, dropout=0.0, slice_num=16, last_layer=False, mlp_chunk_size=0
    )
    block_ref.load_state_dict(block.state_dict())
    block.eval()
    block_ref.eval()

    x = torch.randn(B, N, C)
    with torch.no_grad():
        out_chunked = block(x)
        out_full = block_ref(x)

    diff = (out_chunked - out_full).abs().max().item()
    assert diff < 1e-5, f"Chunked vs full MLP max diff: {diff}"

    # Test with last block
    block_last = Transolver3Block(
        num_heads=4, hidden_dim=C, dropout=0.0, slice_num=16, last_layer=True, out_dim=out_dim, mlp_chunk_size=50
    )
    block_last_ref = Transolver3Block(
        num_heads=4, hidden_dim=C, dropout=0.0, slice_num=16, last_layer=True, out_dim=out_dim, mlp_chunk_size=0
    )
    block_last_ref.load_state_dict(block_last.state_dict())
    block_last.eval()
    block_last_ref.eval()

    with torch.no_grad():
        out_last_chunked = block_last(x)
        out_last_full = block_last_ref(x)

    diff_last = (out_last_chunked - out_last_full).abs().max().item()
    assert diff_last < 1e-5, f"Last block chunked vs full max diff: {diff_last}"


def test_fix3_pointwise_chunked_helper():
    """Fix #3: _pointwise_chunked produces identical results."""
    import torch.nn as nn

    B, N, C = 2, 100, 64
    mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C), nn.GELU(), nn.Linear(C, C))
    mlp.eval()

    x = torch.randn(B, N, C)
    with torch.no_grad():
        out_full = mlp(x)
        out_chunked = _pointwise_chunked(mlp, x, chunk_size=30)

    diff = (out_full - out_chunked).abs().max().item()
    assert diff < 1e-6, f"_pointwise_chunked max diff: {diff}"


def test_fix4_streaming_cache():
    """Fix #4: Streaming (CPU-offloaded) cache matches full GPU cache."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(B, N, space_dim)

    with torch.no_grad():
        cache_full = model._cache_full(x, fx=None, T=None, num_tiles=0)
        cache_chunked = model._cache_chunked(x, fx=None, T=None, num_tiles=0, chunk_size=50)

    assert len(cache_full) == len(cache_chunked), f"Cache lengths differ: {len(cache_full)} vs {len(cache_chunked)}"

    for i, (cf, cc) in enumerate(zip(cache_full, cache_chunked)):
        cc_gpu = cc.to(cf.device)
        diff = (cf - cc_gpu).abs().max().item()
        assert diff < 1e-4, f"Layer {i} cache diff: {diff}"

    with torch.no_grad():
        out_full = model.decode_from_cache(x, cache_full)
        out_chunked = model.decode_from_cache(x, cache_chunked)

    diff_out = (out_full - out_chunked).abs().max().item()
    assert diff_out < 1e-3, f"Decoded output diff: {diff_out}"


def test_fix4_memory_pattern():
    """Fix #4: Verify that _cache_chunked produces correct structure."""
    B, N = 1, 100
    space_dim = 3

    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=8,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(B, N, space_dim)

    with torch.no_grad():
        cache = model._cache_chunked(x, fx=None, T=None, num_tiles=0, chunk_size=25)

    assert len(cache) == 2, f"Expected 2 cache entries, got {len(cache)}"

    for i, c in enumerate(cache):
        assert c.shape[0] == B, f"Layer {i}: batch dim wrong"
        assert c.shape[1] == 4, f"Layer {i}: heads dim wrong"  # n_head=4
        assert c.shape[2] == 8, f"Layer {i}: slice_num dim wrong"  # slice_num=8
        assert c.shape[3] == 8, f"Layer {i}: dim_head dim wrong"  # 32/4=8


def test_fix6_mixed_precision():
    """Fix #6: Model works under torch.amp autocast (mixed precision)."""
    B, N = 1, 100
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=8,
        mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out = model(x)
        assert out.shape == (B, N, out_dim), f"Autocast forward shape: {out.shape}"
        loss = relative_l2_loss(out.float(), target)
        loss.backward()

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            engine = CachedInference(model, cache_chunk_size=30, decode_chunk_size=30)
            out_cached = engine.predict(x)
            assert out_cached.shape == (B, N, out_dim), f"Autocast cached shape: {out_cached.shape}"


def test_fix6_autocast_numerics():
    """Fix #6: Verify autocast doesn't produce NaN/Inf."""
    B, N = 1, 100
    space_dim = 3

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=2,
        slice_num=16,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(B, N, space_dim)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(x)

    assert not torch.isnan(out).any(), "Output contains NaN under autocast"
    assert not torch.isinf(out).any(), "Output contains Inf under autocast"


def test_model_with_mlp_chunking():
    """Test full model with mlp_chunk_size set, verifying end-to-end."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    model_chunked = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
        mlp_chunk_size=50,
    )
    model_ref = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
        mlp_chunk_size=0,
    )
    model_ref.load_state_dict(model_chunked.state_dict())
    model_chunked.eval()
    model_ref.eval()

    x = torch.randn(B, N, space_dim)

    with torch.no_grad():
        out_chunked = model_chunked(x)
        out_ref = model_ref(x)

    diff = (out_chunked - out_ref).abs().max().item()
    assert diff < 1e-4, f"Model mlp_chunk_size=50 vs 0 diff: {diff}"


def test_tile_size_model():
    """Test Transolver3 with tile_size at constructor and forward levels."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    # Constructor-level tile_size
    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
        tile_size=50,
    )
    model.eval()

    x = torch.randn(B, N, space_dim)
    with torch.no_grad():
        out_ctor = model(x)
        # Forward-level tile_size overrides
        out_fwd = model(x, tile_size=100)
        # Equivalent num_tiles
        out_ref = model(x, num_tiles=4, tile_size=0)  # tile_size=0 disables

    assert out_ctor.shape == (B, N, out_dim)
    assert out_fwd.shape == (B, N, out_dim)

    # ctor tile_size=50 -> 4 tiles, should match num_tiles=4
    diff = (out_ctor - out_ref).abs().max().item()
    assert diff < 1e-5, f"Constructor tile_size vs num_tiles diff: {diff}"
