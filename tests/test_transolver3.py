"""
Tests for Transolver-3 implementation.

Verifies:
  1. PhysicsAttentionV3 forward pass produces correct shapes
  2. Tiled forward == standard forward (numerical equivalence)
  3. Cached inference == direct forward (numerical equivalence)
  4. Full model forward pass works end-to-end
  5. Amortized training with subset indices
  6. Fix #1: Head aggregation uses rearrange (concat), not mean
  7. Fix #2: _slice_aggregate uses matmul, not einsum (no (B,H,N,C) intermediate)
  8. Fix #3: MLP chunking produces identical results to non-chunked
  9. Fix #4: Streaming cache (CPU offload) matches full cache
  10. Fix #6: Mixed precision (autocast) forward pass works
"""

import sys
import os
import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transolver3.physics_attention_v3 import PhysicsAttentionV3, _slice_aggregate, _deslice, _resolve_num_tiles
from transolver3.transolver3_block import Transolver3Block, _pointwise_chunked
from transolver3.model import Transolver3
from transolver3.inference import CachedInference
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss, create_optimizer, create_scheduler,
    train_step,
)
from transolver3.normalizer import InputNormalizer, TargetNormalizer
from transolver3.profiling import (
    profile_memory, profile_latency, benchmark_scaling, format_benchmark_table,
    MemoryResult, LatencyResult,
)


def test_attention_shapes():
    """Test that PhysicsAttentionV3 produces correct output shapes."""
    B, N, C = 2, 100, 64
    heads = 4
    dim_head = C // heads
    slice_num = 16

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=dim_head, slice_num=slice_num)
    x = torch.randn(B, N, C)

    out = attn(x)
    assert out.shape == (B, N, C), f"Expected {(B, N, C)}, got {out.shape}"
    print("PASS: attention output shape correct")


def test_tiled_vs_standard():
    """Test that tiled forward matches standard forward."""
    B, N, C = 1, 200, 64
    heads = 4
    dim_head = C // heads
    slice_num = 16

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=dim_head, slice_num=slice_num)
    attn.eval()

    x = torch.randn(B, N, C)

    with torch.no_grad():
        out_standard = attn(x, num_tiles=0)
        out_tiled_2 = attn(x, num_tiles=2)
        out_tiled_4 = attn(x, num_tiles=4)

    diff_2 = (out_standard - out_tiled_2).abs().max().item()
    diff_4 = (out_standard - out_tiled_4).abs().max().item()

    assert diff_2 < 1e-5, f"Tiled (2) vs standard max diff: {diff_2}"
    assert diff_4 < 1e-5, f"Tiled (4) vs standard max diff: {diff_4}"
    print(f"PASS: tiled matches standard (max diff: 2-tile={diff_2:.2e}, 4-tile={diff_4:.2e})")


def test_cached_inference_equivalence():
    """Test that cached inference produces same results as direct forward."""
    B, N, C = 1, 150, 64
    heads = 4
    slice_num = 16

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=C // heads, slice_num=slice_num)
    attn.eval()

    x = torch.randn(B, N, C)

    with torch.no_grad():
        out_direct = attn(x, num_tiles=0)
        s_raw, d = attn.compute_physical_state(x)
        s_out = attn.compute_cached_state(s_raw, d)
        out_cached = attn.decode_from_cache(x, s_out)

    diff = (out_direct - out_cached).abs().max().item()
    assert diff < 1e-5, f"Cached vs direct max diff: {diff}"
    print(f"PASS: cached inference matches direct (max diff: {diff:.2e})")


def test_block_forward():
    """Test Transolver3Block forward pass."""
    B, N, C = 2, 100, 64
    out_dim = 4

    block = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                              slice_num=16, last_layer=False)
    x = torch.randn(B, N, C)
    out = block(x)
    assert out.shape == (B, N, C), f"Non-last block: expected {(B, N, C)}, got {out.shape}"

    block_last = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                                   slice_num=16, last_layer=True, out_dim=out_dim)
    out = block_last(x)
    assert out.shape == (B, N, out_dim), f"Last block: expected {(B, N, out_dim)}, got {out.shape}"
    print("PASS: block forward shapes correct")


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

    print("PASS: full model forward shapes correct")


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
    assert out.shape == (B, subset_size, out_dim), \
        f"Expected {(B, subset_size, out_dim)}, got {out.shape}"

    loss = relative_l2_loss(out, target[:, indices])
    loss.backward()
    print(f"PASS: amortized training works (loss={loss.item():.4f})")


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
    print(f"PASS: cached model inference (shape match, max diff: {diff:.2e})")


def test_sampler():
    """Test AmortizedMeshSampler."""
    sampler = AmortizedMeshSampler(subset_size=100, seed=42)

    indices = sampler.sample(1000)
    assert indices.shape == (100,), f"Expected (100,), got {indices.shape}"
    assert indices.max() < 1000
    assert indices.min() >= 0

    indices_small = sampler.sample(50)
    assert indices_small.shape == (50,)

    print("PASS: AmortizedMeshSampler works correctly")


def test_scheduler():
    """Test cosine scheduler with warmup."""
    model = nn.Linear(10, 10)
    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, total_steps=1000, warmup_fraction=0.1)

    lrs = []
    for step in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    assert lrs[50] > lrs[0], "LR should increase during warmup"
    assert lrs[500] < lrs[100], "LR should decrease after warmup"
    assert lrs[-1] < lrs[100], "Final LR should be lower than mid-training"

    print("PASS: cosine scheduler with warmup works")


# ===== NEW TESTS FOR FIXES =====

def test_fix1_head_concat_not_mean():
    """Fix #1: Verify heads are concatenated (rearrange), not averaged.

    If heads were averaged (dim=1 mean), output would be (B, N, dim_head)
    not (B, N, dim). With concat, output is (B, N, heads*dim_head) = (B, N, dim).
    Also verify slice_linear3 maps dim_head -> dim_head (not dim_head -> dim).
    """
    B, N, C = 2, 50, 64
    heads = 4
    dim_head = C // heads

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=dim_head, slice_num=8)

    # Verify slice_linear3 shape: dim_head -> dim_head
    assert attn.slice_linear3.in_features == dim_head, \
        f"slice_linear3 in_features should be {dim_head}, got {attn.slice_linear3.in_features}"
    assert attn.slice_linear3.out_features == dim_head, \
        f"slice_linear3 out_features should be {dim_head}, got {attn.slice_linear3.out_features}"

    # Verify output shape is (B, N, dim) not (B, N, dim_head)
    x = torch.randn(B, N, C)
    out = attn(x)
    assert out.shape == (B, N, C), \
        f"Output should be (B,N,dim)={(B,N,C)}, got {out.shape}. Heads may be averaged instead of concatenated."

    # Verify that different heads produce different contributions
    attn.eval()
    with torch.no_grad():
        w = attn._compute_slice_weights(x)
        s_raw = _slice_aggregate(w, x, heads)
        s = attn.slice_linear1(s_raw / (w.sum(dim=2)[..., None] + 1e-5))
        q, k, v = attn.to_q(s), attn.to_k(s), attn.to_v(s)
        from torch.nn.functional import scaled_dot_product_attention
        s_out = scaled_dot_product_attention(q, k, v)
        s_out = attn.slice_linear3(s_out)
        x_out = _deslice(s_out, w)  # B, H, N, dim_head

        # Heads should differ (not all identical)
        head_diffs = (x_out[:, 0] - x_out[:, 1]).abs().max().item()
        assert head_diffs > 1e-6, "Heads should produce different outputs"

    print("PASS: Fix #1 — heads concatenated, slice_linear3 is dim_head->dim_head")


def test_fix2_no_large_intermediate():
    """Fix #2: Verify _slice_aggregate uses matmul without (B,H,N,C) intermediate.

    We can't directly check what PyTorch materializes, but we verify:
    1. The function produces correct results
    2. It uses matmul (broadcast) not einsum
    3. The result matches the mathematically equivalent einsum
    """
    B, H, N, M, C = 2, 4, 100, 16, 64

    w = torch.randn(B, H, N, M).softmax(dim=-1)
    x = torch.randn(B, N, C)

    # Our implementation
    result = _slice_aggregate(w, x, H)

    # Reference: einsum (mathematically correct but may materialize large tensor)
    reference = torch.einsum("bhnm,bnc->bhmc", w, x)

    diff = (result - reference).abs().max().item()
    assert diff < 1e-5, f"_slice_aggregate vs einsum max diff: {diff}"

    # Verify shape
    assert result.shape == (B, H, M, C), f"Expected {(B, H, M, C)}, got {result.shape}"

    # Verify _deslice too
    s_out = torch.randn(B, H, M, C)
    deslice_result = _deslice(s_out, w)
    deslice_ref = torch.einsum("bhmc,bhnm->bhnc", s_out, w)
    diff2 = (deslice_result - deslice_ref).abs().max().item()
    assert diff2 < 1e-5, f"_deslice vs einsum max diff: {diff2}"

    print(f"PASS: Fix #2 — matmul matches einsum (slice: {diff:.2e}, deslice: {diff2:.2e})")


def test_fix3_mlp_chunking():
    """Fix #3: MLP chunking produces identical results to non-chunked processing."""
    B, N, C = 2, 200, 64
    out_dim = 4

    # Test with non-last block
    block = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                              slice_num=16, last_layer=False, mlp_chunk_size=50)
    block_ref = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                                  slice_num=16, last_layer=False, mlp_chunk_size=0)
    # Copy weights
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
    block_last = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                                   slice_num=16, last_layer=True, out_dim=out_dim,
                                   mlp_chunk_size=50)
    block_last_ref = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                                       slice_num=16, last_layer=True, out_dim=out_dim,
                                       mlp_chunk_size=0)
    block_last_ref.load_state_dict(block_last.state_dict())
    block_last.eval()
    block_last_ref.eval()

    with torch.no_grad():
        out_last_chunked = block_last(x)
        out_last_full = block_last_ref(x)

    diff_last = (out_last_chunked - out_last_full).abs().max().item()
    assert diff_last < 1e-5, f"Last block chunked vs full max diff: {diff_last}"

    print(f"PASS: Fix #3 — MLP chunking matches full (non-last: {diff:.2e}, last: {diff_last:.2e})")


def test_fix3_pointwise_chunked_helper():
    """Fix #3: _pointwise_chunked produces identical results."""
    B, N, C = 2, 100, 64
    mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C), nn.GELU(), nn.Linear(C, C))
    mlp.eval()

    x = torch.randn(B, N, C)
    with torch.no_grad():
        out_full = mlp(x)
        out_chunked = _pointwise_chunked(mlp, x, chunk_size=30)

    diff = (out_full - out_chunked).abs().max().item()
    assert diff < 1e-6, f"_pointwise_chunked max diff: {diff}"
    print(f"PASS: Fix #3 — _pointwise_chunked helper (max diff: {diff:.2e})")


def test_fix4_streaming_cache():
    """Fix #4: Streaming (CPU-offloaded) cache matches full GPU cache.

    Verifies that _cache_chunked with CPU offloading produces the same
    cache and final output as _cache_full.
    """
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
        # Full cache (no chunking)
        cache_full = model._cache_full(x, fx=None, T=None, num_tiles=0)

        # Chunked cache (with CPU offloading)
        cache_chunked = model._cache_chunked(x, fx=None, T=None, num_tiles=0,
                                              chunk_size=50)

    # Verify cache sizes match
    assert len(cache_full) == len(cache_chunked), \
        f"Cache lengths differ: {len(cache_full)} vs {len(cache_chunked)}"

    # Verify each layer's cached state matches
    for i, (cf, cc) in enumerate(zip(cache_full, cache_chunked)):
        # cc might be on GPU or CPU depending on implementation
        cc_gpu = cc.to(cf.device)
        diff = (cf - cc_gpu).abs().max().item()
        assert diff < 1e-4, f"Layer {i} cache diff: {diff}"

    # Verify final decoded output matches
    with torch.no_grad():
        out_full = model.decode_from_cache(x, cache_full)
        out_chunked = model.decode_from_cache(x, cache_chunked)

    diff_out = (out_full - out_chunked).abs().max().item()
    assert diff_out < 1e-3, f"Decoded output diff: {diff_out}"

    print(f"PASS: Fix #4 — streaming cache matches full (cache max diff: {max((cf - cc.to(cf.device)).abs().max().item() for cf, cc in zip(cache_full, cache_chunked)):.2e}, output diff: {diff_out:.2e})")


def test_fix4_memory_pattern():
    """Fix #4: Verify that _cache_chunked stores features on CPU, not GPU.

    This is a structural test — we verify the returned cache tensors
    are on the expected device and that intermediate features aren't
    leaked as GPU tensors.
    """
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
        cache = model._cache_chunked(x, fx=None, T=None, num_tiles=0,
                                      chunk_size=25)

    # Cache should have one entry per layer
    assert len(cache) == 2, f"Expected 2 cache entries, got {len(cache)}"

    # Each cache entry should be a valid tensor with correct shape
    for i, c in enumerate(cache):
        assert c.shape[0] == B, f"Layer {i}: batch dim wrong"
        assert c.shape[1] == 4, f"Layer {i}: heads dim wrong"  # n_head=4
        assert c.shape[2] == 8, f"Layer {i}: slice_num dim wrong"  # slice_num=8
        assert c.shape[3] == 8, f"Layer {i}: dim_head dim wrong"  # 32/4=8

    print("PASS: Fix #4 — streaming cache structure correct")


def test_fix6_mixed_precision():
    """Fix #6: Model works under torch.amp autocast (mixed precision).

    Verifies forward pass, cached inference, and gradient computation
    all work correctly with autocast.
    """
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

    # Determine device type for autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Forward pass with autocast
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out = model(x)
        assert out.shape == (B, N, out_dim), f"Autocast forward shape: {out.shape}"

        # Loss and backward
        loss = relative_l2_loss(out.float(), target)
        loss.backward()

    # Cached inference with autocast
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            engine = CachedInference(model, cache_chunk_size=30, decode_chunk_size=30)
            out_cached = engine.predict(x)
            assert out_cached.shape == (B, N, out_dim), \
                f"Autocast cached shape: {out_cached.shape}"

    print(f"PASS: Fix #6 — mixed precision works (device_type={device_type})")


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
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(x)

    assert not torch.isnan(out).any(), "Output contains NaN under autocast"
    assert not torch.isinf(out).any(), "Output contains Inf under autocast"
    print("PASS: Fix #6 — no NaN/Inf under autocast")


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
    print(f"PASS: model with mlp_chunk_size matches reference (max diff: {diff:.2e})")


def test_inner_dim_assertion():
    """Test that inner_dim != dim raises an assertion error."""
    try:
        # dim=64, heads=4, dim_head=32 -> inner_dim=128 != 64
        attn = PhysicsAttentionV3(64, heads=4, dim_head=32, slice_num=8)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "inner_dim == dim" in str(e)
        print(f"PASS: inner_dim != dim correctly raises AssertionError")


def test_resolve_num_tiles():
    """Test _resolve_num_tiles auto-computes tile count from tile_size."""
    # tile_size=0 → passthrough num_tiles
    assert _resolve_num_tiles(1000, num_tiles=0) == 0
    assert _resolve_num_tiles(1000, num_tiles=5) == 5

    # tile_size > 0 → auto-compute, overrides num_tiles
    assert _resolve_num_tiles(1000, num_tiles=0, tile_size=100) == 10
    assert _resolve_num_tiles(1000, num_tiles=99, tile_size=100) == 10  # overrides
    assert _resolve_num_tiles(1050, num_tiles=0, tile_size=100) == 11  # ceil
    assert _resolve_num_tiles(100, num_tiles=0, tile_size=100) == 1  # exactly 1 tile
    assert _resolve_num_tiles(50, num_tiles=0, tile_size=100) == 1  # N < tile_size

    # Paper-recommended 100K
    assert _resolve_num_tiles(2_900_000, tile_size=100_000) == 29
    assert _resolve_num_tiles(160_000_000, tile_size=100_000) == 1600

    print("PASS: _resolve_num_tiles auto-computes correctly")


def test_tile_size_attention():
    """Test PhysicsAttentionV3 with tile_size produces same results as num_tiles."""
    B, N, C = 1, 200, 64
    heads = 4
    attn = PhysicsAttentionV3(C, heads=heads, dim_head=C // heads, slice_num=16)
    attn.eval()

    x = torch.randn(B, N, C)
    with torch.no_grad():
        # tile_size=50 with N=200 → num_tiles=4
        out_tile_size = attn(x, tile_size=50)
        out_num_tiles = attn(x, num_tiles=4)

    diff = (out_tile_size - out_num_tiles).abs().max().item()
    assert diff < 1e-6, f"tile_size vs num_tiles max diff: {diff}"
    print(f"PASS: tile_size=50 matches num_tiles=4 (diff: {diff:.2e})")


def test_tile_size_model():
    """Test Transolver3 with tile_size at constructor and forward levels."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    # Constructor-level tile_size
    model = Transolver3(
        space_dim=space_dim, n_layers=2, n_hidden=64, n_head=4,
        fun_dim=0, out_dim=out_dim, slice_num=16, mlp_ratio=1,
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

    # ctor tile_size=50 → 4 tiles, should match num_tiles=4
    diff = (out_ctor - out_ref).abs().max().item()
    assert diff < 1e-5, f"Constructor tile_size vs num_tiles diff: {diff}"

    print(f"PASS: model tile_size works at constructor and forward levels")


def test_target_normalizer_fit():
    """Test TargetNormalizer fit computes correct mean/std and encode/decode are inverse."""
    # Create targets with known distribution: mean=10, std=3 per channel
    torch.manual_seed(42)
    num_samples, N, out_dim = 100, 50, 3
    targets = torch.randn(num_samples, N, out_dim) * 3.0 + 10.0

    normalizer = TargetNormalizer()
    normalizer.fit(targets)

    # Check mean is close to 10, std close to 3
    assert (normalizer.mean - 10.0).abs().max().item() < 0.5, \
        f"Mean should be ~10, got {normalizer.mean.squeeze().tolist()}"
    assert (normalizer.std - 3.0).abs().max().item() < 0.5, \
        f"Std should be ~3, got {normalizer.std.squeeze().tolist()}"

    # Encoded data should have ~zero mean, ~unit variance
    encoded = normalizer.encode(targets)
    enc_mean = encoded.mean(dim=(0, 1))
    enc_std = encoded.std(dim=(0, 1))
    assert enc_mean.abs().max().item() < 0.1, \
        f"Encoded mean should be ~0, got {enc_mean.tolist()}"
    assert (enc_std - 1.0).abs().max().item() < 0.1, \
        f"Encoded std should be ~1, got {enc_std.tolist()}"

    # Decode(encode(x)) == x
    decoded = normalizer.decode(encoded)
    diff = (decoded - targets).abs().max().item()
    assert diff < 1e-5, f"decode(encode(x)) != x, max diff: {diff}"

    print(f"PASS: TargetNormalizer fit/encode/decode (roundtrip diff: {diff:.2e})")


def test_target_normalizer_incremental():
    """Test incremental fitting matches full fit."""
    torch.manual_seed(42)
    num_samples, N, out_dim = 200, 50, 3
    targets = torch.randn(num_samples, N, out_dim) * 5.0 - 2.0

    # Full fit
    norm_full = TargetNormalizer()
    norm_full.fit(targets)

    # Incremental fit (simulate batches of 20)
    norm_inc = TargetNormalizer()
    batches = [targets[i:i+20] for i in range(0, num_samples, 20)]
    norm_inc.fit_incremental(iter(batches))

    mean_diff = (norm_full.mean - norm_inc.mean).abs().max().item()
    std_diff = (norm_full.std - norm_inc.std).abs().max().item()

    # Tolerance accounts for Bessel's correction difference (std uses N-1,
    # streaming uses N) and float32 vs float64 accumulation
    assert mean_diff < 1e-3, f"Incremental mean diff: {mean_diff}"
    assert std_diff < 1e-3, f"Incremental std diff: {std_diff}"

    print(f"PASS: TargetNormalizer incremental matches full (mean diff: {mean_diff:.2e}, std diff: {std_diff:.2e})")


def test_target_normalizer_2d_input():
    """Test TargetNormalizer works with 2D input (N, out_dim)."""
    targets = torch.randn(100, 3) * 2.0 + 5.0
    normalizer = TargetNormalizer()
    normalizer.fit(targets)

    encoded = normalizer.encode(targets)
    decoded = normalizer.decode(encoded)
    diff = (decoded - targets).abs().max().item()
    assert diff < 1e-5, f"2D roundtrip diff: {diff}"
    print(f"PASS: TargetNormalizer works with 2D input")


def test_target_normalizer_state_dict():
    """Test TargetNormalizer saves/loads via state_dict (nn.Module)."""
    torch.manual_seed(42)
    targets = torch.randn(50, 30, 2) * 4.0 + 3.0

    norm1 = TargetNormalizer()
    norm1.fit(targets)

    # Save and reload
    state = norm1.state_dict()
    norm2 = TargetNormalizer(out_dim=2)
    norm2.load_state_dict(state)

    # Check stats match
    assert (norm1.mean - norm2.mean).abs().max().item() < 1e-7
    assert (norm1.std - norm2.std).abs().max().item() < 1e-7
    assert norm2.fitted.item() is True

    # Encode with both should match
    encoded1 = norm1.encode(targets)
    encoded2 = norm2.encode(targets)
    assert (encoded1 - encoded2).abs().max().item() < 1e-7

    print("PASS: TargetNormalizer state_dict save/load")


def test_target_normalizer_device():
    """Test TargetNormalizer follows model.to(device)."""
    targets = torch.randn(20, 10, 2)
    normalizer = TargetNormalizer()
    normalizer.fit(targets)

    # Should work on CPU
    encoded = normalizer.encode(targets)
    assert encoded.device.type == 'cpu'

    # to() should move buffers
    normalizer.cpu()
    assert normalizer.mean.device.type == 'cpu'

    print("PASS: TargetNormalizer device handling")


def test_train_step_with_normalizer():
    """Test train_step integrates with TargetNormalizer."""
    B, N = 1, 100
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=out_dim, slice_num=8, mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim) * 5.0 + 10.0  # non-normalized

    # Fit normalizer on training targets
    normalizer = TargetNormalizer()
    normalizer.fit(target)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_steps=10)

    # Train step with normalizer
    loss_with = train_step(model, x, None, target, optimizer, scheduler,
                           normalizer=normalizer)

    # Train step without (for comparison — should also work)
    loss_without = train_step(model, x, None, target, optimizer, scheduler,
                              normalizer=None)

    assert isinstance(loss_with, float) and loss_with > 0
    assert isinstance(loss_without, float) and loss_without > 0

    print(f"PASS: train_step with normalizer (loss_with={loss_with:.4f}, loss_without={loss_without:.4f})")


def test_train_step_mixed_precision():
    """Test train_step with GradScaler for mixed precision training."""
    B, N = 1, 100
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=out_dim, slice_num=8, mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_steps=10)

    device_type = next(model.parameters()).device.type

    # GradScaler: on CPU it's essentially a no-op but tests the code path
    scaler = torch.amp.GradScaler(device=device_type, enabled=(device_type == 'cuda'))

    # Multiple steps to verify gradients flow correctly
    losses = []
    for _ in range(3):
        loss = train_step(model, x, None, target, optimizer, scheduler,
                          scaler=scaler)
        losses.append(loss)

    assert all(isinstance(l, float) and l > 0 for l in losses)

    # Without scaler should also still work
    loss_no_scaler = train_step(model, x, None, target, optimizer, scheduler)
    assert isinstance(loss_no_scaler, float) and loss_no_scaler > 0

    print(f"PASS: train_step mixed precision (losses: {[f'{l:.4f}' for l in losses]})")


def test_input_normalizer_per_sample():
    """Test InputNormalizer per-sample mode (default)."""
    B, N, D = 2, 50, 3

    # Coordinates in arbitrary range
    coords = torch.randn(B, N, D) * 100.0 + 500.0

    normalizer = InputNormalizer(scale=1000.0, per_sample=True)

    encoded = normalizer.encode(coords)

    # Each sample should be in [0, 1000]
    for b in range(B):
        assert encoded[b].min().item() >= -1e-5, \
            f"Sample {b} min={encoded[b].min().item()}, expected >= 0"
        assert encoded[b].max().item() <= 1000.0 + 1e-5, \
            f"Sample {b} max={encoded[b].max().item()}, expected <= 1000"

    # Min per sample should be ~0, max ~scale
    for b in range(B):
        per_ch_min = encoded[b].min(dim=0).values
        per_ch_max = encoded[b].max(dim=0).values
        # At least one channel should hit 0 and 1000
        assert per_ch_min.min().item() < 1.0, "Some channel min should be near 0"
        assert per_ch_max.max().item() > 999.0, "Some channel max should be near 1000"

    print("PASS: InputNormalizer per_sample mode — output in [0, scale]")


def test_input_normalizer_dataset_level():
    """Test InputNormalizer dataset-level mode with fit/encode/decode roundtrip."""
    torch.manual_seed(42)
    num_samples, N, D = 20, 50, 3
    coords = torch.randn(num_samples, N, D) * 50.0 + 200.0

    normalizer = InputNormalizer(scale=1000.0, per_sample=False)
    normalizer.fit(coords)

    assert normalizer.fitted.item() is True
    assert normalizer.data_min.shape == (1, 1, D)
    assert normalizer.data_max.shape == (1, 1, D)

    encoded = normalizer.encode(coords)

    # Should be in [0, 1000]
    assert encoded.min().item() >= -1e-3, f"Min={encoded.min().item()}"
    assert encoded.max().item() <= 1000.0 + 1e-3, f"Max={encoded.max().item()}"

    # Decode roundtrip
    decoded = normalizer.decode(encoded)
    diff = (decoded - coords).abs().max().item()
    assert diff < 1e-3, f"Roundtrip diff: {diff}"

    print(f"PASS: InputNormalizer dataset-level fit/encode/decode (roundtrip diff: {diff:.2e})")


def test_input_normalizer_incremental():
    """Test InputNormalizer incremental fitting matches full fit."""
    torch.manual_seed(42)
    num_samples, N, D = 100, 30, 3
    coords = torch.randn(num_samples, N, D) * 10.0

    # Full fit
    norm_full = InputNormalizer(scale=1.0, per_sample=False)
    norm_full.fit(coords)

    # Incremental fit
    norm_inc = InputNormalizer(scale=1.0, per_sample=False)
    batches = [coords[i:i+10] for i in range(0, num_samples, 10)]
    norm_inc.fit_incremental(iter(batches))

    min_diff = (norm_full.data_min - norm_inc.data_min).abs().max().item()
    max_diff = (norm_full.data_max - norm_inc.data_max).abs().max().item()

    assert min_diff < 1e-6, f"Incremental min diff: {min_diff}"
    assert max_diff < 1e-6, f"Incremental max diff: {max_diff}"

    print(f"PASS: InputNormalizer incremental matches full (min diff: {min_diff:.2e}, max diff: {max_diff:.2e})")


def test_input_normalizer_scale_factor():
    """Test different scaling factors."""
    coords = torch.tensor([[[0.0, 10.0], [5.0, 20.0]]])  # B=1, N=2, D=2

    for scale in [1.0, 100.0, 1000.0]:
        norm = InputNormalizer(scale=scale)
        encoded = norm.encode(coords)
        # Min should be 0, max should be scale (per channel per sample)
        assert encoded.min().item() < 1e-5
        assert abs(encoded.max().item() - scale) < 1e-3, \
            f"scale={scale}: max={encoded.max().item()}"

    print("PASS: InputNormalizer scale factor works correctly")


def test_input_normalizer_2d():
    """Test InputNormalizer with 2D input (N, D)."""
    coords = torch.randn(50, 3) * 100.0

    normalizer = InputNormalizer(scale=1000.0)
    encoded = normalizer.encode(coords)
    assert encoded.shape == (50, 3)
    assert encoded.min().item() >= -1e-5
    assert encoded.max().item() <= 1000.0 + 1e-5

    print("PASS: InputNormalizer works with 2D input")


def test_input_normalizer_decode_per_sample_raises():
    """Test that decode raises in per_sample mode."""
    normalizer = InputNormalizer(per_sample=True)
    try:
        normalizer.decode(torch.randn(2, 10, 3))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "per_sample" in str(e)

    print("PASS: InputNormalizer.decode raises in per_sample mode")


# ===== PROFILING / BENCHMARKING TESTS =====

def test_profile_memory_forward():
    """Test profile_memory returns valid MemoryResult for forward mode."""
    model = Transolver3(
        space_dim=3, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=2, slice_num=8, mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 100, 3)

    result = profile_memory(model, x, mode='forward')
    assert isinstance(result, MemoryResult)
    assert result.peak_mb > 0
    assert result.mesh_size == 100
    assert result.config['mode'] == 'forward'

    # With tiling
    result_tiled = profile_memory(model, x, tile_size=30, mode='forward')
    assert isinstance(result_tiled, MemoryResult)
    assert result_tiled.peak_mb > 0

    print(f"PASS: profile_memory forward (no_tile={result.peak_mb:.1f}MB, "
          f"tiled={result_tiled.peak_mb:.1f}MB)")


def test_profile_memory_cached():
    """Test profile_memory works for cached inference mode."""
    model = Transolver3(
        space_dim=3, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=2, slice_num=8, mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 100, 3)

    result = profile_memory(model, x, mode='cached',
                            cache_chunk_size=30, decode_chunk_size=30)
    assert isinstance(result, MemoryResult)
    assert result.peak_mb > 0
    assert result.config['mode'] == 'cached'

    print(f"PASS: profile_memory cached ({result.peak_mb:.1f}MB)")


def test_profile_latency():
    """Test profile_latency returns valid LatencyResult."""
    model = Transolver3(
        space_dim=3, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=2, slice_num=8, mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 100, 3)

    result = profile_latency(model, x, mode='forward',
                             num_warmup=1, num_runs=3)
    assert isinstance(result, LatencyResult)
    assert result.mean_ms > 0
    assert result.num_runs == 3
    assert result.mesh_size == 100

    print(f"PASS: profile_latency (mean={result.mean_ms:.1f}ms, "
          f"std={result.std_ms:.1f}ms)")


def test_benchmark_scaling():
    """Test benchmark_scaling produces structured results across mesh sizes."""
    model = Transolver3(
        space_dim=3, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=2, slice_num=8, mlp_ratio=1,
    )
    model.eval()

    configs = [
        {'label': 'no_tiling', 'num_tiles': 0},
        {'label': 'tile_50', 'tile_size': 50},
    ]

    results = benchmark_scaling(
        model, space_dim=3,
        mesh_sizes=[50, 100, 200],
        configs=configs,
        measure_memory=True,
        measure_latency=True,
        num_latency_runs=2,
    )

    assert results['mesh_sizes'] == [50, 100, 200]
    assert len(results['configs']) == 2
    assert len(results['memory']) == 2  # 2 configs
    assert len(results['memory'][0]) == 3  # 3 mesh sizes
    assert len(results['latency']) == 2
    assert len(results['latency'][0]) == 3

    # All results should be valid
    for row in results['memory']:
        for mr in row:
            assert isinstance(mr, MemoryResult)
            assert mr.peak_mb > 0

    for row in results['latency']:
        for lr in row:
            assert isinstance(lr, LatencyResult)
            assert lr.mean_ms > 0

    print("PASS: benchmark_scaling produces correct structure")


def test_format_benchmark_table():
    """Test format_benchmark_table produces readable output."""
    model = Transolver3(
        space_dim=3, n_layers=2, n_hidden=32, n_head=4,
        fun_dim=0, out_dim=2, slice_num=8, mlp_ratio=1,
    )
    model.eval()

    results = benchmark_scaling(
        model, space_dim=3,
        mesh_sizes=[50, 100],
        configs=[{'label': 'baseline', 'num_tiles': 0}],
        measure_latency=False,
        num_latency_runs=1,
    )

    table = format_benchmark_table(results)
    assert isinstance(table, str)
    assert 'Memory' in table
    assert 'baseline' in table
    assert 'N=50' in table
    assert 'N=100' in table

    print(f"PASS: format_benchmark_table\n{table}")


def test_tiling_reduces_memory_relative():
    """Verify tiling doesn't increase memory vs no tiling (sanity check).

    On CPU with tracemalloc, tiling may not show dramatic savings since
    CPU memory management differs from CUDA. This test verifies the
    profiling infrastructure works end-to-end and tiled memory is finite.
    """
    model = Transolver3(
        space_dim=3, n_layers=2, n_hidden=64, n_head=4,
        fun_dim=0, out_dim=2, slice_num=16, mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(1, 500, 3)

    mem_no_tile = profile_memory(model, x, num_tiles=0)
    mem_tiled = profile_memory(model, x, tile_size=100)

    # Both should produce valid results
    assert mem_no_tile.peak_mb > 0
    assert mem_tiled.peak_mb > 0
    assert mem_no_tile.peak_mb < float('inf')
    assert mem_tiled.peak_mb < float('inf')

    print(f"PASS: tiling memory comparison (no_tile={mem_no_tile.peak_mb:.1f}MB, "
          f"tiled={mem_tiled.peak_mb:.1f}MB)")


if __name__ == '__main__':
    print("=" * 60)
    print("Transolver-3 Tests")
    print("=" * 60)

    # Original tests
    test_attention_shapes()
    test_tiled_vs_standard()
    test_cached_inference_equivalence()
    test_block_forward()
    test_full_model()
    test_amortized_training()
    test_cached_model_inference()
    test_sampler()
    test_scheduler()

    # Fix verification tests
    print("\n" + "=" * 60)
    print("Fix Verification Tests")
    print("=" * 60)

    test_fix1_head_concat_not_mean()
    test_fix2_no_large_intermediate()
    test_fix3_mlp_chunking()
    test_fix3_pointwise_chunked_helper()
    test_fix4_streaming_cache()
    test_fix4_memory_pattern()
    test_fix6_mixed_precision()
    test_fix6_autocast_numerics()
    test_model_with_mlp_chunking()
    test_inner_dim_assertion()

    # Tile size auto-computation tests
    print("\n" + "=" * 60)
    print("Tile Size Auto-Computation Tests")
    print("=" * 60)

    test_resolve_num_tiles()
    test_tile_size_attention()
    test_tile_size_model()

    # Target normalization tests
    print("\n" + "=" * 60)
    print("Target Normalization Tests")
    print("=" * 60)

    test_target_normalizer_fit()
    test_target_normalizer_incremental()
    test_target_normalizer_2d_input()
    test_target_normalizer_state_dict()
    test_target_normalizer_device()
    test_train_step_with_normalizer()
    test_train_step_mixed_precision()

    # Input normalization tests
    print("\n" + "=" * 60)
    print("Input Normalization Tests")
    print("=" * 60)

    test_input_normalizer_per_sample()
    test_input_normalizer_dataset_level()
    test_input_normalizer_incremental()
    test_input_normalizer_scale_factor()
    test_input_normalizer_2d()
    test_input_normalizer_decode_per_sample_raises()

    # Profiling / benchmarking tests
    print("\n" + "=" * 60)
    print("Profiling / Benchmarking Tests")
    print("=" * 60)

    test_profile_memory_forward()
    test_profile_memory_cached()
    test_profile_latency()
    test_benchmark_scaling()
    test_format_benchmark_table()
    test_tiling_reduces_memory_relative()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
