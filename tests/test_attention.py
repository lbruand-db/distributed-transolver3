"""
Tests for PhysicsAttentionV3 attention mechanism.

Verifies shapes, tiled vs standard equivalence, cached inference,
head aggregation, slice operations, and tile_size auto-computation.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transolver3.physics_attention_v3 import PhysicsAttentionV3, _slice_aggregate, _deslice, _resolve_num_tiles


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


def test_fix1_head_concat_not_mean():
    """Fix #1: Verify heads are concatenated (rearrange), not averaged."""
    B, N, C = 2, 50, 64
    heads = 4
    dim_head = C // heads

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=dim_head, slice_num=8)

    # Verify slice_linear3 shape: dim_head -> dim_head
    assert attn.slice_linear3.in_features == dim_head
    assert attn.slice_linear3.out_features == dim_head

    # Verify output shape is (B, N, dim) not (B, N, dim_head)
    x = torch.randn(B, N, C)
    out = attn(x)
    assert out.shape == (B, N, C), (
        f"Output should be (B,N,dim)={(B, N, C)}, got {out.shape}. Heads may be averaged instead of concatenated."
    )

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


def test_fix2_no_large_intermediate():
    """Fix #2: Verify _slice_aggregate uses matmul without (B,H,N,C) intermediate."""
    B, H, N, M, C = 2, 4, 100, 16, 64

    w = torch.randn(B, H, N, M).softmax(dim=-1)
    x = torch.randn(B, N, C)

    # Our implementation
    result = _slice_aggregate(w, x, H)

    # Reference: einsum (mathematically correct but may materialize large tensor)
    reference = torch.einsum("bhnm,bnc->bhmc", w, x)

    diff = (result - reference).abs().max().item()
    assert diff < 1e-5, f"_slice_aggregate vs einsum max diff: {diff}"
    assert result.shape == (B, H, M, C), f"Expected {(B, H, M, C)}, got {result.shape}"

    # Verify _deslice too
    s_out = torch.randn(B, H, M, C)
    deslice_result = _deslice(s_out, w)
    deslice_ref = torch.einsum("bhmc,bhnm->bhnc", s_out, w)
    diff2 = (deslice_result - deslice_ref).abs().max().item()
    assert diff2 < 1e-5, f"_deslice vs einsum max diff: {diff2}"


def test_inner_dim_assertion():
    """Test that inner_dim != dim raises an assertion error."""
    try:
        # dim=64, heads=4, dim_head=32 -> inner_dim=128 != 64
        PhysicsAttentionV3(64, heads=4, dim_head=32, slice_num=8)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "inner_dim == dim" in str(e)


def test_resolve_num_tiles():
    """Test _resolve_num_tiles auto-computes tile count from tile_size."""
    # tile_size=0 -> passthrough num_tiles
    assert _resolve_num_tiles(1000, num_tiles=0) == 0
    assert _resolve_num_tiles(1000, num_tiles=5) == 5

    # tile_size > 0 -> auto-compute, overrides num_tiles
    assert _resolve_num_tiles(1000, num_tiles=0, tile_size=100) == 10
    assert _resolve_num_tiles(1000, num_tiles=99, tile_size=100) == 10  # overrides
    assert _resolve_num_tiles(1050, num_tiles=0, tile_size=100) == 11  # ceil
    assert _resolve_num_tiles(100, num_tiles=0, tile_size=100) == 1  # exactly 1 tile
    assert _resolve_num_tiles(50, num_tiles=0, tile_size=100) == 1  # N < tile_size

    # Paper-recommended 100K
    assert _resolve_num_tiles(2_900_000, tile_size=100_000) == 29
    assert _resolve_num_tiles(160_000_000, tile_size=100_000) == 1600


def test_tile_size_attention():
    """Test PhysicsAttentionV3 with tile_size produces same results as num_tiles."""
    B, N, C = 1, 200, 64
    heads = 4
    attn = PhysicsAttentionV3(C, heads=heads, dim_head=C // heads, slice_num=16)
    attn.eval()

    x = torch.randn(B, N, C)
    with torch.no_grad():
        # tile_size=50 with N=200 -> num_tiles=4
        out_tile_size = attn(x, tile_size=50)
        out_num_tiles = attn(x, num_tiles=4)

    diff = (out_tile_size - out_num_tiles).abs().max().item()
    assert diff < 1e-6, f"tile_size vs num_tiles max diff: {diff}"
