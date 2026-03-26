"""
Transolver-3 encoder block.

Fix #3: MLP and LayerNorm can be processed in tiles for large N,
avoiding O(N * hidden * mlp_ratio) peak memory in a single allocation.
"""

import torch
import torch.nn as nn
from transolver3.common import MLP
from transolver3.physics_attention_v3 import PhysicsAttentionV3


def _pointwise_chunked(fn, x, chunk_size):
    """Apply a pointwise function in chunks along dim=1 to limit peak memory.

    Args:
        fn: callable, e.g. lambda x: mlp(ln(x))
        x: (B, N, C) input
        chunk_size: max points per chunk

    Returns:
        (B, N, C') output
    """
    B, N, C = x.shape
    if chunk_size <= 0 or chunk_size >= N:
        return fn(x)

    outputs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        outputs.append(fn(x[:, start:end]))
    return torch.cat(outputs, dim=1)


class Transolver3Block(nn.Module):
    """Transolver-3 encoder block with optimized Physics-Attention.

    Fix #3: Added mlp_chunk_size parameter. When N exceeds this, MLP and
    LayerNorm are processed in chunks to avoid O(N * hidden * mlp_ratio)
    peak memory. Default 0 (no chunking, backward-compatible).
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            mlp_chunk_size=0,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.mlp_chunk_size = mlp_chunk_size
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttentionV3(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def _mlp_residual(self, fx):
        """LN + MLP residual, possibly chunked for large N."""
        def fn(x):
            return self.mlp(self.ln_2(x))
        return _pointwise_chunked(fn, fx, self.mlp_chunk_size) + fx

    def _last_layer_head(self, fx):
        """Final projection, possibly chunked."""
        def fn(x):
            return self.mlp2(self.ln_3(x))
        return _pointwise_chunked(fn, fx, self.mlp_chunk_size)

    def forward(self, fx, num_tiles=0, tile_size=0):
        fx = self.Attn(self.ln_1(fx), num_tiles=num_tiles, tile_size=tile_size) + fx
        fx = self._mlp_residual(fx)
        if self.last_layer:
            return self._last_layer_head(fx)
        return fx

    def compute_physical_state(self, fx):
        """Compute raw physical state for caching (inference).

        Returns (s_raw, d) that can be accumulated across mesh chunks.
        """
        return self.Attn.compute_physical_state(self.ln_1(fx))

    def compute_cached_state(self, s_raw, d):
        """Finalize cached state from accumulated (s_raw, d)."""
        return self.Attn.compute_cached_state(s_raw, d)

    def forward_from_cache(self, fx, cached_s_out):
        """Forward using pre-computed physical states (inference decoding).

        Paper Eq. 5: uses cached s'_out instead of recomputing attention.
        """
        fx = self.Attn.decode_from_cache(self.ln_1(fx), cached_s_out) + fx
        fx = self._mlp_residual(fx)
        if self.last_layer:
            return self._last_layer_head(fx)
        return fx
