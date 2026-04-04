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
Optimized Physics-Attention mechanism for Transolver-3.

Key innovations over Transolver v1 (Physics_Attention_Irregular_Mesh):
  1. Faster Slice & Deslice: Linear projections moved from O(N) mesh domain
     to O(M) slice domain via matrix multiplication associativity.
  2. Geometry Slice Tiling: Input partitioned into tiles processed sequentially
     with gradient checkpointing, reducing peak memory from O(NM) to O(N_t*M).
  3. Physical State Caching: Decoupled inference separating state estimation
     from field decoding for industrial-scale meshes.

Reference: Transolver-3 paper (arXiv:2602.04940), Section 3.1-3.3, Algorithm 1.

Fixes vs initial implementation:
  - #1: Head aggregation uses rearrange (concat) not mean. slice_linear3 is
        dim_head->dim_head, output heads are concatenated like v1.
  - #2: einsum replaced by per-head matmul to avoid materializing (B,H,N,C).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


def _resolve_num_tiles(N, num_tiles=0, tile_size=0):
    """Compute num_tiles from either num_tiles or tile_size.

    Paper Table 5: "A tile size of 100k serves as an ideal choice."

    Args:
        N: number of mesh points
        num_tiles: explicit tile count (0 or 1 = no tiling)
        tile_size: target points per tile (0 = disabled). If >0,
                   overrides num_tiles: num_tiles = ceil(N / tile_size).

    Returns:
        int: resolved num_tiles (0 means no tiling)
    """
    if tile_size > 0:
        return math.ceil(N / tile_size)
    return num_tiles


def _slice_aggregate(w, x, heads):
    """Memory-efficient w^T @ x without materializing (B,H,N,C).

    Computes s_raw[b,h,m,c] = sum_n w[b,h,n,m] * x[b,n,c] per head
    using batched matmul: (M,N) @ (N,C) = (M,C) for each (b,h).

    Args:
        w: (B, H, N, M) slice weights
        x: (B, N, C) input features

    Returns:
        s_raw: (B, H, M, C)
    """
    B, H, N, M = w.shape
    # w transposed: (B, H, M, N), x expanded: (B, 1, N, C) -> (B, H, N, C)
    # matmul: (B, H, M, N) @ (B, H, N, C) -> (B, H, M, C)
    # Expand x without materializing the full (B,H,N,C) tensor in memory:
    # torch.matmul broadcasts x (B,1,N,C) against w_t (B,H,M,N) correctly.
    w_t = w.transpose(2, 3)  # B, H, M, N
    x_expanded = x.unsqueeze(1)  # B, 1, N, C — broadcast, not materialized
    s_raw = torch.matmul(w_t, x_expanded)  # B, H, M, C
    return s_raw


def _deslice(s_out, w):
    """Memory-efficient deslice: w @ s_out without (B,H,N,dim) overhead.

    Args:
        s_out: (B, H, M, dim_head)
        w: (B, H, N, M) slice weights

    Returns:
        x_out: (B, H, N, dim_head)
    """
    # (B, H, N, M) @ (B, H, M, dim_head) -> (B, H, N, dim_head)
    return torch.matmul(w, s_out)


class PhysicsAttentionV3(nn.Module):
    """Optimized Physics-Attention for irregular meshes.

    Compared to v1, this eliminates two O(N)-domain linear projections:
      - in_project_x / in_project_fx are replaced by slice_linear1 operating
        on M slice tokens (M << N).
      - to_out is replaced by slice_linear3 operating on M slice tokens.

    The slice weight computation (in_project_slice) now operates directly on
    the raw input x in dim-space, not on a pre-projected dim_head-space.

    Head aggregation: each head outputs dim_head features after deslice,
    heads are concatenated (rearrange) to produce inner_dim = heads * dim_head = dim.
    Since inner_dim == dim in the standard config, no final N-domain linear is needed.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        assert inner_dim == dim, (
            f"Transolver-3 requires inner_dim == dim (heads*dim_head={inner_dim} != dim={dim}). "
            f"This is needed because to_out is absorbed into slice_linear3."
        )
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_p = dropout

        # Learnable temperature for slice weight sharpness
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # Slice weight projection: raw x (dim) -> per-head slice weights
        # In v1: in_project_x(dim->inner_dim) then in_project_slice(dim_head->slice_num)
        # In v3: single projection from dim directly to heads*slice_num
        self.in_project_slice = nn.Linear(dim, heads * slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Slice-domain Linear1: replaces in_project_fx but operates on M tokens not N
        # Projects from raw feature dim (dim) to per-head dim (dim_head)
        self.slice_linear1 = nn.Linear(dim, dim_head)

        # Attention Q/K/V projections (unchanged from v1, operate on M slice tokens)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # Slice-domain Linear3: replaces to_out but operates on M tokens not N
        # FIX #1: dim_head -> dim_head (not dim_head -> dim).
        # After deslice, heads are concatenated via rearrange to get dim.
        self.slice_linear3 = nn.Linear(dim_head, dim_head)
        self.out_dropout = nn.Dropout(dropout)

    def _compute_slice_weights(self, x):
        """Compute slice weights from raw input x.

        Args:
            x: (B, N, C) raw input features

        Returns:
            w: (B, H, N, M) normalized slice weights
        """
        B, N, C = x.shape
        logits = self.in_project_slice(x)  # B, N, H*M
        logits = logits.reshape(B, N, self.heads, self.slice_num)
        logits = logits.permute(0, 2, 1, 3).contiguous()  # B, H, N, M

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        w = self.softmax(logits / temperature)  # B, H, N, M
        return w

    def forward(self, x, num_tiles=0, tile_size=0):
        """Forward pass with optional geometry slice tiling.

        Args:
            x: (B, N, C) input features where N is mesh resolution
            num_tiles: number of tiles for memory-efficient processing.
                       0 or 1 = no tiling (standard path).
                       >1 = tiled processing with gradient checkpointing.
            tile_size: target points per tile. If >0, overrides num_tiles.
                       Paper recommends 100_000 (Table 5).

        Returns:
            x_out: (B, N, C) output features
        """
        N = x.shape[1]
        tiles = _resolve_num_tiles(N, num_tiles, tile_size)
        if tiles > 1:
            return self._forward_tiled(x, tiles)
        return self._forward_standard(x)

    def _forward_standard(self, x):
        """Standard forward pass (no tiling). Paper Eq. 3."""
        B, N, C = x.shape

        # (1) Faster Slice
        w = self._compute_slice_weights(x)  # B, H, N, M
        d = w.sum(dim=2)  # B, H, M

        # FIX #2: use matmul instead of einsum to avoid (B,H,N,C) intermediate
        s_raw = _slice_aggregate(w, x, self.heads)  # B, H, M, C
        s = self.slice_linear1(s_raw / (d[..., None] + 1e-5))  # B, H, M, dim_head

        # (2) Self-attention among slice tokens
        q, k, v = self.to_q(s), self.to_k(s), self.to_v(s)
        s_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )  # B, H, M, dim_head

        # (3) Faster Deslice
        s_out = self.slice_linear3(s_out)  # B, H, M, dim_head

        # FIX #1: use matmul for deslice, then rearrange (concat heads) not mean
        x_out = _deslice(s_out, w)  # B, H, N, dim_head
        x_out = rearrange(x_out, "b h n d -> b n (h d)")  # B, N, dim
        return self.out_dropout(x_out)

    def _forward_tiled(self, x, num_tiles):
        """Tiled forward pass with gradient checkpointing. Paper Algorithm 1."""
        B, N, C = x.shape
        tile_size = math.ceil(N / num_tiles)
        device = x.device

        s_raw_accum = torch.zeros(B, self.heads, self.slice_num, C, device=device)
        d_accum = torch.zeros(B, self.heads, self.slice_num, device=device)

        # Phase 1: Accumulate slice contributions from each tile
        for t in range(num_tiles):
            start = t * tile_size
            end = min(start + tile_size, N)
            x_t = x[:, start:end]

            def _tile_slice_and_aggregate(x_tile):
                w_tile = self._compute_slice_weights(x_tile)  # B, H, N_t, M
                # FIX #2: per-head matmul
                s_raw_tile = _slice_aggregate(w_tile, x_tile, self.heads)
                d_tile = w_tile.sum(dim=2)  # B, H, M
                return s_raw_tile, d_tile

            if self.training:
                s_raw_t, d_t = checkpoint(_tile_slice_and_aggregate, x_t, use_reentrant=False)
            else:
                s_raw_t, d_t = _tile_slice_and_aggregate(x_t)

            s_raw_accum = s_raw_accum + s_raw_t
            d_accum = d_accum + d_t

        # Phase 2: Slice-domain operations (on M tokens only)
        s = self.slice_linear1(s_raw_accum / (d_accum[..., None] + 1e-5))
        q, k, v = self.to_q(s), self.to_k(s), self.to_v(s)
        s_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        s_out = self.slice_linear3(s_out)  # B, H, M, dim_head

        # Phase 3: Deslice per tile
        outputs = []
        for t in range(num_tiles):
            start = t * tile_size
            end = min(start + tile_size, N)
            x_t = x[:, start:end]

            def _tile_deslice(x_tile, s_out_fixed):
                w_tile = self._compute_slice_weights(x_tile)
                # FIX #1 + #2: matmul deslice + rearrange
                x_out_tile = _deslice(s_out_fixed, w_tile)  # B, H, N_t, dim_head
                return rearrange(x_out_tile, "b h n d -> b n (h d)")

            if self.training:
                x_out_t = checkpoint(_tile_deslice, x_t, s_out, use_reentrant=False)
            else:
                x_out_t = _tile_deslice(x_t, s_out)

            outputs.append(x_out_t)

        x_out = torch.cat(outputs, dim=1)  # B, N, C
        return self.out_dropout(x_out)

    # --- Physical State Caching (Inference) ---

    @torch.no_grad()
    def compute_physical_state(self, x, num_tiles=0, tile_size=0):
        """Compute raw physical state accumulators from input features.

        Args:
            x: (B, N, C) input features (possibly a chunk of the full mesh)
            num_tiles: tiling for memory efficiency within this chunk
            tile_size: target points per tile (overrides num_tiles if >0)

        Returns:
            s_raw: (B, H, M, C) raw aggregated features (for accumulation)
            d: (B, H, M) normalization diagonal (for accumulation)
        """
        B, N, C = x.shape
        num_tiles = _resolve_num_tiles(N, num_tiles, tile_size)

        if num_tiles > 1:
            tile_size = math.ceil(N / num_tiles)
            s_raw = torch.zeros(B, self.heads, self.slice_num, C, device=x.device)
            d = torch.zeros(B, self.heads, self.slice_num, device=x.device)
            for t in range(num_tiles):
                start = t * tile_size
                end = min(start + tile_size, N)
                x_t = x[:, start:end]
                w_t = self._compute_slice_weights(x_t)
                s_raw = s_raw + _slice_aggregate(w_t, x_t, self.heads)
                d = d + w_t.sum(dim=2)
        else:
            w = self._compute_slice_weights(x)
            s_raw = _slice_aggregate(w, x, self.heads)
            d = w.sum(dim=2)

        return s_raw, d

    @torch.no_grad()
    def compute_cached_state(self, s_raw, d):
        """From accumulated s_raw and d, compute the final cached state.

        Args:
            s_raw: (B, H, M, C) accumulated raw slice features
            d: (B, H, M) accumulated normalization

        Returns:
            s_out: (B, H, M, dim_head) the cached physical state
        """
        s = self.slice_linear1(s_raw / (d[..., None] + 1e-5))
        q, k, v = self.to_q(s), self.to_k(s), self.to_v(s)
        s_out = F.scaled_dot_product_attention(q, k, v)
        s_out = self.slice_linear3(s_out)  # B, H, M, dim_head
        return s_out

    @torch.no_grad()
    def decode_from_cache(self, x_query, cached_s_out):
        """Decode predictions for query points using cached physical states.

        Paper Eq. 5: w^(l) = Softmax(Linear2(x^(l))), x_out = w * s'_out

        Args:
            x_query: (B, N_q, C) query point features
            cached_s_out: (B, H, M, dim_head) cached state

        Returns:
            x_out: (B, N_q, C) decoded output
        """
        w = self._compute_slice_weights(x_query)  # B, H, N_q, M
        # FIX #1 + #2: matmul deslice + rearrange
        x_out = _deslice(cached_s_out, w)  # B, H, N_q, dim_head
        x_out = rearrange(x_out, "b h n d -> b n (h d)")  # B, N_q, dim
        return x_out
