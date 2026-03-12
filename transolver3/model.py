"""
Transolver-3 model for irregular meshes.

Supports:
  - Standard forward pass (small meshes)
  - Geometry slice tiling (medium meshes, single GPU)
  - Geometry amortized training (large meshes, subset per iteration)
  - Physical state caching + full mesh decoding (inference on industrial-scale meshes)

Fixes vs initial implementation:
  - #4: _cache_chunked streams chunks one at a time on GPU, offloading
        intermediate features to CPU. Never stores all N features on GPU.
  - #5: get_grid uses fixed [0,1] linspace, so it's chunk-safe by design.
  - #6: Mixed precision via torch.amp autocast context manager.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from timm.layers import trunc_normal_

from transolver3.common import MLP
from transolver3.transolver3_block import Transolver3Block


class Transolver3(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 num_tiles=0,
                 tile_size=0,
                 mlp_chunk_size=0,
                 ):
        super(Transolver3, self).__init__()
        self.__name__ = 'Transolver3'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.num_tiles = num_tiles
        self.tile_size = tile_size

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2,
                                  n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2,
                                  n_hidden, n_layers=0, res=False, act=act)

        if Time_Input:
            from PDE_Solving_StandardBenchmark.model.Embedding import timestep_embedding
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                nn.Linear(n_hidden, n_hidden),
            )

        self.blocks = nn.ModuleList([
            Transolver3Block(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(i == n_layers - 1),
                mlp_chunk_size=mlp_chunk_size,
            )
            for i in range(n_layers)
        ])

        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        """Compute distance-based positional encoding from a reference grid.

        Fix #5: Uses fixed [0,1] linspace for the reference grid, making it
        chunk-safe — the reference grid is independent of input min/max.
        """
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(x.device).reshape(
            batchsize, self.ref * self.ref, 2
        )
        pos = torch.sqrt(
            torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
        ).reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def _preprocess(self, x, fx=None, T=None):
        """Shared preprocessing: positional encoding, feature projection, time embedding."""
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None and self.Time_Input:
            from PDE_Solving_StandardBenchmark.model.Embedding import timestep_embedding
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        return fx

    def forward(self, x, fx=None, T=None, num_tiles=None, tile_size=None,
                subset_indices=None):
        """Forward pass.

        Args:
            x: (B, N, space_dim) spatial coordinates (or positional features)
            fx: (B, N, fun_dim) optional function values at mesh points
            T: optional timestep for time-dependent problems
            num_tiles: override per-call tiling (0=no tiling, >1=tiled)
            tile_size: target points per tile. If >0, overrides num_tiles.
                       Paper recommends 100_000 (Table 5).
            subset_indices: (n,) indices for geometry amortized training.
                           If provided, only these mesh points are used.

        Returns:
            output: (B, N', out_dim) predictions at mesh points
        """
        tiles = num_tiles if num_tiles is not None else self.num_tiles
        ts = tile_size if tile_size is not None else self.tile_size

        # Geometry amortized training: subsample mesh
        if subset_indices is not None:
            x = x[:, subset_indices]
            if fx is not None:
                fx = fx[:, subset_indices]

        fx = self._preprocess(x, fx, T)

        for block in self.blocks:
            fx = block(fx, num_tiles=tiles, tile_size=ts)

        return fx

    # --- Physical State Caching (Inference) ---

    @torch.no_grad()
    def cache_physical_states(self, x, fx=None, T=None, num_tiles=0,
                              chunk_size=None):
        """Build physical state cache from the full mesh.

        Processes the mesh in chunks if chunk_size is specified, accumulating
        physical states layer-by-layer per paper Figure 3(a).

        Fix #4: When chunked, intermediate features are offloaded to CPU
        between layers. Only one chunk is on GPU at a time, so GPU memory
        is O(chunk_size * hidden_dim) not O(N * hidden_dim).

        Args:
            x: (B, N, space_dim) full mesh coordinates
            fx: (B, N, fun_dim) optional full mesh features
            T: optional timestep
            num_tiles: tiling within each chunk's attention
            chunk_size: if set, partition the mesh into chunks of this size
                       for memory-efficient state accumulation

        Returns:
            cache: list of s_out tensors, one per layer
        """
        N = x.shape[1]

        if chunk_size is None or chunk_size >= N:
            return self._cache_full(x, fx, T, num_tiles)
        else:
            return self._cache_chunked(x, fx, T, num_tiles, chunk_size)

    @torch.no_grad()
    def _cache_full(self, x, fx, T, num_tiles):
        """Cache states by processing the full mesh in one pass."""
        fx_current = self._preprocess(x, fx, T)
        cache = []

        for block in self.blocks:
            s_raw, d = block.compute_physical_state(fx_current)
            s_out = block.compute_cached_state(s_raw, d)
            cache.append(s_out)
            fx_current = block(fx_current, num_tiles=num_tiles)

        return cache

    @torch.no_grad()
    def _cache_chunked(self, x, fx, T, num_tiles, chunk_size):
        """Cache states by streaming chunks through the model.

        Fix #4: Instead of storing all preprocessed features on GPU,
        intermediate chunk features are kept on CPU. Only one chunk
        is moved to GPU at a time for processing.

        Memory: O(chunk_size * hidden_dim) GPU + O(N * hidden_dim) CPU
        instead of O(N * hidden_dim) GPU.
        """
        B, N, _ = x.shape
        num_chunks = math.ceil(N / chunk_size)
        device = x.device

        # Step 1: Preprocess each chunk, storing results on CPU
        chunks_fx_cpu = []
        for k in range(num_chunks):
            start = k * chunk_size
            end = min(start + chunk_size, N)
            x_k = x[:, start:end]
            fx_k = fx[:, start:end] if fx is not None else None
            chunk_fx = self._preprocess(x_k, fx_k, T)
            chunks_fx_cpu.append(chunk_fx.cpu())
            del chunk_fx  # free GPU memory

        cache = []
        for layer_idx, block in enumerate(self.blocks):
            # Phase A: Accumulate s_raw and d across chunks
            s_raw_accum = None
            d_accum = None

            for k in range(num_chunks):
                chunk_fx_gpu = chunks_fx_cpu[k].to(device)
                s_raw_k, d_k = block.compute_physical_state(chunk_fx_gpu)
                if s_raw_accum is None:
                    s_raw_accum = s_raw_k
                    d_accum = d_k
                else:
                    s_raw_accum = s_raw_accum + s_raw_k
                    d_accum = d_accum + d_k
                del chunk_fx_gpu  # free GPU memory

            # Finalize cached state for this layer
            s_out = block.compute_cached_state(s_raw_accum, d_accum)
            cache.append(s_out)
            del s_raw_accum, d_accum

            # Phase B: Advance all chunks through this layer, storing on CPU
            for k in range(num_chunks):
                chunk_fx_gpu = chunks_fx_cpu[k].to(device)
                advanced = block(chunk_fx_gpu, num_tiles=num_tiles)
                chunks_fx_cpu[k] = advanced.cpu()
                del chunk_fx_gpu, advanced  # free GPU memory

        return cache

    @torch.no_grad()
    def decode_from_cache(self, x_query, cache, fx_query=None, T=None):
        """Decode predictions for query points using cached physical states.

        Paper Figure 3(b): for each layer, compute slice weights for the
        query point and multiply by the cached s'_out.

        Args:
            x_query: (B, N_q, space_dim) query point coordinates
            cache: list of cached s_out tensors from cache_physical_states
            fx_query: (B, N_q, fun_dim) optional query features
            T: optional timestep

        Returns:
            output: (B, N_q, out_dim) predictions
        """
        fx = self._preprocess(x_query, fx_query, T)

        for block, cached_s_out in zip(self.blocks, cache):
            fx = block.forward_from_cache(fx, cached_s_out)

        return fx

    @torch.no_grad()
    def full_mesh_inference(self, x, fx=None, T=None, num_tiles=0,
                            cache_chunk_size=None, decode_chunk_size=None):
        """End-to-end inference on a full industrial-scale mesh.

        Two phases:
          1. Build physical state cache (possibly chunked)
          2. Decode all mesh points using cache (possibly chunked)

        Args:
            x: (B, N, space_dim) full mesh coordinates
            fx: (B, N, fun_dim) optional features
            T: optional timestep
            num_tiles: tiling for attention within chunks
            cache_chunk_size: chunk size for building cache (None=full mesh)
            decode_chunk_size: chunk size for decoding (None=full mesh)

        Returns:
            output: (B, N, out_dim) predictions for all mesh points
        """
        # Phase 1: Build cache
        cache = self.cache_physical_states(
            x, fx, T, num_tiles=num_tiles, chunk_size=cache_chunk_size
        )

        N = x.shape[1]
        if decode_chunk_size is None or decode_chunk_size >= N:
            return self.decode_from_cache(x, cache, fx_query=fx, T=T)

        # Phase 2: Decode in chunks
        outputs = []
        for start in range(0, N, decode_chunk_size):
            end = min(start + decode_chunk_size, N)
            x_q = x[:, start:end]
            fx_q = fx[:, start:end] if fx is not None else None
            out_chunk = self.decode_from_cache(x_q, cache, fx_query=fx_q, T=T)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=1)
