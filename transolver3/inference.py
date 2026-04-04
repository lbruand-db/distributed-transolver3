"""
Physical State Caching and Full Mesh Decoding for Transolver-3.

During inference on industrial-scale meshes (>10^8 cells), processing the
full mesh in a single forward pass is memory-prohibitive. Transolver-3
introduces a two-phase decoupled inference:

  Phase 1 - Physical State Caching:
    Partition the full mesh into memory-compatible chunks, accumulate physical
    state contributions (s_raw, d) across chunks for each layer, then compute
    the cached state s'_out per layer.

  Phase 2 - Full Mesh Decoding:
    For each query point (or chunk of query points), decode predictions by
    computing slice weights and multiplying with the cached s'_out. This
    requires only O(1) incremental computation per point per layer.

Reference: Transolver-3 paper, Section 3.3, Figure 3.
"""

import torch
import torch.distributed as dist


def _decode_chunked(model, x_query, cache, decode_chunk_size, fx_query=None, T=None):
    """Decode predictions in chunks to limit GPU memory usage.

    Shared implementation for CachedInference and DistributedCachedInference.
    """
    N_q = x_query.shape[1]
    if decode_chunk_size is None or decode_chunk_size >= N_q:
        return model.decode_from_cache(x_query, cache, fx_query=fx_query, T=T)

    outputs = []
    for start in range(0, N_q, decode_chunk_size):
        end = min(start + decode_chunk_size, N_q)
        x_q = x_query[:, start:end]
        fx_q = fx_query[:, start:end] if fx_query is not None else None
        out = model.decode_from_cache(x_q, cache, fx_query=fx_q, T=T)
        outputs.append(out)

    return torch.cat(outputs, dim=1)


class CachedInference:
    """Manages two-phase inference for industrial-scale meshes.

    Usage:
        engine = CachedInference(model, cache_chunk_size=100000, decode_chunk_size=50000)
        output = engine.predict(x, fx=fx)

    Args:
        model: Transolver3 model instance
        cache_chunk_size: number of mesh points per chunk during state caching.
                         Smaller = less memory but more chunks. Default 100K per paper.
        decode_chunk_size: number of query points decoded per batch during
                          full mesh decoding. Default 50K.
        num_tiles: number of tiles for attention within each chunk (0=no tiling)
    """

    def __init__(self, model, cache_chunk_size=100000, decode_chunk_size=50000, num_tiles=0):
        self.model = model
        self.cache_chunk_size = cache_chunk_size
        self.decode_chunk_size = decode_chunk_size
        self.num_tiles = num_tiles

    @torch.no_grad()
    def predict(self, x, fx=None, T=None):
        """End-to-end prediction on a full mesh of arbitrary size.

        Args:
            x: (B, N, space_dim) full mesh coordinates
            fx: (B, N, fun_dim) optional input features
            T: optional timestep

        Returns:
            output: (B, N, out_dim) predictions
        """
        return self.model.full_mesh_inference(
            x,
            fx=fx,
            T=T,
            num_tiles=self.num_tiles,
            cache_chunk_size=self.cache_chunk_size,
            decode_chunk_size=self.decode_chunk_size,
        )

    @torch.no_grad()
    def build_cache(self, x, fx=None, T=None):
        """Build physical state cache (Phase 1 only).

        Useful when you want to decode different query sets against
        the same cached states (e.g., different evaluation resolutions).

        Args:
            x: (B, N, space_dim) full mesh coordinates
            fx: (B, N, fun_dim) optional features
            T: optional timestep

        Returns:
            cache: list of cached states, one per layer
        """
        return self.model.cache_physical_states(
            x,
            fx=fx,
            T=T,
            num_tiles=self.num_tiles,
            chunk_size=self.cache_chunk_size,
        )

    @torch.no_grad()
    def decode(self, x_query, cache, fx_query=None, T=None):
        """Decode predictions for query points using existing cache (Phase 2).

        Args:
            x_query: (B, N_q, space_dim) query coordinates
            cache: cached states from build_cache
            fx_query: (B, N_q, fun_dim) optional query features
            T: optional timestep

        Returns:
            output: (B, N_q, out_dim) predictions
        """
        return _decode_chunked(self.model, x_query, cache, self.decode_chunk_size, fx_query=fx_query, T=T)


class DistributedCachedInference:
    """Mesh-sharded two-phase inference across multiple GPUs.

    Each rank owns a disjoint partition of the mesh. During cache build,
    each rank computes local s_raw/d accumulators from its partition, then
    all-reduces them (only ~514 KB per layer). The resulting cache is
    identical on all ranks. During decode, each rank decodes its own
    partition of query points independently, then results are gathered.

    Falls back to single-GPU CachedInference when not distributed.

    Usage:
        engine = DistributedCachedInference(model, cache_chunk_size=100000)

        # Each rank passes only its LOCAL shard of the mesh
        cache = engine.build_cache(x_local)

        # Each rank passes only its LOCAL shard of query points
        local_pred = engine.decode(x_query_local, cache)

        # Or: predict with automatic gather across ranks
        # Each rank passes its local shard; returns full predictions on rank 0
        full_pred = engine.predict(x_local, total_points=N_total)

    Args:
        model: Transolver3 model instance (unwrapped, not DDP)
        cache_chunk_size: chunk size for local cache building
        decode_chunk_size: chunk size for local decoding
        num_tiles: tiles for attention within each chunk
    """

    def __init__(self, model, cache_chunk_size=100000, decode_chunk_size=50000, num_tiles=0):
        self.model = model
        self.cache_chunk_size = cache_chunk_size
        self.decode_chunk_size = decode_chunk_size
        self.num_tiles = num_tiles

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @torch.no_grad()
    def build_cache(self, x_local, fx_local=None, T=None):
        """Build physical state cache from this rank's mesh shard.

        Each rank processes its local partition of the mesh. The additive
        accumulators (s_raw, d) are all-reduced across ranks so every rank
        ends up with the identical cache — as if the full mesh were processed
        on a single GPU.

        All-reduce volume per layer: s_raw (B,H,M,C) + d (B,H,M) ≈ 514 KB.

        Args:
            x_local: (B, N_local, space_dim) this rank's mesh partition
            fx_local: (B, N_local, fun_dim) optional features for this partition
            T: optional timestep

        Returns:
            cache: list of cached states per layer (identical on all ranks)
        """
        if not self._is_distributed():
            # Fall back to single-GPU
            return self.model.cache_physical_states(
                x_local,
                fx=fx_local,
                T=T,
                num_tiles=self.num_tiles,
                chunk_size=self.cache_chunk_size,
            )

        def _allreduce_hook(s_raw_accum, d_accum):
            """All-reduce accumulators across ranks (s_raw ≈ 512 KB, d ≈ 2 KB)."""
            dist.all_reduce(s_raw_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(d_accum, op=dist.ReduceOp.SUM)
            return s_raw_accum, d_accum

        return self.model._cache_chunked(
            x_local, fx_local, T, self.num_tiles, self.cache_chunk_size, accumulator_hook=_allreduce_hook
        )

    @torch.no_grad()
    def decode(self, x_query_local, cache, fx_query_local=None, T=None):
        """Decode predictions for this rank's query points using the cache.

        Each rank decodes independently — no communication needed.

        Args:
            x_query_local: (B, N_q_local, space_dim) this rank's query points
            cache: cached states from build_cache (identical on all ranks)
            fx_query_local: optional features for query points
            T: optional timestep

        Returns:
            output: (B, N_q_local, out_dim) predictions for this rank's points
        """
        return _decode_chunked(self.model, x_query_local, cache, self.decode_chunk_size, fx_query=fx_query_local, T=T)

    @torch.no_grad()
    def predict(self, x_local, fx_local=None, T=None, gather=True):
        """End-to-end sharded prediction: cache build + decode + optional gather.

        Each rank passes its local mesh shard. The cache is built via
        all-reduce on the tiny accumulators. Each rank decodes its own
        shard. If gather=True, results are concatenated across ranks
        and returned on all ranks.

        Args:
            x_local: (B, N_local, space_dim) this rank's mesh shard
            fx_local: optional features
            T: optional timestep
            gather: if True, all_gather results into (B, N_total, out_dim)

        Returns:
            output: (B, N_local, out_dim) if gather=False
                    (B, N_total, out_dim) if gather=True
        """
        cache = self.build_cache(x_local, fx_local=fx_local, T=T)
        local_pred = self.decode(x_local, cache, fx_query_local=fx_local, T=T)

        if not gather or not self._is_distributed():
            return local_pred

        # Gather predictions from all ranks
        world_size = dist.get_world_size()
        # Each rank may have different N_local, so use all_gather with padding
        local_n = torch.tensor([local_pred.shape[1]], device=local_pred.device)
        all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
        dist.all_gather(all_n, local_n)
        max_n = max(n.item() for n in all_n)

        # Pad to max_n for uniform all_gather
        B, N_local, D = local_pred.shape
        if N_local < max_n:
            pad = torch.zeros(B, max_n - N_local, D, device=local_pred.device)
            padded = torch.cat([local_pred, pad], dim=1)
        else:
            padded = local_pred

        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)

        # Trim padding and concatenate in rank order
        parts = []
        for i, n in enumerate(all_n):
            parts.append(gathered[i][:, : n.item()])
        return torch.cat(parts, dim=1)
