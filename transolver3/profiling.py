"""
Memory profiling and benchmarking for Transolver-3.

Paper Figure 6 shows precise memory consumption curves. This module
provides tooling to measure and verify memory savings from tiling,
cached inference, and mixed precision.

Supports both CUDA (torch.cuda memory stats) and CPU (tracemalloc)
backends. Produces structured results suitable for plotting.

Usage:
    from transolver3.profiling import profile_memory, profile_latency, benchmark_scaling

    # Single-point measurement
    result = profile_memory(model, x, num_tiles=4)
    print(f"Peak memory: {result['peak_mb']:.1f} MB")

    # Scaling benchmark (like paper Figure 6)
    results = benchmark_scaling(model, mesh_sizes=[1000, 5000, 10000],
                                configs=[{'num_tiles': 0}, {'tile_size': 1000}])
"""

import time
import torch
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class MemoryResult:
    """Result of a memory profiling run."""
    peak_mb: float
    allocated_mb: float
    config: dict = field(default_factory=dict)
    mesh_size: int = 0
    backend: str = "cpu"

    def __repr__(self):
        return (f"MemoryResult(peak={self.peak_mb:.1f}MB, "
                f"alloc={self.allocated_mb:.1f}MB, N={self.mesh_size})")


@dataclass
class LatencyResult:
    """Result of a latency profiling run."""
    mean_ms: float
    std_ms: float
    num_runs: int
    config: dict = field(default_factory=dict)
    mesh_size: int = 0

    def __repr__(self):
        return (f"LatencyResult(mean={self.mean_ms:.1f}ms, "
                f"std={self.std_ms:.1f}ms, N={self.mesh_size})")


@contextmanager
def _track_memory_cuda(device):
    """Context manager that tracks CUDA peak memory."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    alloc_before = torch.cuda.memory_allocated(device)
    yield
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    # Store results on the context manager object
    _track_memory_cuda._peak = peak
    _track_memory_cuda._alloc_before = alloc_before


@contextmanager
def _track_memory_cpu():
    """Context manager that tracks CPU memory via tracemalloc."""
    import tracemalloc
    tracemalloc.start()
    yield
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _track_memory_cpu._peak = peak


def profile_memory(model, x, fx=None, T=None, num_tiles=0, tile_size=0,
                   mode='forward', cache_chunk_size=None,
                   decode_chunk_size=None):
    """Profile peak memory for a single model run.

    Args:
        model: Transolver3 model
        x: (B, N, space_dim) input coordinates
        fx: optional input features
        T: optional timestep
        num_tiles: tiling parameter
        tile_size: tile size parameter (overrides num_tiles)
        mode: 'forward' for standard forward, 'cached' for cached inference
        cache_chunk_size: chunk size for cached inference
        decode_chunk_size: chunk size for cached inference decoding

    Returns:
        MemoryResult with peak and allocated memory
    """
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    N = x.shape[1]

    config = {
        'mode': mode,
        'num_tiles': num_tiles,
        'tile_size': tile_size,
    }

    if use_cuda:
        # Clear cache first
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            if mode == 'forward':
                _ = model(x, fx=fx, T=T, num_tiles=num_tiles, tile_size=tile_size)
            elif mode == 'cached':
                from transolver3.inference import CachedInference
                engine = CachedInference(
                    model,
                    cache_chunk_size=cache_chunk_size or N,
                    decode_chunk_size=decode_chunk_size or N,
                    num_tiles=num_tiles,
                )
                _ = engine.predict(x, fx=fx, T=T)

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)

        return MemoryResult(
            peak_mb=(peak) / 1024 / 1024,
            allocated_mb=(peak - mem_before) / 1024 / 1024,
            config=config,
            mesh_size=N,
            backend='cuda',
        )
    else:
        import tracemalloc
        # Run once to warm up (allocate parameters etc)
        with torch.no_grad():
            if mode == 'forward':
                _ = model(x, fx=fx, T=T, num_tiles=num_tiles, tile_size=tile_size)
            elif mode == 'cached':
                from transolver3.inference import CachedInference
                engine = CachedInference(
                    model,
                    cache_chunk_size=cache_chunk_size or N,
                    decode_chunk_size=decode_chunk_size or N,
                    num_tiles=num_tiles,
                )
                _ = engine.predict(x, fx=fx, T=T)

        # Now measure
        tracemalloc.start()
        with torch.no_grad():
            if mode == 'forward':
                _ = model(x, fx=fx, T=T, num_tiles=num_tiles, tile_size=tile_size)
            elif mode == 'cached':
                from transolver3.inference import CachedInference
                engine = CachedInference(
                    model,
                    cache_chunk_size=cache_chunk_size or N,
                    decode_chunk_size=decode_chunk_size or N,
                    num_tiles=num_tiles,
                )
                _ = engine.predict(x, fx=fx, T=T)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return MemoryResult(
            peak_mb=peak / 1024 / 1024,
            allocated_mb=peak / 1024 / 1024,
            config=config,
            mesh_size=N,
            backend='cpu_tracemalloc',
        )


def profile_latency(model, x, fx=None, T=None, num_tiles=0, tile_size=0,
                    mode='forward', num_warmup=2, num_runs=5,
                    cache_chunk_size=None, decode_chunk_size=None):
    """Profile latency (wall-clock time) for a model run.

    Args:
        model: Transolver3 model
        x: (B, N, space_dim) input
        fx: optional input features
        T: optional timestep
        num_tiles: tiling parameter
        tile_size: tile size parameter
        mode: 'forward' or 'cached'
        num_warmup: warmup iterations (not measured)
        num_runs: measured iterations

    Returns:
        LatencyResult with mean/std in milliseconds
    """
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    N = x.shape[1]

    def _run():
        with torch.no_grad():
            if mode == 'forward':
                _ = model(x, fx=fx, T=T, num_tiles=num_tiles, tile_size=tile_size)
            elif mode == 'cached':
                from transolver3.inference import CachedInference
                engine = CachedInference(
                    model,
                    cache_chunk_size=cache_chunk_size or N,
                    decode_chunk_size=decode_chunk_size or N,
                    num_tiles=num_tiles,
                )
                _ = engine.predict(x, fx=fx, T=T)

    # Warmup
    for _ in range(num_warmup):
        _run()

    # Measure
    times = []
    for _ in range(num_runs):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _run()
        if use_cuda:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times_t = torch.tensor(times)
    return LatencyResult(
        mean_ms=times_t.mean().item(),
        std_ms=times_t.std().item() if len(times) > 1 else 0.0,
        num_runs=num_runs,
        config={'mode': mode, 'num_tiles': num_tiles, 'tile_size': tile_size},
        mesh_size=N,
    )


def benchmark_scaling(model, space_dim=3, mesh_sizes=None, configs=None,
                      measure_memory=True, measure_latency=True,
                      num_latency_runs=5, batch_size=1, device=None):
    """Benchmark memory and latency across mesh sizes and configurations.

    Produces data similar to paper Figure 6 — memory vs mesh size for
    different tiling configurations.

    Args:
        model: Transolver3 model
        space_dim: spatial dimension for synthetic inputs
        mesh_sizes: list of N values to test (default: geometric sequence)
        configs: list of dicts with keys like 'num_tiles', 'tile_size',
                 'mode', 'cache_chunk_size'. Each config is benchmarked
                 at each mesh size. Default: no tiling vs tiling.
        measure_memory: whether to profile memory
        measure_latency: whether to profile latency
        num_latency_runs: runs per latency measurement
        batch_size: batch size
        device: override device (default: model's device)

    Returns:
        dict with keys:
            'mesh_sizes': list of N
            'configs': list of config dicts
            'memory': list of lists of MemoryResult (configs × mesh_sizes)
            'latency': list of lists of LatencyResult (configs × mesh_sizes)
    """
    if mesh_sizes is None:
        mesh_sizes = [100, 500, 1000, 5000, 10000]

    if configs is None:
        configs = [
            {'label': 'no_tiling', 'num_tiles': 0, 'tile_size': 0},
            {'label': 'tile_1k', 'num_tiles': 0, 'tile_size': 1000},
        ]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    memory_results = []
    latency_results = []

    for config in configs:
        label = config.get('label', str(config))
        mem_row = []
        lat_row = []

        for N in mesh_sizes:
            x = torch.randn(batch_size, N, space_dim, device=device)
            nt = config.get('num_tiles', 0)
            ts = config.get('tile_size', 0)
            mode = config.get('mode', 'forward')
            ccs = config.get('cache_chunk_size', None)
            dcs = config.get('decode_chunk_size', None)

            if measure_memory:
                try:
                    mr = profile_memory(
                        model, x, num_tiles=nt, tile_size=ts, mode=mode,
                        cache_chunk_size=ccs, decode_chunk_size=dcs,
                    )
                    mr.config['label'] = label
                    mem_row.append(mr)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    mem_row.append(MemoryResult(
                        peak_mb=float('inf'), allocated_mb=float('inf'),
                        config={'label': label, 'error': 'OOM'},
                        mesh_size=N, backend='oom',
                    ))

            if measure_latency:
                try:
                    lr = profile_latency(
                        model, x, num_tiles=nt, tile_size=ts, mode=mode,
                        num_runs=num_latency_runs,
                        cache_chunk_size=ccs, decode_chunk_size=dcs,
                    )
                    lr.config['label'] = label
                    lat_row.append(lr)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    lat_row.append(LatencyResult(
                        mean_ms=float('inf'), std_ms=0.0, num_runs=0,
                        config={'label': label, 'error': 'OOM'},
                        mesh_size=N,
                    ))

        memory_results.append(mem_row)
        latency_results.append(lat_row)

    return {
        'mesh_sizes': mesh_sizes,
        'configs': configs,
        'memory': memory_results,
        'latency': latency_results,
    }


def format_benchmark_table(results):
    """Format benchmark results as a readable table string.

    Args:
        results: dict from benchmark_scaling

    Returns:
        str: formatted table
    """
    lines = []
    mesh_sizes = results['mesh_sizes']
    configs = results['configs']

    if results.get('memory') and results['memory'][0]:
        lines.append("=== Memory (Peak MB) ===")
        header = f"{'Config':<20}" + "".join(f"{'N='+str(n):>12}" for n in mesh_sizes)
        lines.append(header)
        lines.append("-" * len(header))
        for i, config in enumerate(configs):
            label = config.get('label', f'config_{i}')
            row = f"{label:<20}"
            for mr in results['memory'][i]:
                if mr.peak_mb == float('inf'):
                    row += f"{'OOM':>12}"
                else:
                    row += f"{mr.peak_mb:>11.1f}M"
            lines.append(row)

    if results.get('latency') and results['latency'][0]:
        lines.append("")
        lines.append("=== Latency (ms) ===")
        header = f"{'Config':<20}" + "".join(f"{'N='+str(n):>12}" for n in mesh_sizes)
        lines.append(header)
        lines.append("-" * len(header))
        for i, config in enumerate(configs):
            label = config.get('label', f'config_{i}')
            row = f"{label:<20}"
            for lr in results['latency'][i]:
                if lr.mean_ms == float('inf'):
                    row += f"{'OOM':>12}"
                else:
                    row += f"{lr.mean_ms:>10.1f}ms"
            lines.append(row)

    return "\n".join(lines)
