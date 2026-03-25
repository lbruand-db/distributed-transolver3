#!/usr/bin/env python3
"""
Transolver-3 GPU Memory Benchmark
==================================
Measures peak GPU memory across the 3-phase pipeline at various mesh sizes:
  Phase 1: Training (amortized subset)
  Phase 2: Cache Build (chunked physical state caching)
  Phase 3: Decode (chunked query decoding)

Generates synthetic data matching DrivAer ML surface structure:
  x:      (B, N, 22)  =  coords(3) + normals(3) + params(16)
  target: (B, N, 4)   =  pressure(1) + wall_shear(3)

Usage:
  python gpu_memory_benchmark.py                    # auto-detect GPU
  python gpu_memory_benchmark.py --gpu_type a10g    # force profile
  GPU_TYPE=a100_40 python gpu_memory_benchmark.py   # env var
"""

import os
import sys
import json
import time
import gc
import traceback

# --- Path setup: find transolver3 package ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Databricks spark_python_task runs via exec(), __file__ is not defined.
    # The bundle uploads files under .bundle/<name>/<target>/files/
    SCRIPT_DIR = os.getcwd()
    # Look for the workspace bundle path
    for candidate in sys.argv:
        if "files/benchmarks" in candidate:
            SCRIPT_DIR = os.path.dirname(candidate)
            break

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Databricks workspace bundle path fallback
# DABs uploads to /Workspace/Users/<user>/.bundle/<name>/<target>/files/
if not os.path.isdir(os.path.join(REPO_ROOT, "transolver3")):
    # Try parent dirs until we find transolver3/
    test_dir = SCRIPT_DIR
    for _ in range(5):
        test_dir = os.path.dirname(test_dir)
        if os.path.isdir(os.path.join(test_dir, "transolver3")):
            REPO_ROOT = test_dir
            sys.path.insert(0, REPO_ROOT)
            break

# einops and timm are installed via DAB job libraries or pip

import torch
import torch.nn as nn

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler,
    train_step,
    create_optimizer,
    create_scheduler,
)
from transolver3.inference import CachedInference

# ═══════════════════════════════════════════════════════════════════
# GPU PROFILES
# ═══════════════════════════════════════════════════════════════════

GPU_PROFILES = {
    "a10g": {
        "description": "NVIDIA A10G (g5.xlarge) — 24 GB — paper config",
        "instance_type": "g5.xlarge",
        "model": dict(
            space_dim=22, n_layers=24, n_hidden=256, n_head=8,
            slice_num=64, fun_dim=0, out_dim=4, mlp_ratio=1,
        ),
        "subset_size": 800_000,
        "tile_size": 100_000,
        "mlp_chunk_size": 100_000,
        "cache_chunk_size": 100_000,
        "decode_chunk_size": 50_000,
        "use_fp16": True,
        "mesh_sizes": [
            50_000, 100_000, 200_000, 400_000, 800_000,
            1_000_000, 2_000_000, 4_000_000, 8_000_000,
        ],
        "train_steps": 3,
    },
    "a100_40": {
        "description": "NVIDIA A100 40 GB (p4d.24xlarge) — paper config",
        "instance_type": "p4d.24xlarge",
        "model": dict(
            space_dim=22, n_layers=24, n_hidden=256, n_head=8,
            slice_num=64, fun_dim=0, out_dim=4, mlp_ratio=1,
        ),
        "subset_size": 800_000,
        "tile_size": 100_000,
        "mlp_chunk_size": 100_000,
        "cache_chunk_size": 200_000,
        "decode_chunk_size": 100_000,
        "use_fp16": True,
        "mesh_sizes": [
            50_000, 100_000, 200_000, 400_000, 800_000,
            1_000_000, 2_000_000, 4_000_000, 8_000_000,
        ],
        "train_steps": 3,
    },
    "a100_80": {
        "description": "NVIDIA A100 80 GB (p4de.24xlarge)",
        "instance_type": "p4de.24xlarge",
        "model": dict(
            space_dim=22, n_layers=24, n_hidden=256, n_head=8,
            slice_num=64, fun_dim=0, out_dim=4, mlp_ratio=1,
        ),
        "subset_size": 400_000,
        "tile_size": 100_000,
        "mlp_chunk_size": 100_000,
        "cache_chunk_size": 200_000,
        "decode_chunk_size": 100_000,
        "use_fp16": True,
        "mesh_sizes": [
            10_000, 100_000, 400_000, 1_000_000,
            4_000_000, 8_000_000,
        ],
        "train_steps": 3,
    },
}


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_drivaer(N, space_dim=22, out_dim=4, device="cuda"):
    """Synthetic data matching DrivAer ML surface structure.

    x:      coords(3) + normals(3) + params(16)  = 22 dims
    target: pressure(1) + wall_shear(3)           = 4  dims
    """
    x = torch.randn(1, N, space_dim, device=device)
    target = torch.randn(1, N, out_dim, device=device)
    return x, target


def reset_gpu():
    """Clear GPU cache and reset peak memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def peak_mb():
    """Peak GPU memory allocated (MB) since last reset."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def gpu_info():
    """Return (gpu_name, total_vram_mb)."""
    if not torch.cuda.is_available():
        return "CPU", 0
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    return props.name, total / (1024 ** 2)


def detect_gpu_type():
    """Auto-detect GPU profile from device name."""
    name = torch.cuda.get_device_name(0).upper()
    if "A10" in name:
        return "a10g"
    if "A100" in name:
        total_gb = gpu_info()[1] * (1024 ** 2) / (1024 ** 3)
        return "a100_80" if total_gb > 50 else "a100_40"
    # Fallback: pick profile based on VRAM
    total_gb = gpu_info()[1] * (1024 ** 2) / (1024 ** 3)
    if total_gb >= 60:
        return "a100_80"
    if total_gb >= 30:
        return "a100_40"
    return "a10g"


# ═══════════════════════════════════════════════════════════════════
# PHASE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

def benchmark_training(model, N, profile, device):
    """Phase 1: Amortized training — measure peak GPU memory over a few steps."""
    model.train()
    reset_gpu()

    space_dim = profile["model"]["space_dim"]
    out_dim = profile["model"]["out_dim"]
    subset_size = min(profile["subset_size"], N)

    x, target = generate_synthetic_drivaer(N, space_dim, out_dim, device)

    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, total_steps=100)
    sampler = AmortizedMeshSampler(subset_size)
    scaler = torch.amp.GradScaler("cuda") if profile["use_fp16"] else None

    loss_val = None
    t0 = time.time()
    for _ in range(profile["train_steps"]):
        loss_val = train_step(
            model, x, None, target, optimizer, scheduler,
            sampler=sampler,
            tile_size=profile["tile_size"],
            scaler=scaler,
            grad_clip=1.0,
        )
    elapsed = time.time() - t0
    mem = peak_mb()

    del x, target, optimizer, scheduler, sampler, scaler
    return {"peak_mb": round(mem, 1), "loss": round(loss_val, 6), "time_s": round(elapsed, 2)}


def benchmark_cache_build(model, N, profile, device):
    """Phase 2: Cache physical states (inference phase 1)."""
    model.eval()
    reset_gpu()

    space_dim = profile["model"]["space_dim"]
    out_dim = profile["model"]["out_dim"]
    x, _ = generate_synthetic_drivaer(N, space_dim, out_dim, device)

    engine = CachedInference(
        model,
        cache_chunk_size=profile["cache_chunk_size"],
        decode_chunk_size=profile["decode_chunk_size"],
    )

    t0 = time.time()
    with torch.no_grad():
        cache = engine.build_cache(x)
    elapsed = time.time() - t0
    mem = peak_mb()

    del x
    return {"peak_mb": round(mem, 1), "time_s": round(elapsed, 2)}, cache, engine


def benchmark_decode(engine, N, cache, profile, device):
    """Phase 3: Decode from cache (inference phase 2)."""
    model = engine.model
    model.eval()
    reset_gpu()

    space_dim = profile["model"]["space_dim"]
    out_dim = profile["model"]["out_dim"]
    x_query, _ = generate_synthetic_drivaer(N, space_dim, out_dim, device)

    t0 = time.time()
    with torch.no_grad():
        pred = engine.decode(x_query, cache)
    elapsed = time.time() - t0
    mem = peak_mb()

    assert pred.shape == (1, N, out_dim), f"Expected (1, {N}, {out_dim}), got {pred.shape}"
    del x_query, pred
    return {"peak_mb": round(mem, 1), "time_s": round(elapsed, 2)}


# ═══════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════

def run_oom_safe(fn, label):
    """Run a benchmark function, catching OOM gracefully."""
    try:
        return fn()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"  {label}: OOM")
            torch.cuda.empty_cache()
            gc.collect()
            return {"peak_mb": "OOM", "time_s": None, "error": str(e)[:120]}
        raise


def run_benchmark(gpu_type):
    """Run the full 3-phase benchmark across mesh sizes."""
    profile = GPU_PROFILES[gpu_type]
    device = torch.device("cuda")
    gpu_name, gpu_total_mb = gpu_info()

    print(f"\n{'=' * 72}")
    print(f"  Transolver-3 GPU Memory Benchmark")
    print(f"{'=' * 72}")
    print(f"  GPU:        {gpu_name} ({gpu_total_mb:,.0f} MB)")
    print(f"  Profile:    {gpu_type} — {profile['description']}")
    model_cfg = profile["model"]
    print(f"  Model:      {model_cfg['n_layers']}L / {model_cfg['n_hidden']}C / "
          f"{model_cfg['n_head']}H / {model_cfg['slice_num']}M slices")
    print(f"  FP16:       {profile['use_fp16']}")
    print(f"  Subset:     {profile['subset_size']:,}")
    print(f"  Tile size:  {profile['tile_size']:,}")
    print(f"  Cache/Dec:  {profile['cache_chunk_size']:,} / {profile['decode_chunk_size']:,}")
    print(f"{'=' * 72}\n")

    results = []

    for mesh_idx, N in enumerate(profile["mesh_sizes"]):
        print(f"\n--- Mesh size: {N:>10,} ---")

        # Fresh model for each mesh size (clean state after potential OOM)
        model = Transolver3(
            **model_cfg,
            tile_size=profile["tile_size"],
            mlp_chunk_size=profile["mlp_chunk_size"],
        ).to(device)

        if mesh_idx == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Model parameters: {n_params:,}")

        row = {
            "mesh_size": N,
            "train": None, "cache": None, "decode": None,
            "status": "OK",
        }

        # --- Phase 1: Training ---
        result = run_oom_safe(
            lambda: benchmark_training(model, N, profile, device),
            "Train",
        )
        if result and result.get("peak_mb") != "OOM":
            row["train"] = result
            print(f"  Train:  {result['peak_mb']:>8,.1f} MB | "
                  f"loss={result['loss']:.6f} | {result['time_s']:.2f}s")
        else:
            row["train"] = result
            row["status"] = "OOM_TRAIN"

        # --- Phase 2: Cache Build (runs even if training OOMed) ---
        cache = None
        engine = None
        # Recreate model after OOM to get clean state
        if row["status"] == "OOM_TRAIN":
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = Transolver3(
                **model_cfg,
                tile_size=profile["tile_size"],
                mlp_chunk_size=profile["mlp_chunk_size"],
            ).to(device)

        result = run_oom_safe(
            lambda: benchmark_cache_build(model, N, profile, device),
            "Cache",
        )
        if isinstance(result, tuple):
            # Success: (metrics_dict, cache, engine)
            cache_result, cache, engine = result
            row["cache"] = cache_result
            print(f"  Cache:  {cache_result['peak_mb']:>8,.1f} MB | "
                  f"{cache_result['time_s']:.2f}s")
        else:
            # OOM: run_oom_safe returned a dict with error info
            row["cache"] = result
            if row["status"] == "OK":
                row["status"] = "OOM_CACHE"

        # --- Phase 3: Decode (runs even if training OOMed) ---
        if cache is not None and engine is not None:
            result = run_oom_safe(
                lambda: benchmark_decode(engine, N, cache, profile, device),
                "Decode",
            )
            if result and result.get("peak_mb") != "OOM":
                row["decode"] = result
                print(f"  Decode: {result['peak_mb']:>8,.1f} MB | "
                      f"{result['time_s']:.2f}s")
            else:
                row["decode"] = result
                row["status"] = "OOM_DECODE"

        results.append(row)

        # Cleanup
        del model, cache, engine
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_results_table(results):
    """Print a formatted results table."""
    hdr = (f"{'Mesh Size':>12} | {'Train MB':>10} {'Time':>7} {'Loss':>12} | "
           f"{'Cache MB':>10} {'Time':>7} | {'Decode MB':>10} {'Time':>7} | Status")
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for r in results:
        def _fmt(phase_data, key, fmt_str):
            if phase_data is None:
                return "---"
            val = phase_data.get(key)
            if val is None or val == "OOM":
                return "OOM" if phase_data.get("peak_mb") == "OOM" else "---"
            return fmt_str.format(val)

        t = r["train"]
        c = r["cache"]
        d = r["decode"]

        print(
            f"{r['mesh_size']:>12,} | "
            f"{_fmt(t, 'peak_mb', '{:>10,.1f}'):>10} "
            f"{_fmt(t, 'time_s', '{:>7.2f}'):>7} "
            f"{_fmt(t, 'loss', '{:>12.6f}'):>12} | "
            f"{_fmt(c, 'peak_mb', '{:>10,.1f}'):>10} "
            f"{_fmt(c, 'time_s', '{:>7.2f}'):>7} | "
            f"{_fmt(d, 'peak_mb', '{:>10,.1f}'):>10} "
            f"{_fmt(d, 'time_s', '{:>7.2f}'):>7} | "
            f"{r['status']}"
        )

    print(f"{'=' * len(hdr)}")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Transolver-3 GPU Memory Benchmark")
    parser.add_argument(
        "--gpu_type",
        default=os.environ.get("GPU_TYPE", ""),
        choices=["", *GPU_PROFILES.keys()],
        help="GPU profile (auto-detect if empty)",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_PATH", ""),
        help="Output JSON path (default: benchmark_results_{gpu_type}.json)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    gpu_type = args.gpu_type or detect_gpu_type()
    print(f"Selected GPU profile: {gpu_type}")

    if gpu_type not in GPU_PROFILES:
        print(f"ERROR: Unknown GPU type '{gpu_type}'. Choose from: {list(GPU_PROFILES.keys())}")
        sys.exit(1)

    results = run_benchmark(gpu_type)
    print_results_table(results)

    # Save results
    output_path = args.output or f"benchmark_results_{gpu_type}.json"

    # On Databricks, write to DBFS results dir
    dbfs_results = "/dbfs/transolver3_benchmark/results"
    if os.path.isdir("/dbfs/transolver3_benchmark"):
        os.makedirs(dbfs_results, exist_ok=True)
        output_path = os.path.join(dbfs_results, f"benchmark_{gpu_type}.json")

    output = {
        "gpu_type": gpu_type,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_mb": round(gpu_info()[1] * (1024 ** 2) / (1024 ** 2), 0),
        "profile": {k: v for k, v in GPU_PROFILES[gpu_type].items() if k != "model"},
        "model_config": GPU_PROFILES[gpu_type]["model"],
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
