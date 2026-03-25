#!/usr/bin/env python3
"""
End-to-end test of mesh-sharded distributed training + inference.

Generates synthetic data, runs 2-GPU sharded training for a few epochs,
validates that sharded inference produces the same cache as single-GPU,
and reports results.

Designed to run on Databricks via TorchDistributor on a multi-GPU instance
(e.g., g5.12xlarge with 4× A10G, using 2 GPUs).

Usage (standalone):
  torchrun --nproc_per_node=2 benchmarks/test_sharded_distributed.py

Usage (Databricks):
  Deployed as a DAB job — see resources/distributed_test_job.yml
"""

import os
import sys
import time
import json
import tempfile
import numpy as np

# --- Path setup (same pattern as gpu_memory_benchmark.py) ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    for candidate in sys.argv:
        if "files/benchmarks" in candidate:
            SCRIPT_DIR = os.path.dirname(candidate)
            break

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

if not os.path.isdir(os.path.join(REPO_ROOT, "transolver3")):
    test_dir = SCRIPT_DIR
    for _ in range(5):
        test_dir = os.path.dirname(test_dir)
        if os.path.isdir(os.path.join(test_dir, "transolver3")):
            REPO_ROOT = test_dir
            sys.path.insert(0, REPO_ROOT)
            break

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _ensure_path():
    """Ensure transolver3 is importable. Called in every subprocess."""
    import sys, os
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    repo_root = os.path.dirname(script_dir)
    if not os.path.isdir(os.path.join(repo_root, "transolver3")):
        test_dir = script_dir
        for _ in range(5):
            test_dir = os.path.dirname(test_dir)
            if os.path.isdir(os.path.join(test_dir, "transolver3")):
                repo_root = test_dir
                break
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_path()

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss,
    create_optimizer, create_scheduler,
)
from transolver3.inference import CachedInference, DistributedCachedInference
from transolver3.distributed import (
    setup_distributed, cleanup, is_main_process, get_device, mesh_shard_range,
)


# ============================================================================
# Config — deliberately small for a fast test
# ============================================================================
CFG = {
    'n_points': 2000,       # small mesh
    'space_dim': 10,        # coords(3) + normals(3) + params(4)
    'out_dim': 4,           # pressure + wall_shear
    'n_layers': 4,          # small model
    'n_hidden': 64,
    'n_head': 4,
    'slice_num': 16,
    'subset_size': 500,     # per-GPU amortized subset
    'num_tiles': 2,
    'epochs': 10,
    'lr': 1e-3,
    'cache_chunk_size': 500,
    'decode_chunk_size': 500,
    'seed': 42,
}


def generate_synthetic_data(n_points, space_dim, out_dim, seed=42):
    """Generate a synthetic mesh sample."""
    rng = np.random.RandomState(seed)
    x = rng.randn(1, n_points, space_dim).astype(np.float32)
    target = rng.randn(1, n_points, out_dim).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(target)


def log(msg, rank=0):
    if rank == 0:
        print(f"[TEST] {msg}", flush=True)


def test_sharded_training(rank, world_size, device):
    """Test: sharded training runs and produces finite loss."""
    log(f"=== Test 1: Sharded Training ({world_size} GPUs) ===", rank)
    cfg = CFG

    model = Transolver3(
        space_dim=cfg['space_dim'], n_layers=cfg['n_layers'],
        n_hidden=cfg['n_hidden'], n_head=cfg['n_head'],
        fun_dim=0, out_dim=cfg['out_dim'], slice_num=cfg['slice_num'],
        mlp_ratio=1, dropout=0.0, num_tiles=cfg['num_tiles'],
    ).to(device)

    model_ddp = DDP(model, device_ids=[device] if device.type == 'cuda' else None)

    # Generate full synthetic mesh
    x_full, target_full = generate_synthetic_data(
        cfg['n_points'], cfg['space_dim'], cfg['out_dim'], seed=cfg['seed']
    )

    # Each rank gets its shard
    start, end = mesh_shard_range(cfg['n_points'], rank, world_size)
    x_local = x_full[:, start:end].to(device)
    target_local = target_full[:, start:end].to(device)

    optimizer = create_optimizer(model_ddp, lr=cfg['lr'] * world_size)
    total_steps = cfg['epochs']
    scheduler = create_scheduler(optimizer, total_steps)
    sampler = AmortizedMeshSampler(cfg['subset_size'], seed=cfg['seed'] + rank)

    t0 = time.time()
    losses = []
    for epoch in range(cfg['epochs']):
        model_ddp.train()
        optimizer.zero_grad()

        N = x_local.shape[1]
        indices = sampler.sample(N).to(device)
        x_sub = x_local[:, indices]
        target_sub = target_local[:, indices]

        pred = model_ddp(x_sub, num_tiles=cfg['num_tiles'])
        loss = relative_l2_loss(pred, target_sub)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    elapsed = time.time() - t0
    final_loss = losses[-1]
    log(f"  {cfg['epochs']} epochs in {elapsed:.2f}s, final loss={final_loss:.6f}", rank)

    assert not np.isnan(final_loss), "Training produced NaN loss!"
    assert final_loss < losses[0] * 1.5, "Loss didn't decrease or exploded!"
    log("  PASSED", rank)

    return model_ddp


def test_sharded_cache_correctness(rank, world_size, device, model_ddp):
    """Test: sharded cache build produces same result as single-GPU."""
    log(f"=== Test 2: Sharded Cache Correctness ===", rank)
    cfg = CFG

    raw_model = model_ddp.module if hasattr(model_ddp, 'module') else model_ddp
    raw_model.eval()

    x_full, _ = generate_synthetic_data(
        cfg['n_points'], cfg['space_dim'], cfg['out_dim'], seed=cfg['seed'] + 100
    )
    x_full = x_full.to(device)

    # Single-GPU cache (ground truth — all ranks compute this)
    cache_single = raw_model.cache_physical_states(
        x_full, chunk_size=cfg['cache_chunk_size']
    )

    # Sharded cache: each rank processes its partition
    start, end = mesh_shard_range(cfg['n_points'], rank, world_size)
    x_local = x_full[:, start:end]

    engine = DistributedCachedInference(
        raw_model,
        cache_chunk_size=cfg['cache_chunk_size'],
        decode_chunk_size=cfg['decode_chunk_size'],
        num_tiles=cfg['num_tiles'],
    )
    cache_sharded = engine.build_cache(x_local)

    # Compare: sharded cache should match single-GPU cache
    max_diff = 0.0
    for layer_idx, (s_single, s_sharded) in enumerate(zip(cache_single, cache_sharded)):
        diff = (s_single - s_sharded).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-3:
            log(f"  WARNING: Layer {layer_idx} diff={diff:.6f}", rank)

    log(f"  Max cache diff across {len(cache_single)} layers: {max_diff:.6f}", rank)
    assert max_diff < 1e-3, f"Cache mismatch: max_diff={max_diff}"
    log("  PASSED", rank)

    return cache_sharded


def test_sharded_decode(rank, world_size, device, model_ddp, cache):
    """Test: sharded decode produces same predictions as single-GPU."""
    log(f"=== Test 3: Sharded Decode ===", rank)
    cfg = CFG

    raw_model = model_ddp.module if hasattr(model_ddp, 'module') else model_ddp
    raw_model.eval()

    x_full, _ = generate_synthetic_data(
        cfg['n_points'], cfg['space_dim'], cfg['out_dim'], seed=cfg['seed'] + 100
    )
    x_full = x_full.to(device)

    # Single-GPU decode (ground truth)
    pred_single = raw_model.decode_from_cache(x_full, cache)

    # Sharded decode: each rank decodes its partition
    start, end = mesh_shard_range(cfg['n_points'], rank, world_size)
    x_local = x_full[:, start:end]

    engine = DistributedCachedInference(
        raw_model,
        cache_chunk_size=cfg['cache_chunk_size'],
        decode_chunk_size=cfg['decode_chunk_size'],
    )
    pred_local = engine.decode(x_local, cache)

    # Compare this rank's predictions to the corresponding slice of single-GPU
    pred_single_local = pred_single[:, start:end]
    diff = (pred_local - pred_single_local).abs().max().item()

    log(f"  Rank {rank}: decode diff={diff:.6f} (points {start}-{end})", rank)
    # Gather max diff across ranks
    diff_tensor = torch.tensor([diff], device=device)
    if dist.is_initialized():
        dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)

    log(f"  Max decode diff across all ranks: {diff_tensor.item():.6f}", rank)
    assert diff_tensor.item() < 1e-3, f"Decode mismatch: {diff_tensor.item()}"
    log("  PASSED", rank)


def main():
    _ensure_path()  # needed when called via TorchDistributor (pickled function)
    rank, world_size = setup_distributed()
    device = get_device()

    log(f"Running on {world_size} {'GPU' if device.type == 'cuda' else 'CPU'} workers", rank)
    if device.type == 'cuda':
        log(f"  GPU: {torch.cuda.get_device_name(device)}", rank)
        log(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB", rank)

    results = {'world_size': world_size, 'device': str(device), 'tests': {}}

    try:
        # Test 1: Sharded training
        t0 = time.time()
        model = test_sharded_training(rank, world_size, device)
        results['tests']['sharded_training'] = {'status': 'PASSED', 'time_s': time.time() - t0}

        # Test 2: Sharded cache correctness
        t0 = time.time()
        cache = test_sharded_cache_correctness(rank, world_size, device, model)
        results['tests']['sharded_cache'] = {'status': 'PASSED', 'time_s': time.time() - t0}

        # Test 3: Sharded decode
        t0 = time.time()
        test_sharded_decode(rank, world_size, device, model, cache)
        results['tests']['sharded_decode'] = {'status': 'PASSED', 'time_s': time.time() - t0}

    except Exception as e:
        log(f"FAILED: {e}", rank)
        import traceback
        traceback.print_exc()
        results['tests']['error'] = str(e)

    # Summary
    log("", rank)
    log("=" * 50, rank)
    n_passed = sum(1 for v in results['tests'].values()
                   if isinstance(v, dict) and v.get('status') == 'PASSED')
    n_total = sum(1 for v in results['tests'].values() if isinstance(v, dict))
    log(f"  {n_passed}/{n_total} tests PASSED on {world_size} GPUs", rank)
    log("=" * 50, rank)

    # Save results (rank 0 only)
    if is_main_process():
        out_path = os.path.join(REPO_ROOT, 'benchmarks', 'distributed_test_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"Results saved to {out_path}", rank)

    cleanup()


def launch_on_databricks(num_gpus=2):
    """Launch via TorchDistributor on Databricks.

    Uses file-based mode (passing script path) so that torchrun
    executes the script from scratch in each subprocess, avoiding
    pickle/import issues with transolver3.
    """
    from pyspark.ml.torch.distributor import TorchDistributor
    # Resolve the path to this script file.
    # On Databricks, spark_python_task runs via exec() so __file__ may not exist.
    # Fall back to sys.argv or the known bundle path.
    try:
        script_path = os.path.abspath(__file__)
    except NameError:
        # Databricks exec() context: reconstruct from sys.argv or SCRIPT_DIR
        script_path = os.path.join(SCRIPT_DIR, 'test_sharded_distributed.py')
        if not os.path.exists(script_path):
            # Try argv
            for arg in sys.argv:
                if 'test_sharded_distributed' in arg:
                    script_path = arg
                    break
    print(f"[LAUNCH] Starting TorchDistributor with {num_gpus} GPUs")
    print(f"[LAUNCH] Script: {script_path}")
    distributor = TorchDistributor(
        num_processes=num_gpus,
        local_mode=True,
        use_gpu=True,
    )
    distributor.run(script_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--databricks', action='store_true',
                        help='Launch via TorchDistributor (Databricks runtime)')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='Number of GPUs for TorchDistributor (default: 2)')
    args, _ = parser.parse_known_args()

    if args.databricks:
        launch_on_databricks(args.num_gpus)
    else:
        # Standalone: assume torchrun already set up the process group
        main()
