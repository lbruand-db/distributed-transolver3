"""
Mesh-sharded distributed training of Transolver-3 on DrivAerML.

Each GPU owns a disjoint partition of the mesh (via mmap range reads).
Each GPU subsamples its local partition for amortized training.
Gradients are all-reduced via DDP.

Launch:
  torchrun --nproc_per_node=8 exp_drivaer_ml_distributed.py --data_dir /path/to/data

Or on Databricks:
  from pyspark.ml.torch.distributor import TorchDistributor
  TorchDistributor(num_processes=8, local_mode=True, use_gpu=True).run(main)
"""

import sys
import os
import argparse
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss,
    create_optimizer, create_scheduler,
)
from transolver3.inference import CachedInference, DistributedCachedInference
from transolver3.distributed import (
    setup_distributed, cleanup, is_main_process, get_device,
)
from dataset.drivaer_ml import DrivAerMLDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Transolver-3 on DrivAerML')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_dir', default='./checkpoints/drivaer_ml_distributed')
    parser.add_argument('--field', default='surface', choices=['surface', 'volume', 'both'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subset_size', type=int, default=400000,
                        help='Total subset size across all GPUs. Each GPU gets subset_size/world_size.')
    parser.add_argument('--n_layers', type=int, default=24)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=64)
    parser.add_argument('--num_tiles', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--cache_chunk_size', type=int, default=100000)
    parser.add_argument('--decode_chunk_size', type=int, default=50000)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shard-mesh', action='store_true', dest='no_shard_mesh',
                        help='Disable mesh sharding. Each GPU loads the full mesh '
                             '(classic DDP). Use for smaller meshes that fit in RAM.')
    parser.add_argument('--shard-eval', action='store_true', dest='shard_eval',
                        help='Use distributed sharded inference for evaluation. '
                             'Each GPU processes its mesh shard; cache accumulators '
                             'are all-reduced (~514 KB/layer). Required for meshes '
                             'that do not fit in single-node memory.')
    return parser.parse_args()


def get_field_key(field):
    return f'{field}_x', f'{field}_target'


def log(msg):
    """Print only on rank 0."""
    if is_main_process():
        print(msg, flush=True)


def train_epoch(model, dataloader, optimizer, scheduler, sampler, args, device):
    model.train()
    total_loss = 0.0
    count = 0
    x_key, t_key = get_field_key(args.field)

    for batch in dataloader:
        x = batch[x_key].to(device)
        target = batch[t_key].to(device)

        optimizer.zero_grad()

        N = x.shape[1]
        indices = sampler.sample(N).to(device)
        x_sub = x[:, indices]
        target_sub = target[:, indices]

        pred = model(x_sub, num_tiles=args.num_tiles)
        loss = relative_l2_loss(pred, target_sub)
        loss.backward()
        # DDP handles gradient all-reduce automatically on backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, dataloader, args, device, sharded=False):
    """Evaluate using cached inference.

    Args:
        model: DDP-wrapped or raw model
        dataloader: test DataLoader (sharded or full depending on mode)
        args: parsed arguments
        device: torch device
        sharded: if True, use DistributedCachedInference where each rank
                 processes its mesh shard and all-reduces the tiny cache
                 accumulators. If False, rank 0 evaluates on full data.
    """
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    all_errors = []
    x_key, t_key = get_field_key(args.field)

    if sharded:
        engine = DistributedCachedInference(
            raw_model,
            cache_chunk_size=args.cache_chunk_size,
            decode_chunk_size=args.decode_chunk_size,
            num_tiles=args.num_tiles,
        )
    else:
        engine = CachedInference(
            raw_model,
            cache_chunk_size=args.cache_chunk_size,
            decode_chunk_size=args.decode_chunk_size,
            num_tiles=args.num_tiles,
        )

    for batch in dataloader:
        x = batch[x_key].to(device)
        target = batch[t_key].to(device)

        if sharded:
            # Each rank has its local shard; build_cache all-reduces accumulators
            cache = engine.build_cache(x)
            pred = engine.decode(x, cache)
        else:
            cache = engine.build_cache(x)
            pred = engine.decode(x, cache)

        diff_norm = torch.norm(pred - target, p=2, dim=(1, 2))
        target_norm = torch.norm(target, p=2, dim=(1, 2))
        error = diff_norm / (target_norm + 1e-8)
        all_errors.append(error.cpu())

    if all_errors:
        mean_error = torch.cat(all_errors).mean().item()
    else:
        mean_error = float('inf')

    if sharded and dist.is_initialized():
        # Average the error across ranks for consistent reporting
        error_tensor = torch.tensor([mean_error], device=device)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
        mean_error = error_tensor.item() / dist.get_world_size()

    return mean_error


def main():
    args = parse_args()

    # --- Distributed setup ---
    rank, world_size = setup_distributed()
    device = get_device()
    log(f"Distributed: {world_size} workers, device={device}")

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    # Mesh sharding: each GPU loads only 1/K of the mesh from disk.
    # With --no-shard-mesh: every GPU loads the full mesh (classic DDP).
    shard_mesh = not args.no_shard_mesh and world_size > 1

    # Per-GPU subset size
    local_subset_size = args.subset_size // world_size
    log(f"Total subset: {args.subset_size:,}, per-GPU: {local_subset_size:,}")
    shard_msg = f"ON (each GPU loads 1/{world_size} of mesh)" if shard_mesh else "OFF (full mesh on each GPU)"
    log(f"Mesh sharding: {shard_msg}")

    # --- Datasets ---
    train_dataset = DrivAerMLDataset(
        args.data_dir, split='train', field=args.field,
        subset_size=local_subset_size,
        shard_id=rank if shard_mesh else None,
        num_shards=world_size if shard_mesh else None,
    )
    # Test dataset: sharded if --shard-eval, otherwise full mesh on rank 0
    test_dataset = DrivAerMLDataset(
        args.data_dir, split='test', field=args.field,
        subset_size=None,
        shard_id=rank if args.shard_eval else None,
        num_shards=world_size if args.shard_eval else None,
    )

    # DistributedSampler ensures each rank sees different samples when
    # there are multiple samples in the dataset
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Model ---
    sample = train_dataset[0]
    x_key, t_key = get_field_key(args.field)
    space_dim = sample[x_key].shape[-1]
    out_dim = sample[t_key].shape[-1]

    model = Transolver3(
        space_dim=space_dim,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=args.slice_num,
        mlp_ratio=1,
        dropout=0.0,
        num_tiles=args.num_tiles,
    ).to(device)

    # Wrap in DDP — gradient all-reduce is automatic on backward()
    model = DDP(model, device_ids=[device] if device.type == 'cuda' else None)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}")

    if args.checkpoint:
        # Load checkpoint on all ranks
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.module.load_state_dict(state_dict)
        log(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval_only:
        if args.shard_eval:
            # All ranks participate in sharded eval
            error = evaluate(model, test_loader, args, device, sharded=True)
            log(f"Test relative L2 error: {error:.4f} ({error*100:.2f}%)")
        elif is_main_process():
            error = evaluate(model, test_loader, args, device, sharded=False)
            log(f"Test relative L2 error: {error:.4f} ({error*100:.2f}%)")
        cleanup()
        return

    # --- Training ---
    # Scale LR by world_size (linear scaling rule)
    scaled_lr = args.lr * world_size
    optimizer = create_optimizer(model, lr=scaled_lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = create_scheduler(optimizer, total_steps)

    # Each rank gets a unique seed for different random subsets
    mesh_sampler = AmortizedMeshSampler(local_subset_size, seed=args.seed + rank)

    log(f"Training: {args.epochs} epochs, lr={scaled_lr:.1e} (scaled {world_size}x)")
    log(f"Tiles: {args.num_tiles}, Cache chunks: {args.cache_chunk_size:,}")

    best_error = float('inf')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # shuffle differently each epoch
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  mesh_sampler, args, device)
        t1 = time.time()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            if args.shard_eval:
                # All ranks participate in sharded eval
                test_error = evaluate(model, test_loader, args, device, sharded=True)
                log(f"Epoch {epoch+1}/{args.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"test_L2={test_error:.4f} ({test_error*100:.2f}%) | "
                    f"time={t1-t0:.1f}s")
                if is_main_process() and test_error < best_error:
                    best_error = test_error
                    torch.save(model.module.state_dict(),
                               os.path.join(args.save_dir, 'best_model.pt'))
            else:
                if is_main_process():
                    test_error = evaluate(model, test_loader, args, device, sharded=False)
                    log(f"Epoch {epoch+1}/{args.epochs} | "
                        f"train_loss={train_loss:.6f} | "
                        f"test_L2={test_error:.4f} ({test_error*100:.2f}%) | "
                        f"time={t1-t0:.1f}s")
                    if test_error < best_error:
                        best_error = test_error
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.save_dir, 'best_model.pt'))
                if dist.is_initialized():
                    dist.barrier()
        else:
            log(f"Epoch {epoch+1}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | time={t1-t0:.1f}s")

    log(f"\nBest test relative L2 error: {best_error:.4f} ({best_error*100:.2f}%)")
    cleanup()


if __name__ == '__main__':
    main()
