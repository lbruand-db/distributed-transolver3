"""
Distributed training utilities for Transolver-3.

Supports mesh-sharded training where each GPU/worker owns a disjoint
partition of the mesh. The model is replicated via DDP; gradients are
all-reduced automatically.

Launch with:
  torchrun --nproc_per_node=K script.py [args]
Or via Databricks TorchDistributor.
"""

import os
import torch
import torch.distributed as dist


def setup_distributed(backend="nccl"):
    """Initialize distributed process group.

    Called automatically by torchrun / TorchDistributor. Falls back to
    gloo on CPU.

    Returns:
        (rank, world_size) — or (0, 1) if not distributed.
    """
    if not dist.is_available() or "RANK" not in os.environ:
        return 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size


def cleanup():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """True on rank 0 or when not distributed."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_local_rank():
    """Get local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_device():
    """Get the torch device for the current rank."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")


def unwrap_ddp_model(model):
    """Unwrap a DDP-wrapped model to get the underlying module."""
    return model.module if hasattr(model, "module") else model


def mesh_shard_range(total_points, rank, world_size):
    """Compute the [start, end) index range for this rank's mesh shard.

    Divides total_points as evenly as possible across world_size workers.
    The last worker may get slightly fewer points.

    Args:
        total_points: total number of mesh points (N)
        rank: this worker's rank
        world_size: total number of workers

    Returns:
        (start, end) index range for this rank
    """
    chunk = total_points // world_size
    remainder = total_points % world_size
    # First `remainder` ranks get one extra point
    if rank < remainder:
        start = rank * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (rank - remainder) * chunk
        end = start + chunk
    return start, end
