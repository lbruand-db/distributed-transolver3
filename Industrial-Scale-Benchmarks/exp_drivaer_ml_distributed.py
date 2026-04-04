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

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler,
    relative_l2_loss,
    create_optimizer,
    create_scheduler,
)
from transolver3.inference import CachedInference, DistributedCachedInference
from transolver3.distributed import (
    setup_distributed,
    cleanup,
    is_main_process,
    get_device,
    log,
    logall,
)
from dataset.drivaer_ml import DrivAerMLDataset

# Optional MLflow integration (guarded import)
try:
    import mlflow
    import mlflow.pytorch
    from transolver3.mlflow_utils import log_training_run, log_model_with_signature

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Transolver-3 on DrivAerML")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./checkpoints/drivaer_ml_distributed")
    parser.add_argument("--field", default="surface", choices=["surface", "volume", "both"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--subset_size",
        type=int,
        default=400000,
        help="Total subset size across all GPUs. Each GPU gets subset_size/world_size.",
    )
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--slice_num", type=int, default=64)
    parser.add_argument("--num_tiles", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * world_size * accumulation_steps.",
    )
    parser.add_argument("--cache_chunk_size", type=int, default=100000)
    parser.add_argument("--decode_chunk_size", type=int, default=50000)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--mlflow_run_id_file",
        type=str,
        default=None,
        help="Path to file containing MLflow run_id. If set, loads model from MLflow instead of --checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience: stop if test error doesn't improve for N eval cycles. "
        "0 = disabled (default). Eval happens every 10 epochs, so --patience 5 = 50 epochs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to full training checkpoint (model + optimizer + scheduler + epoch). "
        "Resumes training from the saved epoch. Overrides --checkpoint.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save full training checkpoint every N epochs (default: 10).",
    )
    parser.add_argument(
        "--no-shard-mesh",
        action="store_true",
        dest="no_shard_mesh",
        help="Disable mesh sharding. Each GPU loads the full mesh "
        "(classic DDP). Use for smaller meshes that fit in RAM.",
    )
    parser.add_argument(
        "--shard-eval",
        action="store_true",
        dest="shard_eval",
        help="Use distributed sharded inference for evaluation. "
        "Each GPU processes its mesh shard; cache accumulators "
        "are all-reduced (~514 KB/layer). Required for meshes "
        "that do not fit in single-node memory.",
    )
    parser.add_argument(
        "--use-distributor",
        action="store_true",
        dest="use_distributor",
        help="Launch via TorchDistributor on Databricks instead of torchrun.",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8, dest="num_gpus", help="Number of GPUs (only used with --use-distributor)"
    )
    return parser.parse_args()


def get_field_key(field):
    return f"{field}_x", f"{field}_target"


def train_epoch(model, dataloader, optimizer, scheduler, sampler, args, device):
    model.train()
    total_loss = 0.0
    count = 0
    accum = args.accumulation_steps
    x_key, t_key = get_field_key(args.field)

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        x = batch[x_key].to(device)
        target = batch[t_key].to(device)

        N = x.shape[1]
        indices = sampler.sample(N).to(device)
        x_sub = x[:, indices]
        target_sub = target[:, indices]

        # Scale loss by accumulation steps so gradients average correctly
        pred = model(x_sub, num_tiles=args.num_tiles)
        loss = relative_l2_loss(pred, target_sub) / accum
        loss.backward()
        # DDP handles gradient all-reduce automatically on backward()

        total_loss += loss.item() * accum  # unscale for logging
        count += 1

        if (step + 1) % accum == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

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
    from transolver3.distributed import unwrap_ddp_model

    raw_model = unwrap_ddp_model(model)
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

        error = relative_l2_loss(pred, target)
        all_errors.append(error.cpu().view(1))

    if all_errors:
        mean_error = torch.cat(all_errors).mean().item()
    else:
        mean_error = float("inf")

    if sharded and dist.is_initialized():
        # Average the error across ranks for consistent reporting
        error_tensor = torch.tensor([mean_error], device=device)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
        mean_error = error_tensor.item() / dist.get_world_size()

    return mean_error


def main():
    logall(f"main() entered (pid={os.getpid()})")
    args = parse_args()
    logall("args parsed, calling setup_distributed()")

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Distributed setup ---
    rank, world_size = setup_distributed()
    device = get_device()
    logall(f"distributed setup done: device={device}")
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
    logall("Creating train dataset...")
    train_dataset = DrivAerMLDataset(
        args.data_dir,
        split="train",
        field=args.field,
        subset_size=local_subset_size,
        shard_id=rank if shard_mesh else None,
        num_shards=world_size if shard_mesh else None,
        seed=args.seed + rank,
    )
    logall(f"Train dataset: {len(train_dataset)} samples")
    # Test dataset: sharded if --shard-eval, otherwise full mesh on rank 0
    test_dataset = DrivAerMLDataset(
        args.data_dir,
        split="test",
        field=args.field,
        subset_size=None,
        shard_id=rank if args.shard_eval else None,
        num_shards=world_size if args.shard_eval else None,
    )
    logall(f"Test dataset: {len(test_dataset)} samples")

    # DistributedSampler ensures each rank sees different samples when
    # there are multiple samples in the dataset
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Model ---
    logall("Loading first sample to infer dimensions...")
    sample = train_dataset[0]
    logall("First sample loaded, waiting at barrier...")
    if dist.is_initialized():
        dist.barrier()
    logall("Barrier passed, creating model...")
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
    logall(f"Model created on {device}")

    # Wrap in DDP — gradient all-reduce is automatic on backward()
    logall("Wrapping in DDP...")
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    logall("DDP ready")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}")

    if args.mlflow_run_id_file and os.path.exists(args.mlflow_run_id_file):
        # Load model from MLflow (preferred over --checkpoint)
        with open(args.mlflow_run_id_file) as f:
            run_id = f.read().strip()
        log(f"Loading model from MLflow run: {run_id}")
        loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/transolver3", map_location=device)
        model.module.load_state_dict(loaded_model.state_dict())
        log(f"Loaded model from MLflow run {run_id}")
    elif args.checkpoint:
        # Fallback: load from checkpoint file
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.module.load_state_dict(state_dict)
        log(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval_only:
        if args.shard_eval:
            # All ranks participate in sharded eval
            error = evaluate(model, test_loader, args, device, sharded=True)
            log(f"Test relative L2 error: {error:.4f} ({error * 100:.2f}%)")
        elif is_main_process():
            error = evaluate(model, test_loader, args, device, sharded=False)
            log(f"Test relative L2 error: {error:.4f} ({error * 100:.2f}%)")
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

    # Resume from full training checkpoint (model + optimizer + scheduler + epoch)
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        log(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_error = ckpt.get("best_error", float("inf"))
        log(f"Resumed from epoch {ckpt['epoch']}, best_error={best_error:.4f}, continuing at epoch {start_epoch}")

    log(f"Training: epochs {start_epoch + 1}-{args.epochs}, lr={scaled_lr:.1e} (scaled {world_size}x)")
    log(f"Tiles: {args.num_tiles}, Cache chunks: {args.cache_chunk_size:,}")

    # --- MLflow tracking (rank 0 only) ---
    # Auth env vars are propagated by _propagate_databricks_auth_env() in the
    # driver before TorchDistributor launches, so child processes can reach MLflow.
    mlflow_run = None
    if _HAS_MLFLOW and is_main_process():
        try:
            mlflow.set_experiment("/Shared/transolver3-experiments")
            mlflow_run = mlflow.start_run(run_name=f"drivaer-{args.field}-{args.n_layers}L")
            log_training_run(
                model,
                {
                    "field": args.field,
                    "epochs": args.epochs,
                    "lr": scaled_lr,
                    "weight_decay": args.weight_decay,
                    "subset_size": args.subset_size,
                    "n_layers": args.n_layers,
                    "n_hidden": args.n_hidden,
                    "n_head": args.n_head,
                    "slice_num": args.slice_num,
                    "num_tiles": args.num_tiles,
                    "world_size": world_size,
                    "shard_mesh": shard_mesh,
                    "seed": args.seed,
                },
            )
        except Exception as e:
            log(f"MLflow tracking unavailable ({e}), continuing without it")
            mlflow_run = None

    # --- Training loop ---
    if start_epoch == 0:
        best_error = float("inf")
    evals_without_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)  # shuffle differently each epoch
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, mesh_sampler, args, device)
        t1 = time.time()

        # Evaluate
        did_eval = (epoch + 1) % 10 == 0 or epoch == args.epochs - 1
        if did_eval:
            if args.shard_eval:
                test_error = evaluate(model, test_loader, args, device, sharded=True)
            elif is_main_process():
                test_error = evaluate(model, test_loader, args, device, sharded=False)
            else:
                test_error = None

            if is_main_process():
                log(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"test_L2={test_error:.4f} ({test_error * 100:.2f}%) | "
                    f"time={t1 - t0:.1f}s"
                )
                if test_error < best_error:
                    best_error = test_error
                    evals_without_improvement = 0
                    torch.save(model.module.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
                else:
                    evals_without_improvement += 1

            if dist.is_initialized():
                dist.barrier()

            # Early stopping (patience = number of eval cycles without improvement)
            if args.patience > 0 and evals_without_improvement >= args.patience:
                log(f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} eval cycles")
                break
        else:
            log(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.6f} | time={t1 - t0:.1f}s")

        # Save full training checkpoint periodically (for resumption)
        if is_main_process() and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, "training_checkpoint.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_error": best_error,
                },
                ckpt_path,
            )
            log(f"Saved training checkpoint at epoch {epoch + 1} to {ckpt_path}")

        # Log metrics to MLflow (live, per-epoch)
        if mlflow_run and is_main_process():
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_time_s", t1 - t0, step=epoch)

    log(f"\nBest test relative L2 error: {best_error:.4f} ({best_error * 100:.2f}%)")

    # Log best model to MLflow and save run_id for downstream tasks
    if mlflow_run and is_main_process():
        mlflow.log_metric("best_test_l2", best_error)
        best_path = os.path.join(args.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            from transolver3.distributed import unwrap_ddp_model

            raw_model = unwrap_ddp_model(model)
            raw_model.load_state_dict(torch.load(best_path, map_location=device))
            sample_x = torch.randn(1, 100, space_dim, device=device)
            log_model_with_signature(raw_model, sample_x)

        # Write run_id so evaluate/register tasks can reference this run
        run_id = mlflow_run.info.run_id
        run_id_path = os.path.join(args.save_dir, "mlflow_run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(run_id)
        log(f"MLflow run_id saved to {run_id_path}: {run_id}")

        mlflow.end_run()

    cleanup()


if __name__ == "__main__":
    # Support --use-distributor for TorchDistributor launch on Databricks
    import sys as _sys

    if "--use-distributor" in _sys.argv:
        from transolver3.databricks_training import launch_distributed_training

        # Parse just num_gpus before handing off
        _idx = _sys.argv.index("--num-gpus") if "--num-gpus" in _sys.argv else None
        _num_gpus = int(_sys.argv[_idx + 1]) if _idx else 8

        # Strip --use-distributor and --num-gpus from argv so child processes
        # launched by torchrun call main() directly (not re-enter distributor)
        _clean_argv = []
        _skip_next = False
        for _a in _sys.argv[1:]:  # skip script name
            if _skip_next:
                _skip_next = False
                continue
            if _a == "--use-distributor":
                continue
            if _a == "--num-gpus":
                _skip_next = True
                continue
            _clean_argv.append(_a)
        _sys.argv = [_sys.argv[0]] + _clean_argv

        launch_distributed_training(main, _num_gpus, cli_args=_clean_argv)
    else:
        main()
