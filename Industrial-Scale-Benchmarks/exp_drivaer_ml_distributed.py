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
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

from transolver3.model import Transolver3
from transolver3.amortized_training import AmortizedMeshSampler, create_optimizer, create_scheduler
from transolver3.distributed import setup_distributed, cleanup, is_main_process, get_device, log, logall
from transolver3.normalizer import TargetNormalizer
from transolver3.samplers import SyncedSampler
from dataset.drivaer_ml import DrivAerMLDataset
from train_eval import get_field_key, train_epoch, evaluate

# Optional MLflow integration (guarded import)
try:
    import mlflow
    import mlflow.data
    import mlflow.pytorch
    from transolver3.mlflow_utils import log_training_run, log_model_with_signature

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Transolver-3 on DrivAerML")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./checkpoints/drivaer_ml_distributed")
    parser.add_argument("--field", default="surface", choices=["surface", "volume", "both"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--no-scale-lr",
        action="store_true",
        dest="no_scale_lr",
        help="Disable linear LR scaling by world_size. Use the exact --lr value. "
        "The paper does not mention LR scaling; use this flag for paper reproduction.",
    )
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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training. "
        "Paper Appendix A.4: 'Training was conducted in either float16 or bfloat16 precision.'",
    )
    parser.add_argument(
        "--amp-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        dest="amp_dtype",
        help="AMP dtype. bfloat16 (default) has float32 dynamic range, avoiding norm overflow "
        "with large N. float16 is faster on older GPUs but overflows for N >= 65K. "
        "A10G supports both; bfloat16 is recommended.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * world_size * accumulation_steps.",
    )
    parser.add_argument("--cache_chunk_size", type=int, default=100000)
    parser.add_argument("--decode_chunk_size", type=int, default=50000)
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=5,
        dest="num_eval_samples",
        help="Number of test samples evaluated during training (every 10 epochs). "
        "Evaluating all 50 DrivAerML test samples takes ~65 min per eval cycle (8.8M pts each). "
        "Use a small value (default 5) for fast early-stopping signals during training. "
        "The dedicated evaluate task always runs on the full test set. 0 = all samples.",
    )
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
        "--num-gpus", type=int, default=8, dest="num_gpus",
        help="Number of GPUs (only used with --use-distributor)",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/transolver3-experiments"),
        help="MLflow experiment name/path. Defaults to MLFLOW_EXPERIMENT_NAME env var "
        "or '/Shared/transolver3-experiments'.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers (called once at startup)
# ---------------------------------------------------------------------------

def _preload_to_ssd(args: argparse.Namespace) -> str:
    """Copy NPZ data from FUSE volume to local NVMe SSD; return local data dir.

    Must run BEFORE dist.init_process_group() — NCCL's watchdog kills ranks
    after 600 s waiting on a collective, and copying 400+ NPZ files takes
    10-15 minutes. Coordination uses LOCAL_RANK + a sentinel file so no
    NCCL collective is open during the copy.

    Returns:
        Path to local data directory (SSD if available, else args.data_dir).
    """
    local_ssd = "/local_disk0/transolver_data"
    sentinel = os.path.join(local_ssd, ".preload_done")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if os.path.exists("/local_disk0") and not os.path.exists(sentinel):
        if local_rank == 0:
            logall(f"[local_rank 0] Preloading data from {args.data_dir} to {local_ssd} ...")
            os.makedirs(local_ssd, exist_ok=True)
            for split_file in ["train.txt", "test.txt", "val.txt"]:
                src = os.path.join(args.data_dir, split_file)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(local_ssd, split_file))
            copied = 0
            for split_file in ["train.txt", "test.txt"]:
                src_path = os.path.join(args.data_dir, split_file)
                if not os.path.exists(src_path):
                    continue
                with open(src_path) as f:
                    sample_names = [line.strip() for line in f if line.strip()]
                for name in sample_names:
                    src = os.path.join(args.data_dir, name)
                    dst = os.path.join(local_ssd, name)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied += 1
                        if copied % 50 == 0:
                            logall(f"[local_rank 0]   Copied {copied} files...")
            logall(f"[local_rank 0] Preloaded {copied} NPZ files; writing sentinel")
            # Sentinel written last — guarantees all files visible to other ranks.
            # fsync flushes to the kernel page cache so polling ranks see it immediately
            # regardless of filesystem write-behind buffering.
            with open(sentinel, "w") as f:
                f.write("done\n")
                f.flush()
                os.fsync(f.fileno())
        else:
            logall(f"[local_rank {local_rank}] Waiting for preload sentinel...")
            while not os.path.exists(sentinel):
                time.sleep(5)
            logall(f"[local_rank {local_rank}] Sentinel found, proceeding")

    if os.path.exists(sentinel):
        logall(f"Using local SSD data dir: {local_ssd}")
        return local_ssd
    return args.data_dir


def _build_dataloaders(args, local_data_dir, rank, world_size, shard_mesh, device):
    """Create train/test datasets, DataLoaders, and infer input/output dims.

    Returns:
        tuple: (train_dataset, test_dataset, train_loader, test_loader,
                train_sampler, x_key, t_key, space_dim, out_dim)
    """
    local_subset_size = args.subset_size // world_size
    log(f"Total subset: {args.subset_size:,}, per-GPU: {local_subset_size:,}")

    logall("Creating train dataset...")
    train_dataset = DrivAerMLDataset(
        local_data_dir,
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
        local_data_dir,
        split="test",
        field=args.field,
        subset_size=None,
        shard_id=rank if args.shard_eval else None,
        num_shards=world_size if args.shard_eval else None,
    )
    logall(f"Test dataset: {len(test_dataset)} samples")

    # Infer dimensions from first sample (all ranks read their shard independently)
    logall("Loading first sample to infer dimensions...")
    sample = train_dataset[0]
    if dist.is_initialized():
        dist.barrier()  # sync before model allocation starts
    logall("Barrier passed")

    x_key, t_key = get_field_key(args.field)
    space_dim = sample[x_key].shape[-1]
    out_dim = sample[t_key].shape[-1]

    # SyncedSampler: all ranks see the same shuffled order each epoch so that
    # each optimizer step combines spatial-shard gradients from the *same* sample
    # — 400 steps/epoch (not 100) and true batch_size=1 matching the paper.
    train_sampler = SyncedSampler(train_dataset, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    # For periodic training eval, limit to a small seeded subset for speed.
    # Evaluating all 50 DrivAerML samples takes ~65 min per eval cycle.
    if not args.eval_only and args.num_eval_samples > 0:
        n_eval = min(args.num_eval_samples, len(test_dataset))
        _rng = np.random.default_rng(seed=42)
        eval_indices = sorted(_rng.choice(len(test_dataset), size=n_eval, replace=False).tolist())
        eval_dataset = Subset(test_dataset, eval_indices)
        log(f"Training eval: {n_eval}/{len(test_dataset)} test samples (indices {eval_indices}, seed=42)")
    else:
        eval_dataset = test_dataset
        log(f"Training eval: all {len(test_dataset)} test samples")

    # num_workers=2: prefetch next sample from SSD while GPU processes current one.
    # pin_memory + persistent_workers reduce DataLoader overhead across eval calls.
    test_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    return train_dataset, test_dataset, train_loader, test_loader, train_sampler, x_key, t_key, space_dim, out_dim


def _build_and_wrap_model(args, space_dim, out_dim, rank, world_size, device):
    """Instantiate Transolver3 and wrap in DDP.

    Returns:
        DDP-wrapped model
    """
    logall("Creating model...")
    model = Transolver3(
        space_dim=space_dim,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=args.slice_num,
        mlp_ratio=1,
        dropout=args.dropout,
        num_tiles=args.num_tiles,
    ).to(device)
    logall(f"Model created on {device}")

    logall("Wrapping in DDP...")
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    logall("DDP ready")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}")
    return model


def _fit_target_normalizer(train_dataset, t_key, out_dim, device):
    """Fit z-score target normalizer on a subset of training samples.

    Uses streaming Welford to avoid loading all targets into memory, then
    all-reduces mean/std across ranks (each rank sees its spatial shard).

    Returns:
        TargetNormalizer fitted and moved to device
    """
    target_normalizer = TargetNormalizer(out_dim=out_dim).to(device)
    n_train = len(train_dataset)
    # With ~8.8M points per sample, 20 samples give ~176M data points —
    # more than enough for stable mean/std estimates.
    n_norm_samples = min(n_train, 20)
    log(f"Fitting target normalizer on {n_norm_samples}/{n_train} training samples...")

    def _target_iter():
        for i in range(n_norm_samples):
            if (i + 1) % 5 == 0 or (i + 1) == n_norm_samples:
                log(f"  Normalizer fitting: [{i + 1}/{n_norm_samples}]")
            yield train_dataset[i][t_key]

    target_normalizer.fit_incremental(_target_iter())

    if dist.is_initialized():
        # All-reduce to get global stats across all spatial shards.
        # Use sum/count representation so the weighted average is exact.
        n_local = torch.tensor([n_norm_samples], dtype=torch.float64, device=device)
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)

        mean_sum = (target_normalizer.mean.double() * n_norm_samples).to(device)
        dist.all_reduce(mean_sum, op=dist.ReduceOp.SUM)
        global_mean = mean_sum / n_local

        # Global variance = avg(local_var) + avg(local_mean²) - global_mean²
        local_var = (target_normalizer.std.double() - target_normalizer.eps) ** 2
        var_sum = (local_var * n_norm_samples).to(device)
        mean_sq_sum = (target_normalizer.mean.double() ** 2 * n_norm_samples).to(device)
        dist.all_reduce(var_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mean_sq_sum, op=dist.ReduceOp.SUM)

        global_var = var_sum / n_local + mean_sq_sum / n_local - global_mean**2
        global_std = torch.sqrt(global_var.clamp(min=0)) + target_normalizer.eps

        target_normalizer.mean = global_mean.float().reshape(1, 1, -1)
        target_normalizer.std = global_std.float().reshape(1, 1, -1)

    log(f"Target normalizer: mean={target_normalizer.mean.squeeze().tolist()}, "
        f"std={target_normalizer.std.squeeze().tolist()}")
    return target_normalizer


def _load_model_weights(args, model, target_normalizer, device):
    """Load model weights from MLflow run or checkpoint file.

    Optionally overrides the target normalizer with the one saved in MLflow.

    Returns:
        target_normalizer (possibly updated from MLflow artifacts)
    """
    if args.mlflow_run_id_file and os.path.exists(args.mlflow_run_id_file):
        with open(args.mlflow_run_id_file) as f:
            run_id = f.read().strip()
        log(f"Loading model from MLflow run: {run_id}")
        loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/transolver3", map_location=device)
        model.module.load_state_dict(loaded_model.state_dict())
        log(f"Loaded model from MLflow run {run_id}")

        try:
            from transolver3.mlflow_utils import load_normalization_artifacts
            _, mlflow_target_norm = load_normalization_artifacts(run_id)
            if mlflow_target_norm is not None:
                target_normalizer.load_state_dict(mlflow_target_norm.state_dict())
                target_normalizer = target_normalizer.to(device)
                log("Loaded target normalizer from MLflow artifacts")
        except Exception as e:
            log(f"Could not load normalizer from MLflow ({e}), using refitted one")

    elif args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.module.load_state_dict(state_dict)
        log(f"Loaded checkpoint: {args.checkpoint}")

    return target_normalizer


def _run_eval_only(args, model, test_loader, target_normalizer, device):
    """Run evaluation on the full test set and log results."""
    if args.shard_eval:
        eval_results = evaluate(model, test_loader, args, device,
                                sharded=True, target_normalizer=target_normalizer)
    elif is_main_process():
        eval_results = evaluate(model, test_loader, args, device,
                                sharded=False, target_normalizer=target_normalizer)
    else:
        eval_results = None

    if eval_results is not None:
        agg = eval_results["aggregate"]
        log(f"Test relative L2 error: {agg:.4f} ({agg * 100:.2f}%)")
        for name, val in eval_results.items():
            if name != "aggregate":
                log(f"  {name}: {val:.4f} ({val * 100:.2f}%)")


def _setup_training(args, model, train_loader, rank, world_size, device, scaled_lr):
    """Create optimizer, scheduler, scaler, mesh sampler; resume if requested.

    Returns:
        tuple: (optimizer, scheduler, scaler, mesh_sampler, start_epoch, best_error)
    """
    optimizer = create_optimizer(model, lr=scaled_lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_loader) + args.accumulation_steps - 1) // args.accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    scheduler = create_scheduler(optimizer, total_steps)

    # GradScaler only needed for float16; bfloat16 has float32 dynamic range
    _amp_dtype_str = getattr(args, "amp_dtype", "bfloat16")
    scaler = torch.amp.GradScaler() if (args.amp and _amp_dtype_str == "float16") else None
    if args.amp:
        log(f"Mixed precision training enabled (AMP dtype={_amp_dtype_str})")

    # Each rank gets a unique seed for different random spatial subsets
    local_subset_size = args.subset_size // world_size
    mesh_sampler = AmortizedMeshSampler(local_subset_size, seed=args.seed + rank)

    start_epoch = 0
    best_error = float("inf")
    if args.resume and os.path.exists(args.resume):
        log(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_error = ckpt.get("best_error", float("inf"))
        log(f"Resumed from epoch {ckpt['epoch']}, best_error={best_error:.4f}, "
            f"continuing at epoch {start_epoch}")

    return optimizer, scheduler, scaler, mesh_sampler, start_epoch, best_error


def _start_mlflow_run(
    args, model, train_ds, test_ds, x_key, t_key,
    space_dim, out_dim, target_normalizer, shard_mesh, world_size, scaled_lr,
):
    """Start an MLflow run and log parameters, package versions, and dataset info.

    Returns:
        mlflow ActiveRun object, or None if MLflow is unavailable / fails.
    """
    if not (_HAS_MLFLOW and is_main_process()):
        return None

    try:
        mlflow.set_experiment(args.mlflow_experiment)
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
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
                "grad_clip": args.grad_clip,
                "patience": args.patience,
                "save_every": args.save_every,
                "cache_chunk_size": args.cache_chunk_size,
                "decode_chunk_size": args.decode_chunk_size,
                "effective_batch_size": args.batch_size * world_size * args.accumulation_steps,
                "space_dim": space_dim,
                "out_dim": out_dim,
                "dropout": args.dropout,
                "amp": args.amp,
                "no_scale_lr": args.no_scale_lr,
            },
            normalizers={"target": target_normalizer},
        )

        # Package versions for reproducibility
        import importlib.metadata
        for pkg in ["torch", "einops", "timm", "numpy", "mlflow"]:
            try:
                mlflow.log_param(f"pkg_{pkg}", importlib.metadata.version(pkg))
            except Exception:
                pass
        mlflow.log_param("python_version", sys.version.split()[0])

        # Dataset lineage — hash split files for exact reproducibility
        import hashlib
        for split_name in ["train", "test"]:
            split_path = os.path.join(args.data_dir, f"{split_name}.txt")
            if os.path.exists(split_path):
                with open(split_path, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()[:16]
                mlflow.log_param(f"split_{split_name}_hash", digest)

        train_sample = train_ds[0]
        mlflow.log_input(
            mlflow.data.from_numpy(
                features=train_sample[x_key].numpy(),
                targets=train_sample[t_key].numpy(),
                source=args.data_dir,
                name=f"drivaer-{args.field}-train",
            ),
            context="training",
        )
        mlflow.log_param("train_samples", len(train_ds))

        test_sample = test_ds[0]
        mlflow.log_input(
            mlflow.data.from_numpy(
                features=test_sample[x_key].numpy(),
                targets=test_sample[t_key].numpy(),
                source=args.data_dir,
                name=f"drivaer-{args.field}-test",
            ),
            context="evaluation",
        )
        mlflow.log_param("test_samples", len(test_ds))
        mlflow.log_param("data_dir", args.data_dir)

        return mlflow_run

    except Exception as e:
        log(f"MLflow tracking unavailable ({e}), continuing without it")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logall(f"main() entered (pid={os.getpid()})")
    args = parse_args()
    logall("args parsed")

    local_data_dir = _preload_to_ssd(args)

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Do NOT set cudnn.deterministic=True — it costs 10-20% throughput and
    # provides no benefit here because DDP all-reduce is already non-deterministic.
    torch.backends.cudnn.benchmark = False

    rank, world_size = setup_distributed()
    device = get_device()
    logall(f"distributed setup done: device={device}")
    log(f"Distributed: {world_size} workers, device={device}")

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    shard_mesh = not args.no_shard_mesh and world_size > 1
    shard_msg = f"ON (each GPU loads 1/{world_size} of mesh)" if shard_mesh else "OFF (full mesh on each GPU)"
    log(f"Mesh sharding: {shard_msg}")

    (train_ds, test_ds, train_loader, test_loader,
     train_sampler, x_key, t_key, space_dim, out_dim) = _build_dataloaders(
        args, local_data_dir, rank, world_size, shard_mesh, device
    )

    model = _build_and_wrap_model(args, space_dim, out_dim, rank, world_size, device)
    target_normalizer = _fit_target_normalizer(train_ds, t_key, out_dim, device)
    target_normalizer = _load_model_weights(args, model, target_normalizer, device)

    if args.eval_only:
        _run_eval_only(args, model, test_loader, target_normalizer, device)
        cleanup()
        return

    scaled_lr = args.lr if args.no_scale_lr else args.lr * world_size
    lr_msg = f"lr={scaled_lr:.1e}" + ("" if args.no_scale_lr else f" (scaled {world_size}x)")
    log(f"Training: epochs 1-{args.epochs}, {lr_msg}")
    log(f"Tiles: {args.num_tiles}, Cache chunks: {args.cache_chunk_size:,}")

    (optimizer, scheduler, scaler,
     mesh_sampler, start_epoch, best_error) = _setup_training(
        args, model, train_loader, rank, world_size, device, scaled_lr
    )

    # Auth env vars are propagated by _propagate_databricks_auth_env() in the
    # driver before TorchDistributor launches, so child processes can reach MLflow.
    mlflow_run = _start_mlflow_run(
        args, model, train_ds, test_ds, x_key, t_key,
        space_dim, out_dim, target_normalizer, shard_mesh, world_size, scaled_lr,
    )

    # --- Training loop ---
    # try/finally ensures mlflow.end_run() is always called, even on OOM /
    # NaN / barrier crash — otherwise the run stays "RUNNING" in the UI forever.
    try:
        log(f"Starting training loop: {args.epochs - start_epoch} epochs, "
            f"{len(train_loader)} batches/epoch, eval every 10 epochs")
        if start_epoch == 0:
            best_error = float("inf")
        evals_without_improvement = 0

        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            t0 = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, mesh_sampler, args, device,
                scaler=scaler, target_normalizer=target_normalizer,
            )
            t1 = time.time()

            did_eval = (epoch + 1) % 10 == 0 or epoch == args.epochs - 1
            eval_results = None
            test_error = None

            if did_eval:
                if args.shard_eval:
                    eval_results = evaluate(model, test_loader, args, device,
                                            sharded=True, target_normalizer=target_normalizer)
                elif is_main_process():
                    eval_results = evaluate(model, test_loader, args, device,
                                            sharded=False, target_normalizer=target_normalizer)

                if is_main_process() and eval_results is not None:
                    test_error = eval_results["aggregate"]
                    per_q = " | ".join(
                        f"{k}={v:.4f}" for k, v in eval_results.items() if k != "aggregate"
                    )
                    log(
                        f"Epoch {epoch + 1}/{args.epochs} | "
                        f"train_loss={train_loss:.6f} | "
                        f"test_L2={test_error:.4f} ({test_error * 100:.2f}%)"
                        + (f" | {per_q}" if per_q else "")
                        + f" | time={t1 - t0:.1f}s"
                    )
                    if test_error < best_error:
                        best_error = test_error
                        evals_without_improvement = 0
                        torch.save(model.module.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
                        torch.save(target_normalizer.state_dict(), os.path.join(args.save_dir, "target_normalizer.pt"))
                    else:
                        evals_without_improvement += 1

                if dist.is_initialized():
                    dist.barrier()

                # Broadcast early stopping decision from rank 0 so all ranks break together
                if args.patience > 0 and dist.is_initialized():
                    should_stop = torch.tensor(
                        [1 if evals_without_improvement >= args.patience else 0], device=device
                    )
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item():
                        log(f"Early stopping at epoch {epoch + 1}: "
                            f"no improvement for {args.patience} eval cycles")
                        break
                elif args.patience > 0 and evals_without_improvement >= args.patience:
                    log(f"Early stopping at epoch {epoch + 1}: "
                        f"no improvement for {args.patience} eval cycles")
                    break
            else:
                current_lr = optimizer.param_groups[0]["lr"]
                remaining = (args.epochs - epoch - 1) * (t1 - t0)
                eta_h, eta_m = divmod(int(remaining), 3600)
                eta_m = eta_m // 60
                log(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"lr={current_lr:.2e} | "
                    f"time={t1 - t0:.1f}s | "
                    f"ETA ~{eta_h}h{eta_m:02d}m"
                )

            # Release unused CUDA cached memory periodically to prevent allocator
            # fragmentation that causes gradual epoch-time increase over many epochs.
            if (epoch + 1) % 10 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

            # Save full training checkpoint periodically (for resumption)
            if (epoch + 1) % args.save_every == 0 and is_main_process():
                ckpt_path = os.path.join(args.save_dir, "training_checkpoint.pt")
                ckpt_data = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_error": best_error,
                    "target_normalizer": target_normalizer.state_dict(),
                }
                if scaler is not None:
                    ckpt_data["scaler"] = scaler.state_dict()
                torch.save(ckpt_data, ckpt_path)
                log(f"Saved training checkpoint at epoch {epoch + 1} to {ckpt_path}")
            if (epoch + 1) % args.save_every == 0 and dist.is_initialized():
                dist.barrier()

            # Live per-epoch metrics to MLflow
            if mlflow_run and is_main_process():
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("epoch_time_s", t1 - t0, step=epoch)
                if did_eval and eval_results is not None:
                    mlflow.log_metric("test_l2", eval_results["aggregate"], step=epoch)
                    for name, val in eval_results.items():
                        if name != "aggregate":
                            mlflow.log_metric(f"test_l2_{name}", val, step=epoch)

        log(f"\nBest test relative L2 error: {best_error:.4f} ({best_error * 100:.2f}%)")

        # Log best model artifact and write run_id for downstream tasks
        if mlflow_run and is_main_process():
            mlflow.log_metric("best_test_l2", best_error)
            best_path = os.path.join(args.save_dir, "best_model.pt")
            if os.path.exists(best_path):
                from transolver3.distributed import unwrap_ddp_model
                raw_model = unwrap_ddp_model(model)
                raw_model.load_state_dict(torch.load(best_path, map_location=device))
                sample_x = torch.randn(1, 100, space_dim, device=device)
                log_model_with_signature(raw_model, sample_x)

            run_id = mlflow_run.info.run_id
            run_id_path = os.path.join(args.save_dir, "mlflow_run_id.txt")
            with open(run_id_path, "w") as f:
                f.write(run_id)
            log(f"MLflow run_id saved to {run_id_path}: {run_id}")

            mlflow.end_run()
            mlflow_run = None  # prevent finally from double-ending
    finally:
        if mlflow_run and is_main_process():
            mlflow.end_run(status="FAILED")

    cleanup()


if __name__ == "__main__":
    if "--use-distributor" in sys.argv:
        from transolver3.databricks_training import launch_distributed_training

        _idx = sys.argv.index("--num-gpus") if "--num-gpus" in sys.argv else None
        _num_gpus = int(sys.argv[_idx + 1]) if _idx else 8

        # Strip --use-distributor and --num-gpus so child processes don't re-enter
        _clean_argv = []
        _skip_next = False
        for _a in sys.argv[1:]:
            if _skip_next:
                _skip_next = False
                continue
            if _a == "--use-distributor":
                continue
            if _a == "--num-gpus":
                _skip_next = True
                continue
            _clean_argv.append(_a)
        sys.argv = [sys.argv[0]] + _clean_argv

        print(f"[Driver] Launching TorchDistributor with {_num_gpus} GPUs", flush=True)
        print(f"[Driver] Args: {_clean_argv}", flush=True)
        print(f"[Driver] CUDA available: {torch.cuda.is_available()}, "
              f"devices: {torch.cuda.device_count()}", flush=True)
        launch_distributed_training(main, _num_gpus, cli_args=_clean_argv)
        print("[Driver] TorchDistributor finished", flush=True)
    else:
        main()
