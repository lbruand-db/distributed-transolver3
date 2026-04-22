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
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
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
from transolver3.normalizer import TargetNormalizer
from dataset.drivaer_ml import DrivAerMLDataset

# Optional MLflow integration (guarded import)
try:
    import mlflow
    import mlflow.data
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
        help="Number of test samples evaluated during training (every --eval-every epochs). "
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
        "--num-gpus", type=int, default=8, dest="num_gpus", help="Number of GPUs (only used with --use-distributor)"
    )
    return parser.parse_args()


def get_field_key(field):
    return f"{field}_x", f"{field}_target"


def train_epoch(model, dataloader, optimizer, scheduler, sampler, args, device, scaler=None, target_normalizer=None):
    model.train()
    total_loss = 0.0
    count = 0
    accum = args.accumulation_steps
    x_key, t_key = get_field_key(args.field)
    _amp_dtype_str = getattr(args, "amp_dtype", "bfloat16")
    if _amp_dtype_str == "bfloat16" or device.type == "cpu":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        x = batch[x_key].to(device)
        target = batch[t_key].to(device)

        N = x.shape[1]
        indices = sampler.sample(N).to(device)
        x_sub = x[:, indices]
        target_sub = target[:, indices]

        # Normalize targets so model learns in z-score space (paper Appendix A.3)
        if target_normalizer is not None:
            target_sub = target_normalizer.encode(target_sub)

        # Forward in AMP; loss computed outside autocast in float32.
        # float16 norm overflows when N >= 65K (sum(x²) > float16 max 65504).
        # relative_l2_loss() always casts to float32 internally, but we also
        # compute it outside autocast to be explicit.
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler is not None):
            pred = model(x_sub, num_tiles=args.num_tiles)

        # Loss in float32 — safe regardless of amp_dtype
        loss = relative_l2_loss(pred, target_sub) / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # DDP handles gradient all-reduce automatically on backward()

        total_loss += loss.item() * accum  # unscale for logging
        count += 1

        if (step + 1) % accum == 0 or (step + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(count, 1)


# Per-quantity channel splits for Table 4 reproduction.
# surface_target = cat([pressure(1), wall_shear(3)], dim=-1) -> 4 channels
# volume_target  = cat([velocity(3), pressure(1)], dim=-1)   -> 4 channels
_QUANTITY_SPLITS = {
    "surface": [("p_s", 0, 1), ("tau", 1, 4)],
    "volume": [("u", 0, 3), ("p_v", 3, 4)],
}


@torch.no_grad()
def evaluate(model, dataloader, args, device, sharded=False, target_normalizer=None):
    """Evaluate using cached inference.

    Args:
        model: DDP-wrapped or raw model
        dataloader: test DataLoader (sharded or full depending on mode)
        args: parsed arguments
        device: torch device
        sharded: if True, use DistributedCachedInference where each rank
                 processes its mesh shard and all-reduces the tiny cache
                 accumulators. If False, rank 0 evaluates on full data.
        target_normalizer: if provided, decode predictions back to original
                 scale before computing L2 error against raw targets.

    Returns:
        dict with keys: "aggregate" (overall L2), plus per-quantity keys
        (e.g. "p_s", "tau" for surface; "u", "p_v" for volume).
    """
    from transolver3.distributed import unwrap_ddp_model

    raw_model = unwrap_ddp_model(model)
    raw_model.eval()
    all_errors = []
    # Per-quantity error accumulators
    quantity_splits = _QUANTITY_SPLITS.get(args.field, [])
    per_quantity_errors = {name: [] for name, _, _ in quantity_splits}
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

        # Decode predictions back to original scale for fair comparison
        if target_normalizer is not None:
            pred = target_normalizer.decode(pred)

        error = relative_l2_loss(pred, target)
        all_errors.append(error.cpu().view(1))

        # Per-quantity L2 errors
        for name, start, end in quantity_splits:
            q_error = relative_l2_loss(pred[..., start:end], target[..., start:end])
            per_quantity_errors[name].append(q_error.cpu().view(1))

    if all_errors:
        mean_error = torch.cat(all_errors).mean().item()
    else:
        mean_error = float("inf")

    results = {}
    for name, errs in per_quantity_errors.items():
        if errs:
            results[name] = torch.cat(errs).mean().item()
        else:
            results[name] = float("inf")

    if sharded and dist.is_initialized():
        # Average errors across ranks for consistent reporting
        error_tensor = torch.tensor([mean_error], device=device)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
        mean_error = error_tensor.item() / dist.get_world_size()

        for name in results:
            t = torch.tensor([results[name]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            results[name] = t.item() / dist.get_world_size()

    results["aggregate"] = mean_error
    return results


def main():
    import shutil

    logall(f"main() entered (pid={os.getpid()})")
    args = parse_args()
    logall("args parsed")

    # --- Preload data to local SSD BEFORE distributed setup ---
    # Must happen before dist.init_process_group() so no NCCL collective is
    # open during the copy. NCCL's watchdog kills ranks after 600 s of waiting
    # on any collective — preloading 400+ NPZ files takes 10-15 minutes.
    #
    # Coordination uses LOCAL_RANK (set by torchrun) + a sentinel file:
    #   LOCAL_RANK 0  → copies files, writes sentinel when done
    #   LOCAL_RANK 1+ → spin-wait on sentinel (no NCCL involved)
    local_ssd = "/local_disk0/transolver_data"
    sentinel = os.path.join(local_ssd, ".preload_done")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_data_dir = args.data_dir

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
            # Sentinel written last — guarantees all files visible to other ranks
            with open(sentinel, "w") as f:
                f.write("done\n")
        else:
            logall(f"[local_rank {local_rank}] Waiting for preload sentinel...")
            while not os.path.exists(sentinel):
                time.sleep(5)
            logall(f"[local_rank {local_rank}] Sentinel found, proceeding")

    if os.path.exists(sentinel):
        local_data_dir = local_ssd
        logall(f"Using local SSD data dir: {local_data_dir}")

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
    # For periodic training eval, limit to a small subset for speed.
    # Evaluating all 50 DrivAerML samples (8.8M pts each) takes ~65 min per
    # eval cycle — dominated by 50×22 cache-build forward passes (100K chunks).
    # Using 5 samples brings this to ~6 min. 0 or eval_only = full set.
    if not args.eval_only and args.num_eval_samples > 0:
        n_eval = min(args.num_eval_samples, len(test_dataset))
        eval_indices = list(range(n_eval))
        eval_dataset = Subset(test_dataset, eval_indices)
        log(f"Training eval: {n_eval}/{len(test_dataset)} test samples (--num-eval-samples)")
    else:
        eval_dataset = test_dataset
        log(f"Training eval: all {len(test_dataset)} test samples")
    test_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

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
        dropout=args.dropout,
        num_tiles=args.num_tiles,
    ).to(device)
    logall(f"Model created on {device}")

    # Wrap in DDP — gradient all-reduce is automatic on backward()
    logall("Wrapping in DDP...")
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    logall("DDP ready")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}")

    # --- Target normalization (paper Appendix A.3) ---
    # Fit z-score stats by streaming over a subset of training targets.
    # With ~8.8M points per sample, 20 samples give ~176M data points —
    # more than enough for stable mean/std estimates. Each rank computes
    # stats on its own shard; all-reduce merges them afterwards.
    target_normalizer = TargetNormalizer(out_dim=out_dim).to(device)
    n_train = len(train_dataset)
    n_norm_samples = min(n_train, 20)
    log(f"Fitting target normalizer on {n_norm_samples}/{n_train} training samples...")

    def _target_iter():
        for i in range(n_norm_samples):
            if (i + 1) % 5 == 0 or (i + 1) == n_norm_samples:
                log(f"  Normalizer fitting: [{i + 1}/{n_norm_samples}]")
            sample = train_dataset[i]
            yield sample[t_key]

    target_normalizer.fit_incremental(_target_iter())

    if dist.is_initialized():
        # All-reduce to get global stats across all shards
        # Convert to sum/count representation for correct distributed averaging
        n_local = torch.tensor([n_norm_samples], dtype=torch.float64, device=device)
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)

        mean_sum = (target_normalizer.mean.double() * n_norm_samples).to(device)
        dist.all_reduce(mean_sum, op=dist.ReduceOp.SUM)
        global_mean = mean_sum / n_local

        # For std, we need the global variance: Var = E[X^2] - E[X]^2
        # local std includes eps, but we recompute from scratch
        local_var = (target_normalizer.std.double() - target_normalizer.eps) ** 2
        var_sum = (local_var * n_norm_samples).to(device)
        mean_sq_sum = (target_normalizer.mean.double() ** 2 * n_norm_samples).to(device)
        dist.all_reduce(var_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mean_sq_sum, op=dist.ReduceOp.SUM)

        # Global variance = avg(local_var) + avg(local_mean^2) - global_mean^2
        global_var = var_sum / n_local + mean_sq_sum / n_local - global_mean**2
        global_std = torch.sqrt(global_var.clamp(min=0)) + target_normalizer.eps

        target_normalizer.mean = global_mean.float().reshape(1, 1, -1)
        target_normalizer.std = global_std.float().reshape(1, 1, -1)

    log(f"Target normalizer: mean={target_normalizer.mean.squeeze().tolist()}, "
        f"std={target_normalizer.std.squeeze().tolist()}")

    if args.mlflow_run_id_file and os.path.exists(args.mlflow_run_id_file):
        # Load model from MLflow (preferred over --checkpoint)
        with open(args.mlflow_run_id_file) as f:
            run_id = f.read().strip()
        log(f"Loading model from MLflow run: {run_id}")
        loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/transolver3", map_location=device)
        model.module.load_state_dict(loaded_model.state_dict())
        log(f"Loaded model from MLflow run {run_id}")

        # Load target normalizer from MLflow artifacts (overrides the one fitted above)
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
        # Fallback: load from checkpoint file
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.module.load_state_dict(state_dict)
        log(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval_only:
        if args.shard_eval:
            eval_results = evaluate(
                model, test_loader, args, device,
                sharded=True, target_normalizer=target_normalizer,
            )
        elif is_main_process():
            eval_results = evaluate(
                model, test_loader, args, device,
                sharded=False, target_normalizer=target_normalizer,
            )
        else:
            eval_results = None
        if eval_results is not None:
            agg = eval_results["aggregate"]
            log(f"Test relative L2 error: {agg:.4f} ({agg * 100:.2f}%)")
            for name, val in eval_results.items():
                if name != "aggregate":
                    log(f"  {name}: {val:.4f} ({val * 100:.2f}%)")
        cleanup()
        return

    # --- Training ---
    # Scale LR by world_size (linear scaling rule) unless --no-scale-lr
    scaled_lr = args.lr if args.no_scale_lr else args.lr * world_size
    optimizer = create_optimizer(model, lr=scaled_lr, weight_decay=args.weight_decay)
    # Scheduler total_steps = number of optimizer steps (not backward passes).
    # With gradient accumulation, optimizer steps happen every accum batches.
    steps_per_epoch = (len(train_loader) + args.accumulation_steps - 1) // args.accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    scheduler = create_scheduler(optimizer, total_steps)

    # Mixed precision (paper Appendix A.4)
    # GradScaler only needed for float16; bfloat16 has float32 dynamic range
    # so it doesn't need loss scaling.
    _amp_dtype_str = getattr(args, "amp_dtype", "bfloat16")
    scaler = torch.amp.GradScaler() if (args.amp and _amp_dtype_str == "float16") else None
    if args.amp:
        log(f"Mixed precision training enabled (AMP dtype={_amp_dtype_str})")

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
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_error = ckpt.get("best_error", float("inf"))
        log(f"Resumed from epoch {ckpt['epoch']}, best_error={best_error:.4f}, continuing at epoch {start_epoch}")

    lr_msg = f"lr={scaled_lr:.1e}" + ("" if args.no_scale_lr else f" (scaled {world_size}x)")
    log(f"Training: epochs {start_epoch + 1}-{args.epochs}, {lr_msg}")
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
            # Log installed package versions for reproducibility (MLOPS-5)
            import importlib.metadata
            for pkg in ["torch", "einops", "timm", "numpy", "mlflow"]:
                try:
                    ver = importlib.metadata.version(pkg)
                    mlflow.log_param(f"pkg_{pkg}", ver)
                except Exception:
                    pass
            mlflow.log_param("python_version", sys.version.split()[0])

            # Log dataset lineage via mlflow.data (MLOPS-2)
            import hashlib

            # Hash split files for exact data reproducibility
            for split_name in ["train", "test"]:
                split_path = os.path.join(args.data_dir, f"{split_name}.txt")
                if os.path.exists(split_path):
                    with open(split_path, "rb") as f:
                        digest = hashlib.sha256(f.read()).hexdigest()[:16]
                    mlflow.log_param(f"split_{split_name}_hash", digest)

            # Log training dataset with schema and source via mlflow.data
            train_sample = train_dataset[0]
            train_ds = mlflow.data.from_numpy(
                features=train_sample[x_key].numpy(),
                targets=train_sample[t_key].numpy(),
                source=args.data_dir,
                name=f"drivaer-{args.field}-train",
            )
            mlflow.log_input(train_ds, context="training")
            mlflow.log_param("train_samples", len(train_dataset))

            # Log test dataset
            test_sample = test_dataset[0]
            test_ds = mlflow.data.from_numpy(
                features=test_sample[x_key].numpy(),
                targets=test_sample[t_key].numpy(),
                source=args.data_dir,
                name=f"drivaer-{args.field}-test",
            )
            mlflow.log_input(test_ds, context="evaluation")
            mlflow.log_param("test_samples", len(test_dataset))
            mlflow.log_param("data_dir", args.data_dir)
        except Exception as e:
            log(f"MLflow tracking unavailable ({e}), continuing without it")
            mlflow_run = None

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
            train_sampler.set_epoch(epoch)  # shuffle differently each epoch
            t0 = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, mesh_sampler, args, device,
                scaler=scaler, target_normalizer=target_normalizer,
            )
            t1 = time.time()

            # Evaluate
            did_eval = (epoch + 1) % 10 == 0 or epoch == args.epochs - 1
            eval_results = None
            test_error = None
            if did_eval:
                if args.shard_eval:
                    eval_results = evaluate(
                        model, test_loader, args, device,
                        sharded=True, target_normalizer=target_normalizer,
                    )
                elif is_main_process():
                    eval_results = evaluate(
                        model, test_loader, args, device,
                        sharded=False, target_normalizer=target_normalizer,
                    )

                if is_main_process() and eval_results is not None:
                    test_error = eval_results["aggregate"]
                    per_q = " | ".join(
                        f"{k}={v:.4f}" for k, v in eval_results.items()
                        if k != "aggregate"
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
                    should_stop = torch.tensor([1 if evals_without_improvement >= args.patience else 0], device=device)
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item():
                        log(f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} eval cycles")
                        break
                elif args.patience > 0 and evals_without_improvement >= args.patience:
                    log(f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} eval cycles")
                    break
            else:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_time = (t1 - t0)  # this epoch's time
                remaining = (args.epochs - epoch - 1) * avg_time
                eta_h, eta_m = divmod(int(remaining), 3600)
                eta_m = eta_m // 60
                log(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"lr={current_lr:.2e} | "
                    f"time={t1 - t0:.1f}s | "
                    f"ETA ~{eta_h}h{eta_m:02d}m"
                )

            # Save full training checkpoint periodically (for resumption)
            if (epoch + 1) % args.save_every == 0:
                if is_main_process():
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
                if dist.is_initialized():
                    dist.barrier()

            # Log metrics to MLflow (live, per-epoch)
            if mlflow_run and is_main_process():
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("epoch_time_s", t1 - t0, step=epoch)
                if did_eval and eval_results is not None:
                    mlflow.log_metric("test_l2", eval_results["aggregate"], step=epoch)
                    for name, val in eval_results.items():
                        if name != "aggregate":
                            mlflow.log_metric(f"test_l2_{name}", val, step=epoch)

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
            mlflow_run = None  # prevent finally from double-ending
    finally:
        # Ensure MLflow run is closed even on OOM / NaN / barrier crash
        if mlflow_run and is_main_process():
            mlflow.end_run(status="FAILED")

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

        print(f"[Driver] Launching TorchDistributor with {_num_gpus} GPUs", flush=True)
        print(f"[Driver] Args: {_clean_argv}", flush=True)
        print(f"[Driver] CUDA available: {torch.cuda.is_available()}, "
              f"devices: {torch.cuda.device_count()}", flush=True)
        launch_distributed_training(main, _num_gpus, cli_args=_clean_argv)
        print("[Driver] TorchDistributor finished", flush=True)
    else:
        main()
