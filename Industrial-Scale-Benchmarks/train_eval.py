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
Training and evaluation routines for Transolver-3 on DrivAerML.

Separated from the experiment entry point so they can be imported by other
scripts (benchmarks, ablations) without pulling in the full CLI machinery.
"""

import torch
import torch.distributed as dist

from transolver3.amortized_training import relative_l2_loss
from transolver3.inference import CachedInference, DistributedCachedInference


def get_field_key(field: str):
    """Return (input_key, target_key) for the given field name."""
    return f"{field}_x", f"{field}_target"


def _all_reduce_mean(value: float, device) -> float:
    """All-reduce a scalar mean across ranks (SUM / world_size).

    Using a single helper prevents forgetting the divide — a bug that would
    silently multiply metrics by world_size with no error.
    """
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


# Per-quantity channel splits for Table 4 reproduction.
# surface_target = cat([pressure(1), wall_shear(3)], dim=-1) -> 4 channels
# volume_target  = cat([velocity(3), pressure(1)], dim=-1)   -> 4 channels
_QUANTITY_SPLITS = {
    "surface": [("p_s", 0, 1), ("tau", 1, 4)],
    "volume": [("u", 0, 3), ("p_v", 3, 4)],
}


def train_epoch(
    model, dataloader, optimizer, scheduler, sampler, args, device,
    scaler=None, target_normalizer=None,
):
    """Run one training epoch and return the mean relative L2 loss.

    Args:
        model: DDP-wrapped Transolver3
        dataloader: training DataLoader
        optimizer: AdamW optimizer
        scheduler: cosine LR scheduler
        sampler: AmortizedMeshSampler for random spatial subsets
        args: parsed CLI arguments (uses .num_tiles, .amp_dtype, .grad_clip,
              .accumulation_steps, .field)
        device: torch.device
        scaler: GradScaler for float16 AMP (None for bfloat16 or no AMP)
        target_normalizer: TargetNormalizer (encodes targets before loss)

    Returns:
        float: mean loss over the epoch
    """
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
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    print("[WARN] GradScaler skipped optimizer step (inf/nan grads), scale reduced", flush=True)
                else:
                    scheduler.step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(count, 1)


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

        cache = engine.build_cache(x)
        pred = engine.decode(x, cache)

        if target_normalizer is not None:
            pred = target_normalizer.decode(pred)

        error = relative_l2_loss(pred, target)
        all_errors.append(error.cpu().view(1))

        for name, start, end in quantity_splits:
            q_error = relative_l2_loss(pred[..., start:end], target[..., start:end])
            per_quantity_errors[name].append(q_error.cpu().view(1))

    mean_error = torch.cat(all_errors).mean().item() if all_errors else float("inf")

    results = {}
    for name, errs in per_quantity_errors.items():
        results[name] = torch.cat(errs).mean().item() if errs else float("inf")

    if sharded and dist.is_initialized():
        mean_error = _all_reduce_mean(mean_error, device)
        for name in results:
            results[name] = _all_reduce_mean(results[name], device)

    results["aggregate"] = mean_error
    return results
