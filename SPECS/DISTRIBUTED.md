# Multi-GPU Distribution Plan for Transolver v3

## Context

Transolver v3 currently runs on a single GPU with memory optimizations (tiling, amortized training, chunked inference, mixed precision). The model is ~300M parameters (24 layers, 256 hidden, 8 heads, 64 slices). Training uses random 400K-point subsets from meshes of 8.8M-160M cells. Inference uses a two-phase cache+decode pipeline where the cache is tiny (768KB total) and decode is embarrassingly parallel.

Goal: plan multi-GPU strategies to speed up training and inference, targeting Databricks GPU clusters with available tiers (A10G 24GB, 8x A100-40, 8x A100-80).

---

## Databricks Runtime Considerations

### Target Instances

| DAB Target | Instance | GPUs | Interconnect | Multi-GPU |
|------------|----------|:----:|-------------|:---------:|
| `a10g` | g5.xlarge | 1x A10G 24GB | N/A (single GPU) | No |
| `a10g_multi` | g5.12xlarge | 4x A10G 24GB | NVSwitch | Yes |
| `a100_40` | p4d.24xlarge | 8x A100 40GB | NVLink 600GB/s | Yes |
| `a100_80` | p4de.24xlarge | 8x A100 80GB | NVLink 600GB/s | Yes |

The p4d/p4de instances are single-node 8-GPU machines with NVLink — ideal for
DDP and decode sharding with negligible communication overhead.

### TorchDistributor (Databricks Native)

Databricks provides `pyspark.ml.torch.distributor.TorchDistributor` to launch
multi-GPU PyTorch training on Spark clusters. It handles:
- Process spawning (equivalent to `torchrun`)
- Setting `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`
- NCCL initialization
- Single-node (`local_mode=True`) and multi-node execution

**Usage pattern:**

```python
from pyspark.ml.torch.distributor import TorchDistributor

def train_fn():
    """Runs on each GPU process."""
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # ... training code with DDP ...

distributor = TorchDistributor(
    num_processes=8,         # 8 GPUs on p4d.24xlarge
    local_mode=True,         # single-node (all GPUs on this machine)
    use_gpu=True,
)
distributor.run(train_fn)
```

**Alternative — `torchrun` via Databricks Job task:**

```yaml
# resources/job.yml
tasks:
  - task_key: train_distributed
    spark_python_task:
      python_file: train_distributed.py
    new_cluster:
      node_type_id: p4d.24xlarge
      num_workers: 0           # single-node
      spark_conf:
        spark.task.resource.gpu.amount: "8"
```

With a wrapper script that calls `torchrun --nproc_per_node=8 exp_drivaer_ml.py`.

### Checkpoint Storage

Databricks checkpoints should be saved to:
- **Unity Catalog Volumes**: `/Volumes/<catalog>/<schema>/<volume>/checkpoints/`
- **DBFS**: `/dbfs/tmp/transolver3/checkpoints/` (legacy)

Only rank 0 should save checkpoints. All ranks can read from the shared filesystem.

---

## Four Options (Ranked by Practical Impact)

### Option A: DDP with Geometry-Parallel Subsets (Recommended)
**Complexity: Low (~150 LOC) | Training speedup: ~7x on 8 GPUs | Inference: unchanged**

Each GPU samples a **different random subset** of the mesh via `AmortizedMeshSampler(seed=base_seed+rank)`, runs forward/backward, then DDP all-reduces gradients. Effective batch = K subsets covering more geometry per step.

| Aspect | Detail |
|--------|--------|
| Communication | 600MB gradient all-reduce per step (NVLink: ~1ms) |
| Memory per GPU | Unchanged (full model + optimizer on each GPU) |
| Files changed | `exp_drivaer_ml.py`, `amortized_training.py`, new `distributed.py` |
| Databricks | `TorchDistributor(num_processes=8, local_mode=True)` on p4d/p4de |
| Risk | Very low — standard PyTorch DDP |

### Option B: Inference Decode Sharding (Recommended)
**Complexity: Low-Medium (~200 LOC) | Training: unchanged | Inference speedup: ~6-7x on 8 GPUs**

Cache build stays single-GPU (the cache is only 768KB). Decode phase splits query points across GPUs — each GPU decodes N/K points independently, then results are gathered. Zero communication during decode.

| Aspect | Detail |
|--------|--------|
| Communication | 768KB cache broadcast + 140MB result gather (one-time each) |
| Memory per GPU | Full model replica + N/K decode activations |
| Files changed | `inference.py` (new `DistributedCachedInference`), `exp_drivaer_ml.py` |
| Databricks | Same `TorchDistributor` context as training |
| Risk | Low — decode is embarrassingly parallel |

### Option C: FSDP (Fully Sharded Data Parallel)
**Complexity: Medium (~250 LOC) | Training: ~0.9x speed, but 3.7GB memory freed | Inference: unchanged**

Shards model params + optimizer across GPUs. Each GPU holds 1/K of model at rest; params are all-gathered per-layer during forward/backward. Frees ~3.7GB per GPU on 8-way sharding.

| Aspect | Detail |
|--------|--------|
| Communication | ~1.2GB all-gather+reduce-scatter per step (overlapped) |
| Memory per GPU | ~525MB model+optimizer (vs 4.2GB single-GPU) |
| Files changed | `exp_drivaer_ml.py`, `amortized_training.py`, `model.py`, `inference.py` |
| Wrap policy | `ModuleWrapPolicy({Transolver3Block})` — 24 natural FSDP units |
| Databricks | Checkpoint save/load needs FSDP `FullStateDictConfig` for UC Volumes |
| Risk | Medium — checkpoint complexity, interaction with cached inference |

### Option D: Pipeline Parallel (Layer Sharding)
**Complexity: High (~500+ LOC) | Training: ~2.5x on 4 GPUs | Inference: ~1.2x**

Split 24 blocks across GPUs. Use micro-batching to reduce pipeline bubble.

| Aspect | Detail |
|--------|--------|
| Communication | 50MB activation transfers per micro-batch per stage boundary |
| Pipeline bubble | 27-43% wasted cycles (K=4, 4-8 micro-batches) |
| Databricks | TorchDistributor doesn't natively support pipeline scheduling |
| Risk | High — complex scheduling, breaks cached inference pattern |

---

## Comparison Matrix

| Criterion | A: DDP | B: Decode Shard | C: FSDP | D: Pipeline |
|-----------|--------|----------------|---------|-------------|
| Training speedup (8 GPU) | **~7x** | None | ~0.9x | ~2.5x |
| Inference speedup (8 GPU) | None | **~6-7x** | None | ~1.2x |
| Memory savings per GPU | None | None | **~3.7GB** | ~3.5GB |
| Implementation LOC | ~150 | ~200 | ~250 | ~500+ |
| Complexity | Low | Low-Med | Medium | High |
| Databricks feasibility | **Best** | **Great** | Good | Poor |
| Risk | Very low | Low | Medium | High |

---

## Recommended Implementation: A + B on Databricks

### Phase 1: Option A (DDP Training) — 1-2 days

1. **Create `transolver3/distributed.py`** (~30 LOC)
   - `setup_distributed()`: calls `dist.init_process_group("nccl")`, sets device
   - `cleanup()`, `is_main_process()`, `get_device()`
   - Works both with `TorchDistributor` and standalone `torchrun`

2. **Update `exp_drivaer_ml.py`**
   - Wrap model in `DistributedDataParallel(model, device_ids=[local_rank])`
   - Give each rank a unique `AmortizedMeshSampler(seed=base_seed + rank)`
   - Scale learning rate: `lr = args.lr * world_size`
   - `DistributedSampler` for the DataLoader
   - Only rank 0 saves checkpoints and logs

3. **Update `amortized_training.py`**
   - DDP handles gradient sync automatically via hooks on `backward()`
   - No explicit `all_reduce` needed

4. **Create `Industrial-Scale-Benchmarks/train_distributed.py`** (Databricks entry point)
   ```python
   from pyspark.ml.torch.distributor import TorchDistributor

   def main():
       from exp_drivaer_ml import main as train_main
       train_main()  # DDP-aware via distributed.py

   TorchDistributor(
       num_processes=8, local_mode=True, use_gpu=True
   ).run(main)
   ```

5. **Launch**: `torchrun --nproc_per_node=8 exp_drivaer_ml.py ...` (local) or via DAB job (Databricks)

### Phase 2: Option B (Distributed Decode) — 2-3 days

1. **Add `DistributedCachedInference` to `inference.py`**
   - Rank 0 calls `build_cache()` (768KB result)
   - `dist.broadcast` cache tensors to all ranks
   - Each rank decodes points `[rank*N/K : (rank+1)*N/K]`
   - `dist.all_gather` collects results
   - Falls back to single-GPU `CachedInference` when not distributed

2. **Update `evaluate()` in `exp_drivaer_ml.py`**
   - Use `DistributedCachedInference` when `world_size > 1`

### Phase 3: Option C (FSDP) — only if needed
Only if A10G (24GB) memory is too tight or model grows larger.

### Phase 4: Option D (Pipeline) — probably never
Only if model scales to 1B+ parameters.

---

## DAB Bundle Updates

Update `databricks.yml` to add multi-GPU targets:

```yaml
targets:
  a10g:
    # Single GPU — existing config
    variables:
      gpu_type: "a10g"
      node_type_id: "g5.xlarge"

  a10g_4gpu:
    # 4x A10G for DDP training
    variables:
      gpu_type: "a10g"
      node_type_id: "g5.12xlarge"
      num_gpus: 4

  a100_40:
    # 8x A100-40 — already multi-GPU instance
    variables:
      gpu_type: "a100_40"
      node_type_id: "p4d.24xlarge"
      num_gpus: 8

  a100_80:
    # 8x A100-80 — already multi-GPU instance
    variables:
      gpu_type: "a100_80"
      node_type_id: "p4de.24xlarge"
      num_gpus: 8
```

---

## Files to Modify

| File | Phase 1 (DDP) | Phase 2 (Decode) | Phase 3 (FSDP) |
|------|:---:|:---:|:---:|
| `transolver3/distributed.py` (new) | X | X | X |
| `Industrial-Scale-Benchmarks/exp_drivaer_ml.py` | X | X | X |
| `Industrial-Scale-Benchmarks/train_distributed.py` (new) | X | X | |
| `transolver3/amortized_training.py` | X | | X |
| `transolver3/inference.py` | | X | X |
| `transolver3/model.py` | | | X |
| `databricks.yml` | X | | |

## Verification

- **Phase 1**: Run `TorchDistributor(num_processes=2).run(train_fn)` on a p4d. Train 10 epochs, compare loss curve to single-GPU baseline. Verify gradients match (mean identical, variance lower with more GPUs).
- **Phase 2**: Run inference on test sample with 1 GPU and 8 GPUs. Output tensors should be bitwise identical (decode is deterministic given the same cache). Measure wall-clock speedup.
- **Phase 3**: Compare FSDP consolidated checkpoint to single-GPU checkpoint. Verify identical model weights after `FullStateDictConfig` consolidation.