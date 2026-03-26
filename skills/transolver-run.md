# transolver-run

Run Transolver-3 simulations on Databricks — from notebook experiments to full distributed training via DABs.

---

## When to use

Use this skill when the user wants to:
- Train a Transolver-3 model on Databricks (notebook or DAB workflow)
- Run the 3-phase pipeline (train, cache, decode) on a GPU cluster
- Choose model configuration presets for their mesh size
- Run a quick sanity check on a GPU cluster
- Launch distributed multi-GPU training via TorchDistributor
- Understand what parameters to set and why

---

## Configuration presets

The model has many parameters. Use these presets as starting points:

### Small (development / testing on single GPU)
```python
config_small = dict(
    space_dim=3,
    n_layers=4,
    n_hidden=64,
    n_head=4,
    slice_num=16,
    out_dim=4,
    tile_size=10_000,
)
# Good for: meshes up to ~50K points, quick iteration on A10G
```

### Medium (single GPU, real experiments)
```python
config_medium = dict(
    space_dim=3,
    n_layers=12,
    n_hidden=128,
    n_head=8,
    slice_num=32,
    out_dim=4,
    tile_size=50_000,
)
# Good for: meshes up to ~1M points, single A10G or A100 GPU
```

### Large (paper configuration, multi-GPU)
```python
config_large = dict(
    space_dim=3,
    n_layers=24,
    n_hidden=256,
    n_head=8,
    slice_num=64,
    out_dim=4,
    tile_size=100_000,
)
# Good for: industrial-scale meshes (100M+ points), multi-GPU A100 cluster
# This is Table 5 from the paper
```

**Key parameter guidance:**
- `slice_num` (M): Number of physics slices. 64 is the paper default. Smaller = faster but less expressive.
- `tile_size`: Points processed per attention tile. 100K per paper. Reduce if OOM.
- `n_layers`: 24 for full accuracy, 4-12 for experiments.
- `n_hidden`: 256 for production, 64-128 for experiments.

---

## Notebook setup (first cell)

```python
%pip install /Workspace/Repos/<user>/Transolver -q
dbutils.library.restartPython()
```

---

## Training in a notebook (single GPU)

Use this on a cluster with a single GPU (e.g., `g5.xlarge` with A10G).

### Minimal training loop

```python
import torch
import numpy as np
import mlflow
from transolver3 import Transolver3
from transolver3.amortized_training import (
    train_step, create_optimizer, create_scheduler, AmortizedMeshSampler
)

# 1. Create model (use a preset from above)
device = torch.device("cuda")
model = Transolver3(**config_small).to(device)

# 2. Load data from UC Volumes
VOLUME_PATH = "/Volumes/ml/transolver3/data"
data = np.load(f"{VOLUME_PATH}/drivaer_001.npz", mmap_mode='r')
x = torch.tensor(data["coordinates"], dtype=torch.float32).unsqueeze(0).to(device)
y = torch.tensor(data["targets"], dtype=torch.float32).unsqueeze(0).to(device)

# 3. Setup training
optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.05)
total_steps = 10_000
scheduler = create_scheduler(optimizer, total_steps)
sampler = AmortizedMeshSampler(subset_size=100_000, seed=42)

# 4. Train with MLflow tracking
mlflow.set_experiment("/Shared/transolver3-experiments")

with mlflow.start_run(run_name="notebook-training"):
    mlflow.log_params({
        "n_layers": config_small["n_layers"],
        "n_hidden": config_small["n_hidden"],
        "slice_num": config_small["slice_num"],
        "total_steps": total_steps,
    })

    for step in range(total_steps):
        loss = train_step(
            model, x, fx=None, target=y,
            optimizer=optimizer, scheduler=scheduler,
            sampler=sampler, tile_size=config_small["tile_size"],
        )
        if step % 100 == 0:
            mlflow.log_metric("train_loss", loss, step=step)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=step)
            print(f"Step {step:5d} | Loss: {loss:.6f}")
```

### With mixed precision (recommended)

```python
scaler = torch.amp.GradScaler()

for step in range(total_steps):
    loss = train_step(
        model, x, fx=None, target=y,
        optimizer=optimizer, scheduler=scheduler,
        sampler=sampler, tile_size=100_000,
        scaler=scaler,  # enables float16 autocast
    )
```

### With target normalization

```python
from transolver3.normalizer import TargetNormalizer

target_norm = TargetNormalizer(out_dim=y.shape[-1])
target_norm.fit(y)
target_norm = target_norm.to(device)

for step in range(total_steps):
    loss = train_step(
        model, x, fx=None, target=y,
        optimizer=optimizer, scheduler=scheduler,
        sampler=sampler, tile_size=100_000,
        normalizer=target_norm,  # model learns in normalized space
    )
```

---

## Three-phase pipeline (notebook)

After training, use the cache-and-decode pipeline for inference on full-size meshes:

### Phase 2: Cache build + Phase 3: Decode

```python
from transolver3 import CachedInference

model.eval()
engine = CachedInference(
    model,
    cache_chunk_size=100_000,   # points per cache-building chunk
    decode_chunk_size=50_000,   # points per decode chunk
    num_tiles=0,                # tiling within chunks (0=auto)
)

# Option A: End-to-end prediction
output = engine.predict(x_full)  # (B, N, out_dim)

# Option B: Separate cache + decode (reuse cache for different queries)
cache = engine.build_cache(x_full)      # Phase 2: ~768 KB total
output = engine.decode(x_query, cache)  # Phase 3: decode at any points
```

---

## Distributed training (multi-GPU notebook)

Use this on a multi-GPU cluster (e.g., `p4d.24xlarge` with 8x A100).

### Via TorchDistributor

```python
from transolver3.databricks_training import launch_distributed_training

# TorchDistributor handles multi-GPU setup automatically
launch_distributed_training(
    data_dir="/Volumes/ml/transolver3/data",
    n_layers=24, n_hidden=256, slice_num=64,
    total_steps=50_000, num_gpus=8,
)
```

### Distributed cached inference (notebook)

```python
from transolver3.inference import DistributedCachedInference
from transolver3.distributed import setup_distributed, mesh_shard_range

# Each GPU gets a disjoint mesh partition
rank, world_size = setup_distributed()
start, end = mesh_shard_range(N_total, rank, world_size)
x_local = x_full[:, start:end]

engine = DistributedCachedInference(model, cache_chunk_size=100_000)
cache = engine.build_cache(x_local)        # all-reduce ~514 KB/layer
local_pred = engine.decode(x_local, cache) # independent per GPU
full_pred = engine.predict(x_local, gather=True)  # gathered on all ranks
```

---

## DAB deployment (production workflows)

### Available targets

| Target | Instance | GPU | VRAM | Use case |
|--------|----------|-----|------|----------|
| `a10g` (default) | g5.xlarge | 1x A10G | 24 GB | Benchmarks, small experiments |
| `a100_40` | p4d.24xlarge | 8x A100 | 320 GB | Multi-GPU training |
| `a100_80` | p4de.24xlarge | 8x A100-80 | 640 GB | Full-scale DrivAerML |

### Deploy and run

```bash
databricks bundle deploy -t a10g

# Available jobs:
databricks bundle run gpu_memory_benchmark     # Single-GPU memory sweep
databricks bundle run distributed_sharded_test # 2-GPU sharding validation
databricks bundle run training_workflow        # Full 5-task pipeline
```

### Full training workflow (5 tasks)

The DAB `training_workflow` runs this automated pipeline:
1. **Preprocess** — Register mesh metadata in Delta, compute normalization stats via Spark
2. **Train** — Distributed training via TorchDistributor on GPU cluster
3. **Evaluate** — Cached inference on validation data, log metrics to MLflow
4. **Register** — Log model + normalizers to Unity Catalog Model Registry
5. **Deploy** — Create/update Model Serving endpoint

---

## Running tests on Databricks

### Via DAB job

```bash
databricks bundle deploy -t a10g
databricks bundle run gpu_memory_benchmark     # GPU memory sweep
databricks bundle run distributed_sharded_test # 2-GPU sharding validation
```

### Via notebook (quick smoke test)

```python
# Run in a notebook cell on a GPU cluster
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "-x", "-k", "test_full_model",
     "/Workspace/Repos/<user>/Transolver/tests/"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
```

### Key test categories

```python
# Verify model forward pass
!python -m pytest -x -k "test_full_model" /Workspace/Repos/<user>/Transolver/tests/

# Verify cached inference matches direct forward
!python -m pytest -x -k "test_cached" /Workspace/Repos/<user>/Transolver/tests/

# Verify tiling produces same results
!python -m pytest -x -k "test_tiled" /Workspace/Repos/<user>/Transolver/tests/

# All tests
!python -m pytest /Workspace/Repos/<user>/Transolver/tests/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| CUDA OOM during training | `tile_size` too large | Reduce `tile_size` (try 50K, then 25K) |
| CUDA OOM during cache build | `cache_chunk_size` too large | Reduce `cache_chunk_size` |
| Loss stuck / not decreasing | LR too low or data not normalized | Check LR schedule, apply `TargetNormalizer` |
| Loss is NaN | Numerical instability | Enable mixed precision (`scaler`), check data for NaN |
| Slow training | No tiling on large mesh | Set `tile_size=100_000` |
| `RuntimeError: NCCL` | Distributed setup issue | Check cluster has multi-GPU instance type |
| `ModuleNotFoundError: transolver3` | Package not installed on cluster | Add `%pip install` cell or cluster init script |
| DAB deploy fails | Not authenticated | Run `databricks configure --token` |
| Job fails on wrong instance | Target mismatch | Check `databricks bundle deploy -t <target>` matches your needs |
