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

### GPU targets (for benchmarks and one-off runs)

| Target | Instance | GPU | VRAM | Use case |
|--------|----------|-----|------|----------|
| `a10g` (default) | g5.xlarge | 1x A10G | 24 GB | Benchmarks, small experiments |
| `a100_40` | p4d.24xlarge | 8x A100 | 320 GB | Multi-GPU training |
| `a100_80` | p4de.24xlarge | 8x A100-80 | 640 GB | Full-scale DrivAerML |

### Environment targets (for the promotion pipeline)

| Target | Catalog | Schema | Endpoint | GPU | Purpose |
|--------|---------|--------|----------|-----|---------|
| `dev` | `dev` | `transolver3` | `transolver3-serving-dev` | A10G | Development iteration |
| `staging` | `staging` | `transolver3` | `transolver3-serving-staging` | 8x A100 40GB | Validation before prod |
| `prod` | `prod` | `transolver3` | `transolver3-serving-prod` | 8x A100 80GB | Production serving |

The environment targets wire catalog/schema/endpoint names automatically — the same code runs across dev/staging/prod with different data isolation.

### DAB variables

All jobs use these configurable variables (set in `databricks.yml`, overridable per target or at deploy time):

| Variable | Default | Description |
|----------|---------|-------------|
| `data_volume_path` | `/Volumes/ml/transolver3/meshes` | UC Volume path to `.npz` mesh files |
| `catalog` | `ml` | Unity Catalog name for Delta tables and model registry |
| `schema` | `transolver3` | UC schema name |
| `model_name` | `transolver3` | Model name in UC Model Registry |
| `endpoint_name` | `transolver3-serving` | Serving endpoint name |
| `gpu_type` | `a10g` | GPU profile label |
| `node_type_id` | `g5.xlarge` | AWS instance type |

Override variables at deploy time:

```bash
# Point to your own data volume
databricks bundle deploy -t dev \
  --var data_volume_path=/Volumes/dev/transolver3/my_meshes

# Use a different model name
databricks bundle deploy -t staging \
  --var model_name=transolver3_v2
```

### Deploy and run

```bash
# Deploy to a GPU target for benchmarking
databricks bundle deploy -t a10g

# Available jobs:
databricks bundle run gpu_memory_benchmark     # Single-GPU memory sweep
databricks bundle run distributed_sharded_test # 2-GPU sharding validation
databricks bundle run training_workflow        # Full 5-task pipeline

# Deploy to an environment for the full pipeline
databricks bundle deploy -t dev
databricks bundle run training_workflow -t dev
```

### Full training workflow (5 tasks)

The DAB `training_workflow` (`resources/training_workflow.yml`) runs 5 sequential tasks. Each task creates its own cluster with the right GPU type and libraries (`einops`, `timm`, `mlflow`, `databricks-sdk`).

1. **preprocess** (`scripts/preprocess.py`)
   - Cluster: `i3.xlarge` (CPU, no GPU needed)
   - Params: `--data_dir ${data_volume_path} --catalog ${catalog} --schema ${schema}`
   - Registers mesh metadata in `${catalog}.${schema}.mesh_metadata`
   - Computes normalization stats in `${catalog}.${schema}.mesh_stats`

2. **train** (`Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py`)
   - Cluster: `${node_type_id}` (GPU), 8 GPUs via `spark.task.resource.gpu.amount: "8"`
   - Params: `--data_dir ${data_volume_path} --use-distributor --num-gpus 8 --epochs 500`
   - Uses TorchDistributor with mesh-sharded DDP
   - Logs metrics to MLflow throughout
   - Outputs: checkpoint to `./checkpoints/drivaer_ml_distributed/best_model.pt`

3. **evaluate** (`exp_drivaer_ml_distributed.py --eval_only`)
   - Cluster: `${node_type_id}` (GPU)
   - Params: `--data_dir ${data_volume_path} --eval_only --checkpoint ./checkpoints/.../best_model.pt`
   - Runs cached inference on validation data
   - Logs eval metrics (relative L2, per-channel errors) to MLflow

4. **register** (`scripts/register_model.py`)
   - Cluster: `i3.xlarge` (CPU)
   - Params: `--catalog ${catalog} --schema ${schema} --model_name ${model_name}`
   - Wraps model + normalizers in `TransolverPyfunc`
   - Registers in UC as `${catalog}.${schema}.${model_name}`

5. **deploy** (`scripts/deploy_endpoint.py`)
   - Cluster: `i3.xlarge` (CPU)
   - Params: `--catalog ${catalog} --schema ${schema} --model_name ${model_name} --endpoint_name ${endpoint_name}`
   - Creates/updates serving endpoint with scale-to-zero

### Customizing training parameters

The train task passes parameters directly to `exp_drivaer_ml_distributed.py`. To change epochs, learning rate, or model architecture, edit `resources/training_workflow.yml`:

```yaml
# In resources/training_workflow.yml, under the "train" task:
parameters:
  - "--data_dir"
  - "${var.data_volume_path}"
  - "--use-distributor"
  - "--num-gpus"
  - "8"
  - "--epochs"
  - "200"          # change from 500
  - "--lr"
  - "5e-4"         # add custom learning rate
  - "--n-layers"
  - "12"           # smaller model for faster iteration
```

### Monitoring DAB job runs

After `databricks bundle run`, the CLI prints a run URL. In the Databricks UI:

- **Workflow Runs** page: see the 5-task DAG, click each task for logs
- **MLflow Experiments** (`/Shared/transolver3-experiments`): training curves, eval metrics
- **Serving Endpoints** page: endpoint status, latency, request logs
- **Unity Catalog**: browse `${catalog}.${schema}` for Delta tables and registered models

### Serving endpoint resource

The DAB also declares a serving endpoint in `resources/serving_endpoint.yml`. This gets created/updated on `bundle deploy`:

```yaml
# resources/serving_endpoint.yml
resources:
  serving_endpoints:
    transolver3_endpoint:
      name: "${var.endpoint_name}"
      config:
        served_entities:
          - entity_name: "${var.catalog}.${var.schema}.${var.model_name}"
            entity_version: "1"
            workload_size: "Small"
            scale_to_zero_enabled: true
```

This means `databricks bundle deploy -t prod` automatically creates/updates `transolver3-serving-prod` pointing to `prod.transolver3.transolver3`.

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
| `RuntimeError: NCCL` | Distributed setup issue | Check cluster has multi-GPU instance type (e.g., `p4d.24xlarge`) |
| `ModuleNotFoundError: transolver3` | Package not installed on cluster | Add `%pip install` cell or check `libraries:` in `resources/*.yml` |
| `databricks bundle deploy` fails | Not authenticated | Run `databricks configure --token` |
| `Error: variable not set` | Missing variable override | Pass with `--var key=value` or check `databricks.yml` defaults |
| Job fails on wrong GPU | Target mismatch | Use `-t a100_40` or `-t prod` to pick the right instance type |
| Train task can't find data | Wrong `data_volume_path` | Override: `--var data_volume_path=/Volumes/dev/transolver3/data` |
| Wrong catalog in job output | Using GPU target not env target | Use `-t dev`/`-t staging`/`-t prod` which set catalog/schema automatically |
| Preprocess task fails | No `.npz` files in volume | Upload data first: `dbutils.fs.cp("file:/tmp/mesh.npz", "/Volumes/...")` |
