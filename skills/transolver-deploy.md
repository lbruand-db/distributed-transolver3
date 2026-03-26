# transolver-deploy

Databricks lifecycle management — model registration, serving, monitoring, and the full training workflow.

---

## When to use

Use this skill when the user wants to:
- Register a trained model in Unity Catalog
- Deploy a model to a Databricks serving endpoint
- Set up inference table logging and quality monitoring
- Run the 5-task DAB training workflow
- Manage DAB targets and cluster configurations
- Set up the end-to-end MLflow tracking pipeline
- Query a deployed model endpoint

---

## Notebook setup (first cell)

```python
%pip install /Workspace/Repos/<user>/Transolver -q
dbutils.library.restartPython()
```

---

## Step 1: MLflow experiment tracking (notebook)

```python
import mlflow
from transolver3.amortized_training import train_step

mlflow.set_experiment("/Shared/transolver3-experiments")

with mlflow.start_run(run_name="drivaer-24layer"):
    mlflow.log_params({
        "n_layers": 24, "n_hidden": 256, "slice_num": 64,
        "tile_size": 100_000, "total_steps": 50_000,
        "lr": 1e-3, "weight_decay": 0.05,
    })

    for step in range(total_steps):
        loss = train_step(model, x, fx, target, optimizer, scheduler,
                          sampler=sampler, tile_size=100_000)
        if step % 100 == 0:
            mlflow.log_metric("train_loss", loss, step=step)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=step)

    mlflow.log_metric("eval_rel_l2", eval_loss)
```

---

## Step 2: Register model in Unity Catalog (notebook)

```python
from transolver3.serving import register_serving_model

# Model config must match constructor args exactly
model_config = dict(
    space_dim=3, n_layers=24, n_hidden=256, n_head=8,
    slice_num=64, out_dim=4, tile_size=100_000,
)

# Register with normalizers
model_info = register_serving_model(
    model=model,
    config=model_config,
    normalizers={
        "input": input_norm,    # fitted InputNormalizer
        "target": target_norm,  # fitted TargetNormalizer
    },
    catalog="ml",
    schema="transolver3",
    model_name="transolver3",
)
print(f"Registered: {model_info.registered_model_name}")
```

**What this does:**
- Saves model state dict, config JSON, and normalizer state dicts as MLflow artifacts
- Wraps everything in a `TransolverPyfunc` that handles normalization + cached inference
- Registers in Unity Catalog as `ml.transolver3.transolver3`

---

## Step 3: Deploy serving endpoint (notebook)

```python
from transolver3.serving import deploy_serving_endpoint

result = deploy_serving_endpoint(
    model_name="transolver3",
    endpoint_name="transolver3-serving",
    catalog="ml",
    schema="transolver3",
)
```

### Query the endpoint from a notebook

```python
from databricks.sdk import WorkspaceClient
import requests

client = WorkspaceClient()
endpoint_url = f"{client.config.host}/serving-endpoints/transolver3-serving/invocations"

payload = {
    "inputs": {
        "coordinates": coords.tolist()  # (N, 3) list of lists
    }
}

response = requests.post(
    endpoint_url,
    headers={"Authorization": f"Bearer {client.config.token}"},
    json=payload,
)
predictions = response.json()
```

---

## Step 4: Set up monitoring (notebook)

### Enable inference table auto-logging

```python
from transolver3.monitoring import setup_inference_table

setup_inference_table(
    endpoint_name="transolver3-serving",
    catalog="ml",
    schema="transolver3",
)
# Logs every request/response to ml.transolver3.transolver3_serving_payload
```

### Create a quality monitor

```python
from transolver3.monitoring import create_quality_monitor

create_quality_monitor(
    catalog="ml",
    schema="transolver3",
    table_name="transolver3_serving_payload",
)
# Runs hourly drift/quality checks via Lakehouse Monitoring
```

### Log drift metrics to Delta

```python
from transolver3.monitoring import log_drift_metrics

drift = log_drift_metrics(
    spark=spark,
    catalog="ml",
    schema="transolver3",
    predictions=predictions,
    baseline_stats={
        "mean": target_norm.mean.squeeze().tolist(),
        "std": target_norm.std.squeeze().tolist(),
    },
)
# Writes per-channel PSI to ml.transolver3.drift_metrics
```

### Query drift history

```python
drift_df = spark.read.table("ml.transolver3.drift_metrics")
display(drift_df.orderBy("logged_at", ascending=False))
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

### Training workflow tasks

The DAB `training_workflow` runs 5 sequential tasks:

1. **preprocess** (`scripts/preprocess.py`)
   - Registers mesh metadata in Delta via `register_mesh_metadata()`
   - Computes normalization stats via `preprocess_with_spark()`
   - Outputs: Delta tables with mesh info + normalization params

2. **train** (distributed)
   - Launches via `TorchDistributor` on the GPU cluster
   - Uses amortized training with random subsets
   - Logs metrics to MLflow throughout
   - Outputs: trained model checkpoint

3. **evaluate**
   - Runs cached inference on validation data
   - Computes relative L2, per-channel errors, bounds checks
   - Logs eval metrics to MLflow

4. **register** (`scripts/register_model.py`)
   - Wraps model + normalizers in `TransolverPyfunc`
   - Registers in Unity Catalog Model Registry
   - Output: `ml.transolver3.transolver3` model version

5. **deploy** (`scripts/deploy_endpoint.py`)
   - Creates/updates serving endpoint
   - Enables inference table auto-logging
   - Output: live `transolver3-serving` endpoint

---

## Data management with Unity Catalog Volumes

```python
from transolver3.data_catalog import register_mesh_metadata, get_mesh_metadata

# Register a mesh dataset
register_mesh_metadata(
    spark=spark,
    catalog="ml",
    schema="transolver3",
    mesh_name="drivaer_ml_001",
    data_path="/Volumes/ml/transolver3/data/drivaer_001.npz",
    num_points=140_000_000,
    space_dim=3,
    features=["pressure", "velocity_x", "velocity_y", "velocity_z"],
)

# Query registered meshes
meshes = get_mesh_metadata(spark, catalog="ml", schema="transolver3")
display(meshes)
```

---

## End-to-end checklist

For a newcomer deploying Transolver-3 on Databricks:

- [ ] **Data**: Upload `.npz` mesh files to UC Volumes (`/Volumes/ml/transolver3/data/`)
- [ ] **Validate**: Use `transolver-data` skill to inspect and validate mesh files in a notebook
- [ ] **Configure**: Choose a config preset from `transolver-run` skill
- [ ] **Deploy DAB**: `databricks bundle deploy -t <target>`
- [ ] **Benchmark**: `databricks bundle run gpu_memory_benchmark` — verify GPU fits your config
- [ ] **Train**: `databricks bundle run training_workflow` or run training notebook on GPU cluster
- [ ] **Evaluate**: Use `transolver-analyze` skill to check results and bounds in a notebook
- [ ] **Register**: Model auto-registered by workflow, or manually via `register_serving_model()` in notebook
- [ ] **Serve**: Endpoint auto-created by workflow, or manually via `deploy_serving_endpoint()` in notebook
- [ ] **Monitor**: Set up inference table + quality monitor + drift logging in notebook
- [ ] **Iterate**: Query `ml.transolver3.drift_metrics`, retrain if PSI > 0.2

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `databricks bundle deploy` fails | Not authenticated | Run `databricks configure --token` |
| Job fails: `ModuleNotFoundError` | Missing deps on cluster | Add `%pip install` to init script or first notebook cell |
| Serving endpoint 503 | Model still loading | Wait 2-5 min for cold start, check endpoint logs in workspace |
| Inference table empty | Auto-capture not enabled | Run `setup_inference_table()` in a notebook |
| `PermissionError` on UC | Missing grants | Need `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE` grants |
| Slow serving latency | No GPU on serving | Check endpoint config uses GPU workload type |
| `spark` not defined | Not in a Databricks notebook | Use Databricks notebook or `from pyspark.sql import SparkSession` |
