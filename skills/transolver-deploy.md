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

### Environment promotion path

The DAB defines 3 environment targets for the dev-to-prod promotion pipeline:

| Target | Catalog | Schema | Endpoint | GPU | Purpose |
|--------|---------|--------|----------|-----|---------|
| `dev` | `dev` | `transolver3` | `transolver3-serving-dev` | A10G | Development iteration |
| `staging` | `staging` | `transolver3` | `transolver3-serving-staging` | 8x A100 40GB | Pre-prod validation |
| `prod` | `prod` | `transolver3` | `transolver3-serving-prod` | 8x A100 80GB | Production serving |

Plus 3 GPU-only targets for benchmarks: `a10g` (default), `a100_40`, `a100_80`.

### DAB variables

All jobs and resources use these variables, overridable per target or at deploy time:

| Variable | Default | Description |
|----------|---------|-------------|
| `data_volume_path` | `/Volumes/ml/transolver3/meshes` | UC Volume path to `.npz` mesh files |
| `catalog` | `ml` | Unity Catalog name |
| `schema` | `transolver3` | UC schema name |
| `model_name` | `transolver3` | Model name in UC registry |
| `endpoint_name` | `transolver3-serving` | Serving endpoint name |

Override at deploy time:

```bash
databricks bundle deploy -t staging \
  --var data_volume_path=/Volumes/staging/transolver3/validated_meshes \
  --var model_name=transolver3_v2
```

### Deploy and run

```bash
# Development cycle
databricks bundle deploy -t dev
databricks bundle run training_workflow -t dev

# Promotion to staging (larger GPU, staging catalog)
databricks bundle deploy -t staging
databricks bundle run training_workflow -t staging

# Production deployment
databricks bundle deploy -t prod
databricks bundle run training_workflow -t prod
```

### Training workflow tasks (detailed)

The `training_workflow` (`resources/training_workflow.yml`) runs 5 sequential tasks. Each creates its own ephemeral cluster with the correct GPU type, Spark version (`15.4.x-gpu-ml-scala2.12`), and pip libraries.

1. **preprocess** (`scripts/preprocess.py`)
   - Cluster: `i3.xlarge` (CPU only)
   - Params: `--data_dir ${data_volume_path} --catalog ${catalog} --schema ${schema}`
   - Writes: `${catalog}.${schema}.mesh_metadata`, `${catalog}.${schema}.mesh_stats`

2. **train** (`Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py`)
   - Cluster: `${node_type_id}` with 8 GPUs
   - Params: `--data_dir ${data_volume_path} --use-distributor --num-gpus 8 --epochs 500`
   - TorchDistributor handles DDP, mesh sharding, and NCCL
   - Logs to MLflow: `train_loss`, `lr`, `eval_rel_l2` per step
   - Writes checkpoint: `./checkpoints/drivaer_ml_distributed/best_model.pt`

3. **evaluate** (`exp_drivaer_ml_distributed.py --eval_only`)
   - Cluster: `${node_type_id}` (GPU)
   - Params: `--eval_only --checkpoint ./checkpoints/.../best_model.pt`
   - Cached inference on validation data, logs eval metrics to MLflow

4. **register** (`scripts/register_model.py`)
   - Cluster: `i3.xlarge` (CPU)
   - Params: `--catalog ${catalog} --schema ${schema} --model_name ${model_name}`
   - Wraps in `TransolverPyfunc`, registers as `${catalog}.${schema}.${model_name}`

5. **deploy** (`scripts/deploy_endpoint.py`)
   - Cluster: `i3.xlarge` (CPU)
   - Params: `--endpoint_name ${endpoint_name} --catalog ${catalog} --schema ${schema} --model_name ${model_name}`
   - Creates/updates the serving endpoint

### Serving endpoint resource

The DAB also declares the endpoint declaratively in `resources/serving_endpoint.yml`:

```yaml
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

`databricks bundle deploy -t prod` creates `transolver3-serving-prod` pointing to `prod.transolver3.transolver3`. No manual API calls needed.

### Monitoring DAB job runs

After `databricks bundle run`, the CLI prints a run URL. In the Databricks workspace:

- **Workflows > Job Runs**: 5-task DAG view, click each task for stdout/stderr logs
- **MLflow Experiments** (`/Shared/transolver3-experiments`): training loss curves, eval metrics, registered model artifacts
- **Models** (UC): browse `${catalog}.${schema}.${model_name}` for versions, lineage
- **Serving Endpoints**: health, latency, scale status, inference logs
- **Data Explorer**: `${catalog}.${schema}.mesh_metadata`, `drift_metrics` tables

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

### First time setup
- [ ] **Auth**: `databricks configure --token` (or use a Databricks config profile)
- [ ] **Data**: Upload `.npz` mesh files to UC Volumes (`/Volumes/<catalog>/transolver3/meshes/`)
- [ ] **Validate**: Use `transolver-data` skill to inspect and validate mesh files in a notebook

### Development (dev target)
- [ ] **Deploy DAB**: `databricks bundle deploy -t dev`
- [ ] **Benchmark**: `databricks bundle run gpu_memory_benchmark -t dev` — verify GPU fits your config
- [ ] **Train**: `databricks bundle run training_workflow -t dev` (or notebook on GPU cluster)
- [ ] **Evaluate**: Check MLflow metrics + use `transolver-analyze` skill for bounds/drift
- [ ] **Iterate**: Adjust training params in `resources/training_workflow.yml`, redeploy

### Promotion to production
- [ ] **Staging**: `databricks bundle deploy -t staging && databricks bundle run training_workflow -t staging`
- [ ] **Validate**: Confirm eval metrics in staging match or improve on dev
- [ ] **Prod**: `databricks bundle deploy -t prod && databricks bundle run training_workflow -t prod`
- [ ] **Monitor**: Set up inference table + quality monitor + drift logging (see Step 4 above)
- [ ] **Ongoing**: Query `${catalog}.${schema}.drift_metrics`, retrain if PSI > 0.2

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `databricks bundle deploy` fails | Not authenticated | Run `databricks configure --token` or check `DATABRICKS_HOST`/`DATABRICKS_TOKEN` env vars |
| `Error: variable not set` | Missing `--var` override | Check `databricks.yml` for required variables, pass with `--var key=value` |
| Job fails: `ModuleNotFoundError` | Missing deps on cluster | Libraries are declared in `resources/*.yml` — check the `libraries:` section of the failing task |
| Train task OOM | Instance too small for mesh | Switch to a larger target: `databricks bundle deploy -t a100_80` |
| Register task fails | Checkpoint path mismatch | Verify `--checkpoint` param in `training_workflow.yml` matches the train task output path |
| Serving endpoint 503 | Model still loading | Wait 2-5 min for cold start, check endpoint logs in Serving Endpoints UI |
| Endpoint serves wrong model version | `entity_version: "1"` hardcoded | Update `serving_endpoint.yml` or use `deploy_endpoint.py` which auto-detects latest version |
| Inference table empty | Auto-capture not enabled | Run `setup_inference_table()` in a notebook after endpoint is created |
| `PermissionError` on UC | Missing grants on target catalog | Need `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE` on the target catalog (dev/staging/prod) |
| Slow serving latency | No GPU on serving endpoint | Endpoint uses `workload_size: "Small"` (CPU) — change to GPU workload in `serving_endpoint.yml` |
| `spark` not defined | Not in a Databricks notebook | Use Databricks notebook or `from pyspark.sql import SparkSession` |
| Wrong catalog/schema in prod | Variables not overridden | Targets `dev`/`staging`/`prod` auto-set catalog/schema — use `databricks bundle deploy -t prod` |
