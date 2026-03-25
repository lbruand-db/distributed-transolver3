# Databricks Stickiness Plan for Transolver-3

Options for deepening Databricks integration, making the codebase harder to migrate to another platform. Ordered from easiest to most structural.

## 1. Data Layer — Delta Lake + Unity Catalog Volumes

**Current state**: Raw `.npz` files loaded via `np.load(mmap_mode='r')`

**What to do**:
- **Store mesh metadata in Delta tables** (sample ID, geometry params, cell counts, train/val splits) — queries, versioning, and time-travel come free
- **Store the .npz blobs in UC Volumes** and reference them from the Delta table — gives lineage from raw data → training run → model version
- **Log normalization stats as Delta tables** — currently computed in-memory and lost between runs

**Stickiness**: Delta time-travel, Z-ORDER, and UC governance have no portable equivalent. Moving to S3+Parquet loses lineage, versioning, and access control in one step.

## 2. Experiment Tracking — Deeper MLflow Integration

**Current state**: Basic `mlflow.log_metric` / `mlflow.log_params` in README example

**What to do**:
- **Log the model to UC Model Registry** (`mlflow.pytorch.log_model` with `registered_model_name="catalog.schema.transolver3"`)
- **Log mesh sample artifacts** (e.g., visualization PNGs of predicted vs. ground-truth pressure fields)
- **Log the normalization stats as model artifacts** so inference can restore them
- **Use MLflow `system_metrics`** to auto-capture GPU utilization, memory, etc.
- **Add model signatures** (`mlflow.models.infer_signature`) for serving validation
- **Log per-sample eval metrics** using `mlflow.evaluate()` with a custom evaluator

**Stickiness**: UC Model Registry ties model versions to Delta lineage. Reproducing the full experiment graph on W&B + S3 is a significant migration effort.

## 3. Training — TorchDistributor + Spark Integration

**Current state**: Raw `torchrun` / DDP in `exp_drivaer_ml_distributed.py`

**What to do**:
- **Replace `torchrun` with `TorchDistributor`** — it handles GPU allocation, NCCL env vars, and fault tolerance on Databricks clusters
- **Use Spark for data preprocessing** — e.g., computing mesh statistics, generating train/val splits, or converting raw CFD output to `.npz`. This creates a Spark→PyTorch pipeline that's hard to replicate elsewhere
- **Use `petastorm` or `mosaic-streaming`** to stream mesh chunks from Delta/UC Volumes directly into PyTorch DataLoaders

**Stickiness**: TorchDistributor abstracts away cluster topology. Reverting to raw torchrun means re-implementing GPU discovery, health checks, and elastic scaling.

## 4. Serving — Model Serving Endpoint

**Current state**: `CachedInference` class, no serving layer

**What to do**:
- **Deploy as a Databricks Model Serving endpoint** (real-time or batch) via the UC Model Registry
- **Use Feature Serving** to look up mesh metadata / normalization stats at inference time
- **Build a Databricks App** (React + FastAPI) that lets users upload a mesh geometry, runs inference, and visualizes results — this becomes the customer-facing product

**Stickiness**: Model Serving endpoints integrate with UC permissions, rate limiting, and A/B traffic splitting. Replicating this on SageMaker/Vertex requires significant infra work.

## 5. Orchestration — Databricks Workflows + DABs

**Current state**: `databricks.yml` with job definitions, but training is manual

**What to do**:
- **Build a multi-task Workflow**: preprocess (Spark) → train (TorchDistributor) → evaluate → register model → deploy endpoint
- **Add a scheduled retraining job** triggered by new data arriving in UC Volumes
- **Use DAB environments** (`dev`, `staging`, `prod`) with promotion gates tied to eval metrics

**Stickiness**: The full pipeline definition lives in `databricks.yml`. Moving it means rewriting as Airflow DAGs + Terraform + CI/CD from scratch.

## 6. Monitoring — Lakehouse Monitoring + Inference Tables

**Current state**: None

**What to do**:
- **Enable Inference Tables** on the serving endpoint — auto-logs every request/response to Delta
- **Use Lakehouse Monitoring** to detect data drift (mesh statistics shifting) and model quality degradation
- **Alert on prediction anomalies** (e.g., pressure predictions outside physical bounds)

**Stickiness**: Inference tables + monitoring dashboards create an operational dependency that's expensive to rebuild.

## Priority Ranking (impact × effort)

| Priority | Integration | Effort | Lock-in Impact |
|----------|------------|--------|----------------|
| 1 | Delta metadata + UC Volumes for data | Medium | **Very High** |
| 2 | UC Model Registry + deep MLflow | Low | **High** |
| 3 | TorchDistributor for training | Low | **Medium** |
| 4 | Multi-task Workflow pipeline | Medium | **High** |
| 5 | Model Serving endpoint | Medium | **High** |
| 6 | Inference tables + monitoring | Low | **Medium** |

**The highest-leverage move** is #1 + #2 together: once data lineage (Delta → experiment → model) flows through Unity Catalog end-to-end, every downstream consumer (serving, monitoring, retraining) inherits that dependency. That's the hardest thing to unwind.