# Production Gap Analysis

Perspective: PhD data scientist migrating from Transolver v1 on Azure ML to distributed-transolver3 on Databricks.

## HIGH Risk (Blockers)

### 1. ~~AWS-only — no Azure support~~ DONE

Added Azure targets (`azure_a10`, `azure_a100`, `azure_a100_80`) to `databricks.yml` with mapped instance types. Removed `aws_attributes` from `training_workflow.yml` to make it cloud-agnostic.

**Files:** `databricks.yml`, `resources/training_workflow.yml`

### 2. ~~No checkpoint resumption~~ DONE

Added `--resume` flag that loads full training state (model + optimizer + scheduler + epoch + best_error). Training checkpoints saved every `--save_every` epochs (default: 10) to `{save_dir}/training_checkpoint.pt`. Resume continues from the exact epoch with correct LR schedule.

**Files:** `Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py`

### 3. ~~No early stopping~~ DONE

Added `--patience N` flag. Stops training if test error doesn't improve for N consecutive eval cycles (eval runs every 10 epochs). `--patience 0` (default) disables early stopping. Also refactored the eval block to reduce duplication between shard_eval and non-shard_eval paths.

**Files:** `Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py`

## MEDIUM Risk (Production friction)

### 4. No data validation

- `.npz` files loaded with no schema enforcement — missing keys or shape mismatches crash at runtime, not at load time
- No NaN/inf detection before training begins
- No data versioning beyond timestamps in the metadata Delta table

**Files:** `Industrial-Scale-Benchmarks/dataset/drivaer_ml.py`, `transolver3/data_catalog.py`, `scripts/preprocess.py`

### 5. Serving has no input validation or health checks

- `TransolverPyfunc.predict()` doesn't validate coordinate shapes or dtypes
- No error handling — exceptions bubble up raw to the caller
- No request timeout for large batches
- Monitoring framework exists in `monitoring.py` but isn't wired into the serving endpoint

**Files:** `transolver3/serving.py`, `transolver3/monitoring.py`

### 6. CI doesn't test deployment

- CI runs lint + 100 CPU tests — good for code quality
- But no DAB bundle validation (`databricks bundle validate`)
- No GPU tests, no distributed tests, no serving endpoint smoke test
- A broken `training_workflow.yml` would pass CI

**Files:** `.github/workflows/ci.yml`

## LOW Risk (Nice to have)

### 7. No gradient accumulation

Effective batch size is fixed at `batch_size x world_size`. No `--gradient_accumulation_steps` flag.

### 8. No v1 to v3 migration guide

Experiments compare v1 vs v3 results (`experiments/COMPARE_v1v3.md`), but no docs on how to port a v1 training pipeline to v3.

### 9. Reproducibility is manual

Seeds and configs are logged correctly to MLflow, but there's no single command to reproduce a past run from its MLflow run ID.

## What's already solid

- **100 tests** with numerical equivalence verification (tiled vs standard, cached vs direct)
- **MLflow integration** is well-done (live per-epoch metrics, model logging, run_id-based artifact flow across tasks)
- **Mesh sharding** strategy is correct and validated on 4x A10G
- **Per-rank logging** with timestamps via `print(flush=True)` — reliable in TorchDistributor subprocess pipes
- **Monitoring framework** exists (PSI drift detection, physical bounds checking) — just needs wiring into serving
- **Distributed architecture** documented with Mermaid diagrams (`SPECS/DISTRIBUTED_ARCHITECTURE.md`)
