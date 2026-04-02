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

### 4. ~~No data validation~~ DONE

Added `validate_npz()` function that checks schema (required keys per field type), NaN/Inf detection, and coordinate dimensions. Dataset accepts `validate=True` to run validation at construction time. Data versioning remains a future improvement.

**Files:** `Industrial-Scale-Benchmarks/dataset/drivaer_ml.py`

### 5. ~~Serving has no input validation or health checks~~ DONE

Added input validation to `TransolverPyfunc.predict()`: checks for missing 'coordinates' key, validates shape (2D or 3D) and `space_dim` match, detects NaN/Inf, and wraps parse errors with clear messages. Monitoring integration and request timeouts remain future improvements.

**Files:** `transolver3/serving.py`

### 6. ~~CI doesn't test deployment~~ DONE

Added `validate-dab` job to CI that installs the Databricks CLI and runs `databricks bundle validate` to catch YAML syntax errors in DAB configs. Uses a dummy host/token since validation only checks syntax, not workspace connectivity. GPU tests and endpoint smoke tests remain future improvements.

**Files:** `.github/workflows/ci.yml`

## LOW Risk (Nice to have)

### 7. ~~No gradient accumulation~~ DONE

Added `--accumulation_steps N` flag. Loss is scaled by 1/N, gradients accumulate over N micro-batches before optimizer step. Effective batch = `batch_size x world_size x accumulation_steps`.

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
