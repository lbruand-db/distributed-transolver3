# Transolver-3

Physics-informed transformer solver for industrial-scale CFD simulations (100M+ mesh cells). PyTorch-based, with Databricks integration for training, tracking, and deployment.

## Quick Reference

- **Language**: Python 3.10+
- **Framework**: PyTorch >= 2.0
- **Package manager**: uv
- **Test runner**: pytest (`uv run pytest`)
- **Core package**: `transolver3/`
- **100 tests**, all should pass

## Common Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run all tests
uv run pytest -x           # Stop on first failure
uv run pytest -k "test_cached"  # Run specific tests
uv run ruff check .        # Lint
uv run ruff format .       # Format (run before committing)

# Databricks Asset Bundle (requires `databricks auth login` first)
databricks bundle deploy -t a10g --profile DEFAULT
databricks bundle run transolver3_training_pipeline -t a10g --profile DEFAULT  # Full 5-task pipeline
databricks bundle run test_mlflow_auth -t a10g --profile DEFAULT               # Smoke test MLflow auth
databricks bundle run test_register_deploy -t a10g --profile DEFAULT           # Serverless register+deploy
```

## Architecture

The model is an encoder-only transformer with Physics-Attention layers that operate in a compressed "slice domain" (M=64 slices << N mesh points).

**Three-phase pipeline**:
1. **Training** — Geometry amortized: subsample 100K-400K points per iteration via `AmortizedMeshSampler`
2. **Cache build** — Stream full mesh in chunks, accumulate slice-domain states (~768 KB total cache for 24 layers)
3. **Decode** — Reconstruct predictions at any query points using the cache

**Key files**:
- `transolver3/physics_attention_v3.py` — Core attention mechanism (Algorithm 1 from paper)
- `transolver3/model.py` — `Transolver3` model class with `cache_physical_states()` and `decode_from_cache()`
- `transolver3/amortized_training.py` — `train_step()`, optimizer/scheduler factories, `AmortizedMeshSampler`
- `transolver3/inference.py` — `CachedInference` and `DistributedCachedInference` wrappers
- `transolver3/distributed.py` — DDP mesh sharding utilities
- `transolver3/normalizer.py` — `InputNormalizer` (min-max), `TargetNormalizer` (z-score)
- `transolver3/databricks_training.py` — TorchDistributor launcher + Databricks auth propagation
- `transolver3/serving.py` — MLflow pyfunc wrapper (`TransolverPyfunc`) + UC Model Registry + serving endpoint
- `Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py` — Multi-GPU training script

## Data Format

Input data is `.npz` files (NumPy compressed) loaded via `np.load(mmap_mode='r')` for memory-mapped I/O. A single DrivAerML sample is ~12 GB (140M points × 22 features × 4 bytes). Use `validate_npz()` from `dataset/drivaer_ml.py` to check schema and data quality.

## Distributed Training

Uses `torch.distributed` DDP with mesh sharding: each GPU loads 1/K of the mesh via mmap range reads, computes local slice accumulators, and all-reduces them (~514 KB/layer). On Databricks, TorchDistributor launches torchrun in `local_mode=True` on a single multi-GPU node (e.g., `g5.12xlarge` with 4x A10G).

**Key design decisions** (learned from production debugging):
- Pass **script path** (not function) to `TorchDistributor.run()` — avoids cloudpickle module resolution failures
- Script path resolved via `__code__.co_filename` — works in Databricks `exec(compile(...))` context
- **Databricks auth propagation**: driver extracts `DATABRICKS_HOST`/`TOKEN` from MLflow's credential provider and sets as env vars before forking, so child processes can reach MLflow
- **Logging**: use `print(..., flush=True)` not `logging` — TorchDistributor captures subprocess stdout via pipe
- `NCCL_P2P_DISABLE=1` for PCIe-connected GPUs (g5.12xlarge A10Gs)
- MLflow logging only on rank 0, live per-epoch metrics

## DAB Pipeline

The 5-task pipeline runs on Databricks Asset Bundles. MLflow is the single source of truth for model artifacts — no checkpoint files passed between tasks.

| Task | Cluster | Purpose |
|------|---------|---------|
| preprocess | i3.xlarge (CPU) | Mesh metadata + stats to Delta |
| train | g5.12xlarge (4x A10G) | Mesh-sharded DDP, live MLflow metrics |
| evaluate | g5.12xlarge (4x A10G) | Load model from MLflow run, cached inference |
| register | i3.xlarge (CPU) | Promote model to UC Model Registry |
| deploy | i3.xlarge (CPU) | Create/update serving endpoint |

**Training features**: `--resume` (checkpoint resumption), `--patience` (early stopping), `--accumulation_steps` (gradient accumulation), `--save_every` (periodic checkpoints).

**Fast iteration jobs** (serverless, no GPU):
- `test_mlflow_auth` — verifies auth propagation to TorchDistributor children
- `test_register` — inspects checkpoint and tests model loading
- `test_register_deploy` — end-to-end register + deploy without training

## Databricks Integration

- **MLflow**: Live per-epoch metrics at `/Shared/transolver3-experiments`, model artifacts logged with signature
- **UC Model Registry**: Models promoted from MLflow runs (no re-logging)
- **Model Serving**: Scale-to-zero endpoint via `TransolverPyfunc` with input validation
- **UC Volumes**: `.npz` mesh data, checkpoints, `mlflow_run_id.txt` for cross-task communication
- **DABs**: `databricks.yml` defines AWS + Azure targets with job resources in `resources/*.yml`

## Specs & Design Docs

- `SPECS/SPEC.md` — Core v3 architecture specification
- `SPECS/DISTRIBUTED.md` — Multi-GPU distribution design (3 implementation options)
- `SPECS/DISTRIBUTED_ARCHITECTURE.md` — Mermaid diagrams: pipeline, process model, data flow, logging
- `SPECS/GAPS.md` — Production gap analysis (all 9 gaps addressed)
- `SPECS/MIGRATION_V1_TO_V3.md` — Guide for migrating from Transolver v1 / Azure ML
- `SPECS/DIFFERENTIATORS.md` — Why Databricks is ideal for this workload
- `SPECS/VALUEADDED.md` — Databricks stickiness plan with prioritized integrations
- `SPECS/CRITICAL_ISSUES.md` — Known issues
- `WORKFLOW.md` — Three-step pipeline description

## Style Guidelines

- Keep dependencies minimal (torch, einops, timm, numpy)
- Use `einops` for tensor reshaping, not manual view/permute chains
- All attention uses `torch.nn.functional.scaled_dot_product_attention`
- Mixed precision via `torch.amp`, not manual casting
- Tests verify numerical equivalence between tiled/untiled and cached/direct paths (tolerance ~1e-7)
- **Formatter/Linter**: ruff — run `uv run ruff format .` and `uv run ruff check .` before committing
- **Logging in distributed training**: use `print(..., flush=True)` with `log()` (rank 0) or `logall()` (all ranks + timestamp)
- **CI**: ruff lint, ty type check, pytest, yamllint on DAB YAML files
