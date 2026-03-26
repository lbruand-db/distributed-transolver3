# Transolver-3

Physics-informed transformer solver for industrial-scale CFD simulations (100M+ mesh cells). PyTorch-based, with Databricks integration for training, tracking, and deployment.

## Quick Reference

- **Language**: Python 3.10+
- **Framework**: PyTorch >= 2.0
- **Package manager**: uv
- **Test runner**: pytest (`uv run pytest`)
- **Core package**: `transolver3/`
- **52 tests**, all should pass

## Common Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run all tests
uv run pytest -x           # Stop on first failure
uv run pytest -k "test_cached"  # Run specific tests
uv run ruff check .        # Lint
uv run ruff format .       # Format (run before committing)

# Databricks Asset Bundle
databricks bundle deploy -t a10g
databricks bundle run gpu_memory_benchmark
databricks bundle run distributed_sharded_test
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
- `Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py` — Multi-GPU training script

## Data Format

Input data is `.npz` files (NumPy compressed) loaded via `np.load(mmap_mode='r')` for memory-mapped I/O. A single DrivAerML sample is ~12 GB (140M points × 22 features × 4 bytes).

## Distributed Training

Uses `torch.distributed` DDP with mesh sharding: each GPU loads 1/K of the mesh via mmap range reads, computes local slice accumulators, and all-reduces them (~514 KB/layer). On Databricks, use `TorchDistributor` on multi-GPU instances.

## Databricks Integration

- **MLflow**: Experiment tracking and model logging (see README Get Started example)
- **DABs**: `databricks.yml` defines 3 GPU targets (a10g, a100_40, a100_80) with job resources in `resources/*.yml`
- **Roadmap**: See `SPECS/VALUEADDED.md` for planned Delta Lake, UC Volumes, Model Serving, and Workflow integrations

## Specs & Design Docs

- `SPECS/SPEC.md` — Core v3 architecture specification
- `SPECS/DISTRIBUTED.md` — Multi-GPU distribution design (3 implementation options)
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