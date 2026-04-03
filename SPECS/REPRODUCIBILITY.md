# Reproducibility

How to reproduce a training run, what is deterministic, and what is not.

## Reproducing a Past Run

Every training run logs its full configuration to MLflow at `/Shared/transolver3-experiments`. To reproduce:

```bash
python scripts/reproduce_run.py --run_id <mlflow_run_id>
```

This fetches all parameters (seed, subset_size, n_layers, lr, etc.) and prints the exact CLI command. Add `--execute` to run it directly.

## Seeds and Determinism

The `--seed` flag (default: 42) controls all sources of randomness. Each GPU derives its own seed as `seed + rank` for independent but reproducible random streams.

### What is seeded

| Source | Seed | Mechanism |
|--------|------|-----------|
| Model weight initialization | `torch.manual_seed(seed)` | Global PyTorch RNG |
| CUDA operations | `torch.cuda.manual_seed_all(seed)` | All GPU RNGs |
| cuDNN algorithm selection | `cudnn.deterministic=True` | Disables non-deterministic autotuner |
| cuDNN benchmarking | `cudnn.benchmark=False` | Prevents algorithm search variance |
| `AmortizedMeshSampler` | `seed + rank` | `torch.Generator` per sampler instance |
| `DrivAerMLDataset._subsample` | `seed + rank + call_count` | `torch.Generator` per `__getitem__` call |
| `DistributedSampler` | `set_epoch(epoch)` | Deterministic shuffle per epoch |

### What is NOT deterministic

| Source | Why | Impact |
|--------|-----|--------|
| NCCL all-reduce | Float summation order across GPUs is hardware-dependent | Differences < 1e-7 per step, may compound over epochs |
| Multi-GPU vs single-GPU | Different gradient averaging (sum of 4 partial gradients vs 1 full gradient) | Expected divergence even with same seed |
| Different `world_size` | `subset_size / world_size` changes per-GPU point count | Different training dynamics |
| `upsample_point_cloud` | Seed parameter exists but not wired to dataset | Different upsampled points each call |

### Reproducibility guarantees

| Configuration | Guarantee |
|--------------|-----------|
| Same seed, same GPU, 1 process | **Bitwise reproducible** |
| Same seed, same GPU count, same hardware | **Reproducible up to NCCL rounding** (~1e-7 per step) |
| Same seed, different GPU count | **Not reproducible** (different per-GPU subsets and gradient averaging) |
| Same seed, different hardware (A10G vs A100) | **Not reproducible** (different cuDNN implementations, different float rounding) |

## How Seeds Flow Through the Pipeline

```
--seed 42, world_size=4

Driver process:
  torch.manual_seed(42)           ← model weight init
  torch.cuda.manual_seed_all(42)  ← CUDA RNGs

Rank 0:                           Rank 1:
  AmortizedMeshSampler(seed=42)     AmortizedMeshSampler(seed=43)
  Dataset(seed=42)                  Dataset(seed=43)
    _subsample: Generator(42+0)       _subsample: Generator(43+0)
    _subsample: Generator(42+1)       _subsample: Generator(43+1)
    ...                               ...

Rank 2:                           Rank 3:
  AmortizedMeshSampler(seed=44)     AmortizedMeshSampler(seed=45)
  Dataset(seed=44)                  Dataset(seed=45)
    _subsample: Generator(44+0)       _subsample: Generator(45+0)
    ...                               ...
```

Each rank gets a unique but deterministic random stream. This ensures:
- Different ranks see different mesh subsets (good for coverage)
- The same rank sees the same subsets across runs (good for reproducibility)

## What is Logged to MLflow

Every run logs these parameters, sufficient to reproduce the configuration:

```
field, epochs, lr, weight_decay, subset_size,
n_layers, n_hidden, n_head, slice_num, num_tiles,
world_size, shard_mesh, seed
```

Plus per-epoch metrics: `train_loss`, `epoch_time_s`, `gpu_peak_mb`, and `best_test_l2`.

## Performance Impact of Deterministic Mode

`cudnn.deterministic=True` and `cudnn.benchmark=False` can reduce training throughput by 5-15% because:
- Deterministic algorithms may be slower than the fastest non-deterministic ones
- Disabling benchmarking prevents cuDNN from searching for optimal kernel implementations

For production training at scale, consider disabling deterministic mode:
```bash
# Add to training script or set env var:
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for full determinism with cublas
```

For debugging and validation, deterministic mode is essential. For final production runs where throughput matters, it can be relaxed.

## Verifying Reproducibility

Run the same configuration twice and compare:

```bash
# Run 1
python exp_drivaer_ml_distributed.py --data_dir /path --seed 42 --epochs 5

# Run 2 (should produce identical output on same hardware)
python exp_drivaer_ml_distributed.py --data_dir /path --seed 42 --epochs 5
```

On a single GPU, the train loss at each epoch should be **bitwise identical**. On multi-GPU, losses should match to ~6 decimal places (NCCL rounding).
