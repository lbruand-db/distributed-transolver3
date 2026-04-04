# Mesh-Sharded Distributed Transolver-3

[![CI](https://github.com/lbruand-db/distributed-transolver3/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand-db/distributed-transolver3/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type--checked-ty-blue)](https://github.com/astral-sh/ty)
[![pytest](https://img.shields.io/badge/tests-pytest-green)](https://docs.pytest.org/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet?logo=anthropic)](https://claude.ai/code)

Scaling Transformer Solvers to Industrial-Scale Geometries (100M+ cells).

Based on the [Transolver paper](https://arxiv.org/abs/2402.02366) (ICML 2024 Spotlight) and the [Transolver-3 paper](https://arxiv.org/pdf/2602.04940).

## Key Innovations

1. **Faster Slice & Deslice** — Linear projections moved from O(N) mesh domain to O(M) slice domain via matrix multiplication associativity
2. **Geometry Slice Tiling** — Input partitioned into tiles with gradient checkpointing, reducing peak memory from O(NM) to O(N_t*M)
3. **Geometry Amortized Training** — Train on random subsets (100K-400K) of full mesh each iteration
4. **Physical State Caching** — Two-phase inference: build cache from chunks, decode any point
5. **Mixed Precision** — Full autocast + GradScaler support, halving memory footprint
6. **Mesh-Sharded Distribution** — Shard meshes >100 GB across GPUs; all-reduce only the tiny slice accumulators (~514 KB/layer)

<p align="center">
<img src="./experiments/results/drivaer_pressure_sideview.png" height="200" alt="DrivAerML pressure comparison" align=center />
</p>

## Setup on Databricks

### Notebook (first cell)

```python
%pip install /Workspace/Repos/<user>/Transolver -q
dbutils.library.restartPython()
```

### DAB deployment

```bash
databricks bundle deploy -t a10g       # 4x A10G (96 GB) — default
databricks bundle deploy -t a100_40    # 8x A100 40GB
databricks bundle deploy -t a100_80    # 8x A100 80GB
```

## Get Started

All code runs in **Databricks notebooks** on GPU clusters. Data lives in **UC Volumes**.

```python
# Cell 1: Setup
%pip install /Workspace/Repos/<user>/Transolver -q
dbutils.library.restartPython()
```

```python
# Cell 2: Imports + model creation
import mlflow
import torch
import numpy as np
from transolver3 import Transolver3, CachedInference
from transolver3.amortized_training import train_step, create_optimizer, create_scheduler, AmortizedMeshSampler

device = torch.device("cuda")
model = Transolver3(
    space_dim=3, n_layers=24, n_hidden=256, n_head=8,
    fun_dim=0, out_dim=4, slice_num=64,
    tile_size=100_000,       # auto-compute tiles (paper Table 5)
    mlp_chunk_size=100_000,  # tile MLP for large N
).to(device)
```

```python
# Cell 3: Load data from UC Volumes
VOLUME_PATH = "/Volumes/ml/transolver3/data"
data = np.load(f"{VOLUME_PATH}/drivaer_001.npz", mmap_mode='r')
x = torch.tensor(data["coordinates"], dtype=torch.float32).unsqueeze(0).to(device)
y = torch.tensor(data["targets"], dtype=torch.float32).unsqueeze(0).to(device)
```

```python
# Cell 4: Training with mixed precision + MLflow tracking
optimizer = create_optimizer(model)
scheduler = create_scheduler(optimizer, total_steps=50000)
scaler = torch.amp.GradScaler()
sampler = AmortizedMeshSampler(subset_size=100_000, seed=42)

mlflow.set_experiment("/Shared/transolver3-experiments")

with mlflow.start_run(run_name="drivaer-ml-24L-256H"):
    mlflow.log_params({
        "n_layers": 24, "n_hidden": 256, "n_head": 8,
        "slice_num": 64, "tile_size": 100_000,
    })

    for step in range(50000):
        loss = train_step(model, x, None, y, optimizer, scheduler,
                          sampler=sampler, tile_size=100_000, scaler=scaler)
        if step % 100 == 0:
            mlflow.log_metric("train_loss", loss, step=step)
```

```python
# Cell 5: Inference on industrial-scale meshes (160M+ cells)
model.eval()
engine = CachedInference(model, cache_chunk_size=100_000, decode_chunk_size=50_000)
output = engine.predict(x_full_mesh)
```

## Claude Skills (Newcomer Guide)

Four Claude Code skills in `skills/` provide step-by-step guidance for newcomers. All skills target **Databricks notebooks** and **DABs** — no local setup required.

| Skill | Purpose |
|-------|---------|
| **[transolver-data](skills/transolver-data.md)** | Load, inspect, validate `.npz` meshes in UC Volumes; normalization; memory estimation |
| **[transolver-run](skills/transolver-run.md)** | Config presets (small/medium/large), training in notebooks, 3-phase pipeline, TorchDistributor, DAB workflows |
| **[transolver-analyze](skills/transolver-analyze.md)** | Loss interpretation, per-channel error stats, physical bounds checking, PSI drift detection, GPU profiling |
| **[transolver-deploy](skills/transolver-deploy.md)** | MLflow tracking, UC model registration, serving endpoints, inference table monitoring, end-to-end checklist |

## File Structure

```
transolver3/                          # Core package
├── physics_attention_v3.py           # Optimized Physics-Attention
├── transolver3_block.py              # Encoder block with tiled MLP
├── model.py                          # Transolver3 model
├── amortized_training.py             # Training (sampler, loss, scheduler, train_step)
├── inference.py                      # CachedInference + DistributedCachedInference
├── distributed.py                    # Multi-GPU mesh sharding utilities
├── normalizer.py                     # InputNormalizer, TargetNormalizer
├── profiling.py                      # Memory/latency benchmarking
├── serving.py                        # MLflow pyfunc wrapper for Model Serving
├── mlflow_utils.py                   # Experiment tracking + model logging
├── data_catalog.py                   # Delta Lake mesh metadata integration
├── databricks_training.py            # TorchDistributor launcher + Spark preprocessing
├── monitoring.py                     # Bounds checking + PSI drift detection
└── common.py                         # MLP, activations, timestep_embedding

resources/                            # DAB job definitions
├── training_workflow.yml             # 5-task pipeline (preprocess → train → evaluate → register → deploy)
├── serving_endpoint.yml              # Model Serving endpoint config
├── gpu_benchmark_job.yml             # Single-GPU memory benchmark
├── distributed_test_job.yml          # 2-GPU mesh-sharded test
├── test_mlflow_auth_job.yml          # Smoke test MLflow auth in TorchDistributor children
├── test_register_job.yml             # Serverless checkpoint inspection test
└── test_register_deploy_job.yml      # Serverless register + deploy test

scripts/                              # Entry points for DAB tasks
├── preprocess.py                     # Register mesh metadata + compute stats
├── register_model.py                 # Promote model from MLflow run to UC registry
├── deploy_endpoint.py                # Deploy to Databricks serving endpoint
├── test_mlflow_auth.py               # MLflow auth propagation smoke test
└── test_register.py                  # Checkpoint inspection + model load test

skills/                               # Claude Code skills for newcomers
├── transolver-data.md                # Mesh data management
├── transolver-run.md                 # Training & simulation
├── transolver-analyze.md             # Results analysis & drift
└── transolver-deploy.md              # Databricks deployment lifecycle

Industrial-Scale-Benchmarks/          # Experiments
├── exp_nasa_crm.py                   # NASA-CRM (~400K cells)
├── exp_ahmed_ml.py                   # AhmedML (~20M cells)
├── exp_drivaer_ml.py                 # DrivAerML (~160M cells, single GPU)
├── exp_drivaer_ml_distributed.py     # DrivAerML distributed (multi-GPU)
├── dataset/                          # Dataset loaders (with mesh sharding)
└── utils/metrics.py                  # Evaluation metrics

experiments/                          # v1 vs v3 comparison
├── compare_v1_v3_drivaer.py          # Synthetic data comparison
├── compare_v1_v3_real_drivaer.py     # Real DrivAerML VTP data comparison
├── COMPARE_v1v3.md                   # Results and analysis
└── results/                          # Pressure heatmap PNGs

benchmarks/                           # GPU benchmarking
├── gpu_memory_benchmark.py           # Sweep mesh sizes, measure all 3 phases
└── test_sharded_distributed.py       # Distributed sharding validation test

SPECS/                                # Design documentation
├── SPEC.md                           # Core v3 architecture specification
├── DISTRIBUTED.md                    # Multi-GPU distribution design
├── DISTRIBUTED_ARCHITECTURE.md       # Mermaid diagrams: pipeline, process model, data flow
├── CRITICAL_ISSUES.md                # Known issues & fixes
├── DIFFERENTIATORS.md                # Why Databricks is ideal
└── VALUEADDED.md                     # Databricks integration roadmap

tests/                                # 100 tests
├── test_transolver3.py               # Core model tests (41)
├── test_distributed.py               # Distributed sharding tests (11)
├── test_serving.py                   # Serving tests (4)
├── test_monitoring.py                # Monitoring tests (5)
├── test_data_catalog.py              # Catalog tests (6)
├── test_mlflow_utils.py              # MLflow tests (4)
└── test_databricks_training.py       # Training integration + auth propagation tests (12)
```

## Memory Scaling

<p align="center">
<img src="./memory_scaling.png" height="250" alt="" align=center />
</p>

With tile_size=100K and fp16, the paper's claim of **2.9M cells on a single A100 80GB** is achievable (~14 GB activations).

## Profiling (notebook on GPU cluster)

```python
from transolver3.profiling import benchmark_scaling, format_benchmark_table

results = benchmark_scaling(model, mesh_sizes=[1000, 10000, 100000],
    configs=[
        {'label': 'no_tiling', 'num_tiles': 0},
        {'label': 'tile_100k', 'tile_size': 100_000},
    ])
print(format_benchmark_table(results))
```

## Distributed Training (Huge Meshes >100 GB)

For meshes that exceed single-node memory, Transolver-3 supports mesh-sharded
distribution across multiple GPUs. Each GPU loads only 1/K of the mesh via mmap
range reads. The key insight: the slice accumulators `s_raw (B,H,M,C)` are
**additive** — they can be independently computed from disjoint mesh partitions
and all-reduced (~514 KB per layer).

### Via DAB Training Pipeline

The full pipeline runs 5 sequential tasks, each on its own cluster. MLflow is the single source of truth for model artifacts — no checkpoint files are passed between tasks.

```bash
databricks bundle deploy -t a10g
databricks bundle run transolver3_training_pipeline
```

| Task | Cluster | What it does |
|------|---------|------|
| **preprocess** | i3.xlarge (CPU) | Register mesh metadata + compute stats in Delta |
| **train** | g5.12xlarge (4x A10G) | Mesh-sharded DDP training via TorchDistributor, live MLflow metrics |
| **evaluate** | g5.12xlarge (4x A10G) | Load model from MLflow run, run cached inference on test set |
| **register** | i3.xlarge (CPU) | Promote already-logged model to UC Model Registry |
| **deploy** | i3.xlarge (CPU) | Create/update Model Serving endpoint with scale-to-zero |

The train task uses `TorchDistributor(local_mode=True)` to launch torchrun on a single multi-GPU node. Each GPU loads a disjoint 1/K shard of the mesh via mmap range reads. Gradients are all-reduced via NCCL. See [SPECS/DISTRIBUTED_ARCHITECTURE.md](SPECS/DISTRIBUTED_ARCHITECTURE.md) for Mermaid diagrams of the full architecture.

### Other DAB jobs

```bash
databricks bundle run gpu_memory_benchmark          # Single-GPU memory sweep
databricks bundle run distributed_sharded_test      # 2-GPU validation
databricks bundle run test_mlflow_auth              # Smoke test MLflow auth in child processes
databricks bundle run test_register_deploy          # Serverless register + deploy (fast iteration)
```

Validated on 4x NVIDIA A10G: sharded cache and decode produce **zero numerical difference** vs single-GPU. See [SPECS/DISTRIBUTED.md](SPECS/DISTRIBUTED.md) for the original design.

## GPU Benchmark (DAB)

Three DAB targets map to different GPU instances:

| Target | Instance | GPU | VRAM | Use case |
|--------|----------|-----|------|----------|
| `a10g` (default) | `g5.12xlarge` | 4x NVIDIA A10G | 96 GB | Multi-GPU training, benchmarks |
| `a100_40` | `p4d.24xlarge` | 8x NVIDIA A100 | 320 GB | Large-scale training |
| `a100_80` | `p4de.24xlarge` | 8x NVIDIA A100 | 640 GB | Full-scale DrivAerML |

```bash
databricks bundle deploy -t a10g
databricks bundle run gpu_memory_benchmark          # Single-GPU memory sweep
databricks bundle run distributed_sharded_test      # 2-GPU validation
databricks bundle run training_workflow             # Full 5-task pipeline
```

The benchmark sweeps mesh sizes and measures peak GPU memory across all 3 pipeline phases (training, cache build, decode) using synthetic DrivAer ML data.

## v1 vs v3 Comparison

Includes experiments comparing Transolver v1 and v3 on both synthetic and real DrivAerML data. See [experiments/COMPARE_v1v3.md](experiments/COMPARE_v1v3.md) for full results and pressure heatmaps on the real DrivAer vehicle.

<p align="center">
<img src="./experiments/results/drivaer_pressure_sideview.png" height="200" alt="DrivAerML pressure comparison" align=center />
</p>

## Citation

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@article{wu2026Transolver3,
  title={Transolver++: Industrial-Scale Simulation with Transformer Solvers},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  journal={arXiv preprint arXiv:2602.04940},
  year={2026}
}
```
