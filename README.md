# Transolver-3

Scaling Transformer Solvers to Industrial-Scale Geometries (100M+ cells).

Based on [Transolver](https://arxiv.org/abs/2402.02366) (ICML 2024 Spotlight) and the [Transolver-3 paper](https://arxiv.org/pdf/2602.04940).

## Key Innovations

1. **Faster Slice & Deslice** — Linear projections moved from O(N) mesh domain to O(M) slice domain via matrix multiplication associativity
2. **Geometry Slice Tiling** — Input partitioned into tiles with gradient checkpointing, reducing peak memory from O(NM) to O(N_t*M)
3. **Geometry Amortized Training** — Train on random subsets (100K-400K) of full mesh each iteration
4. **Physical State Caching** — Two-phase inference: build cache from chunks, decode any point
5. **Mixed Precision** — Full autocast + GradScaler support, halving memory footprint

<p align="center">
<img src="./pic/Transolver.png" height="250" alt="" align=center />
</p>

## Get Started

```python
from transolver3 import Transolver3, CachedInference, InputNormalizer, TargetNormalizer

# Create model
model = Transolver3(
    space_dim=3, n_layers=24, n_hidden=256, n_head=8,
    fun_dim=0, out_dim=4, slice_num=64,
    tile_size=100_000,       # auto-compute tiles (paper Table 5)
    mlp_chunk_size=100_000,  # tile MLP for large N
)

# Training with mixed precision
from transolver3.amortized_training import train_step, create_optimizer, create_scheduler

optimizer = create_optimizer(model)
scheduler = create_scheduler(optimizer, total_steps=50000)
scaler = torch.amp.GradScaler()

loss = train_step(model, x, fx, target, optimizer, scheduler,
                  tile_size=100_000, scaler=scaler)

# Inference on industrial-scale meshes (160M+ cells)
engine = CachedInference(model, cache_chunk_size=100_000, decode_chunk_size=50_000)
output = engine.predict(x_full_mesh)
```

## File Structure

```
transolver3/                          # Core package
├── physics_attention_v3.py           # Optimized Physics-Attention
├── transolver3_block.py              # Encoder block with tiled MLP
├── model.py                          # Transolver3 model
├── amortized_training.py             # Training (sampler, loss, scheduler, train_step)
├── inference.py                      # CachedInference for industrial-scale
├── normalizer.py                     # InputNormalizer, TargetNormalizer
├── profiling.py                      # Memory/latency benchmarking
└── common.py                         # MLP, activations, timestep_embedding

Industrial-Scale-Benchmarks/          # Experiments
├── exp_nasa_crm.py                   # NASA-CRM (~400K cells)
├── exp_ahmed_ml.py                   # AhmedML (~20M cells)
├── exp_drivaer_ml.py                 # DrivAerML (~160M cells)
├── dataset/                          # Dataset loaders
└── utils/metrics.py                  # Evaluation metrics

tests/test_transolver3.py             # 41 tests
```

## Memory Scaling

<p align="center">
<img src="./memory_scaling.png" height="250" alt="" align=center />
</p>

With tile_size=100K and fp16, the paper's claim of **2.9M cells on a single A100 80GB** is achievable (~14 GB activations).

## Profiling

```python
from transolver3.profiling import benchmark_scaling, format_benchmark_table

results = benchmark_scaling(model, mesh_sizes=[1000, 10000, 100000],
    configs=[
        {'label': 'no_tiling', 'num_tiles': 0},
        {'label': 'tile_100k', 'tile_size': 100_000},
    ])
print(format_benchmark_table(results))
```

## Citation

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

For Transolver v1 code (standard benchmarks, airfoil/car design tasks), see the [upstream repo](https://github.com/thuml/Transolver).
