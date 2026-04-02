# Migration Guide: Transolver v1 to v3

For teams running Transolver v1 (original ICML 2024 paper) who want to move to the distributed v3 implementation on Databricks.

## Architecture Changes

| Aspect | v1 | v3 |
|--------|----|----|
| Attention projection | O(N) mesh domain | O(M) slice domain (associativity trick) |
| Memory scaling | O(N*M) per layer | O(N_t*M) with tiling |
| Training | Full mesh per GPU | Amortized: random 100K-400K subset |
| Inference | Single forward pass | Two-phase: cache + decode |
| Distribution | Not supported | Mesh-sharded DDP across K GPUs |
| Max mesh size | ~2M points (A100 80GB) | 100M+ points (multi-GPU) |

## Data Format Changes

**v1**: Expects pre-processed tensors, typically loaded via custom DataLoader.

**v3**: Expects `.npz` files with standardized keys:

```
Surface fields:
  surface_coords:      (N_s, 3)  — xyz coordinates
  surface_normals:     (N_s, 3)  — surface normals
  surface_pressure:    (N_s, 1)  — pressure
  surface_wall_shear:  (N_s, 3)  — wall shear stress

Volume fields:
  volume_coords:       (N_v, 3)  — xyz coordinates
  volume_velocity:     (N_v, 3)  — velocity
  volume_pressure:     (N_v, 1)  — pressure

Metadata:
  params:              (d,)      — geometric deformation parameters
```

**Converting v1 data**: If your v1 data is in `.pt`, `.h5`, or custom format:

```python
import numpy as np

# Load your v1 data
coords = ...  # (N, 3)
normals = ...  # (N, 3)
pressure = ...  # (N, 1)
shear = ...  # (N, 3)
params = ...  # (d,)

# Save as v3 .npz
np.savez_compressed("sample_001.npz",
    surface_coords=coords,
    surface_normals=normals,
    surface_pressure=pressure,
    surface_wall_shear=shear,
    params=params,
)
```

Then create split files (`train.txt`, `test.txt`) listing the `.npz` filenames.

## Model Configuration Changes

**v1 model creation**:
```python
model = Transolver(
    space_dim=3,
    n_layers=12,
    n_hidden=128,
    n_head=8,
    slice_num=32,
)
```

**v3 model creation** (same API, new defaults for industrial scale):
```python
model = Transolver3(
    space_dim=3,       # or 22 if concatenating coords+normals+params
    n_layers=24,       # paper: 24 layers
    n_hidden=256,      # paper: 256
    n_head=8,
    slice_num=64,      # paper: 64 slices
    fun_dim=0,         # no auxiliary features
    out_dim=4,         # pressure + 3 shear
    mlp_ratio=1,
    num_tiles=8,       # tiling for memory efficiency
)
```

Key difference: v3's `space_dim` includes all input features (coords + normals + params), not just spatial coordinates. The DrivAerML surface field uses `space_dim=22` (3 coords + 3 normals + 16 params).

## Training Loop Changes

**v1** (typical):
```python
for step in range(total_steps):
    x, y = next(dataloader)
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

**v3** (amortized + distributed):
```python
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss,
    create_optimizer, create_scheduler,
)

sampler = AmortizedMeshSampler(subset_size=100_000, seed=42)
optimizer = create_optimizer(model, lr=1e-3)
scheduler = create_scheduler(optimizer, total_steps)

for epoch in range(epochs):
    for batch in dataloader:
        indices = sampler.sample(N)
        x_sub = x[:, indices]
        pred = model(x_sub, num_tiles=8)
        loss = relative_l2_loss(pred, target[:, indices])
        loss.backward()
        optimizer.step()
        scheduler.step()
```

Key differences:
- **Amortized subsampling**: don't train on full mesh, sample 100K-400K points per iteration
- **Tiled forward**: `num_tiles=8` partitions attention to reduce peak memory
- **Cosine LR schedule**: with 5% linear warmup (via `create_scheduler`)
- **Relative L2 loss**: normalized by target norm, not MSE

## Inference Changes

**v1**: Single forward pass.

**v3**: Two-phase cached inference for large meshes:
```python
from transolver3 import CachedInference

engine = CachedInference(model, cache_chunk_size=100_000, decode_chunk_size=50_000)
output = engine.predict(x_full)  # handles chunking internally
```

For meshes that fit in memory, the standard forward pass still works:
```python
output = model(x)  # same as v1
```

## Deployment on Databricks

v1 on Azure ML typically involves:
1. Register model in Azure ML Registry
2. Deploy to Azure ML Managed Endpoint

v3 on Databricks:
1. Data in UC Volumes (`.npz` files)
2. Train via DAB pipeline: `databricks bundle run transolver3_training_pipeline`
3. Model logged to MLflow, promoted to UC Model Registry
4. Served via Databricks Model Serving (scale-to-zero)

The DAB pipeline handles all 5 steps automatically. See `SPECS/DISTRIBUTED_ARCHITECTURE.md` for the full architecture.

## Checklist

- [ ] Convert data to `.npz` format with standardized keys
- [ ] Create `train.txt` / `test.txt` split files
- [ ] Upload to UC Volume
- [ ] Update `databricks.yml` with your workspace host and catalog
- [ ] Adjust model config (space_dim, out_dim for your field)
- [ ] Run `databricks bundle deploy -t <target>`
- [ ] Run `databricks bundle run transolver3_training_pipeline`
- [ ] Check MLflow experiment at `/Shared/transolver3-experiments`
