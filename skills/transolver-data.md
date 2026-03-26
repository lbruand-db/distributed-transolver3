# transolver-data

Mesh data loading, inspection, validation, and normalization for Transolver-3 on Databricks.

---

## When to use

Use this skill when the user wants to:
- Load, inspect, or validate `.npz` mesh files stored in UC Volumes
- Understand mesh dimensions, point counts, or memory requirements
- Subsample a large mesh for quick experimentation
- Compute or inspect normalization statistics (InputNormalizer, TargetNormalizer)
- Register mesh metadata in the Unity Catalog data catalog
- Debug data format issues (wrong shapes, NaN/Inf, missing keys)

---

## Data format reference

Transolver-3 uses NumPy `.npz` files loaded via `np.load(mmap_mode='r')` for memory-mapped I/O.

**Expected array keys** (convention from DrivAerML):
- `coordinates` or `x`: `(N, space_dim)` — mesh point coordinates (typically 3D)
- `features` or `fx`: `(N, fun_dim)` — input features at each point (optional)
- `targets` or `y`: `(N, out_dim)` — ground truth outputs (pressure, velocity, etc.)

A single DrivAerML sample is ~12 GB (140M points x 22 features x 4 bytes float32).

**Shape conventions**:
- Training tensors: `(B, N, D)` where B=batch, N=points, D=channels
- Single sample: `(N, D)` — gets unsqueezed to `(1, N, D)` internally

**Storage**: All mesh files live in UC Volumes at `/Volumes/<catalog>/<schema>/data/`.

---

## Prerequisites (notebook cell 1)

Run this in the first cell of any Databricks notebook working with Transolver-3:

```python
# Install transolver3 on the cluster
%pip install /Workspace/Repos/<user>/Transolver -q
dbutils.library.restartPython()
```

Or if the package is published to a Volume:

```python
%pip install /Volumes/ml/transolver3/wheels/transolver3-0.1.0-py3-none-any.whl -q
dbutils.library.restartPython()
```

---

## Common operations

### 1. Upload mesh files to UC Volumes

```python
# Create the volume if it doesn't exist
spark.sql("CREATE VOLUME IF NOT EXISTS ml.transolver3.data")

# Upload via dbutils (from driver local storage or workspace files)
dbutils.fs.cp(
    "file:/tmp/drivaer_001.npz",
    "/Volumes/ml/transolver3/data/drivaer_001.npz"
)

# List available meshes
display(dbutils.fs.ls("/Volumes/ml/transolver3/data/"))
```

### 2. Inspect a mesh file

```python
import numpy as np

VOLUME_PATH = "/Volumes/ml/transolver3/data"
data = np.load(f"{VOLUME_PATH}/drivaer_001.npz", mmap_mode='r')

# List all arrays and their shapes
for key in data.files:
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
          f"size={arr.nbytes / 1024**2:.1f} MB")

# Total file memory footprint
total_mb = sum(data[k].nbytes for k in data.files) / 1024**2
print(f"Total: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
```

### 3. Validate mesh data

```python
import numpy as np

VOLUME_PATH = "/Volumes/ml/transolver3/data"
data = np.load(f"{VOLUME_PATH}/drivaer_001.npz", mmap_mode='r')

def validate_mesh(data, coord_key="coordinates", target_key="targets"):
    """Check mesh data for common issues."""
    issues = []

    if coord_key not in data.files:
        issues.append(f"Missing '{coord_key}' array")
        return issues

    coords = data[coord_key]

    if coords.ndim != 2:
        issues.append(f"Coordinates should be 2D (N, D), got shape {coords.shape}")
    elif coords.shape[1] not in (2, 3):
        issues.append(f"Unexpected space_dim={coords.shape[1]} (expected 2 or 3)")

    # Check a sample for large meshes
    sample = coords[:min(100000, len(coords))]
    if np.any(np.isnan(sample)):
        issues.append("Coordinates contain NaN values")
    if np.any(np.isinf(sample)):
        issues.append("Coordinates contain Inf values")

    if target_key in data.files:
        targets = data[target_key]
        if targets.shape[0] != coords.shape[0]:
            issues.append(f"Point count mismatch: coords={coords.shape[0]}, targets={targets.shape[0]}")
        t_sample = targets[:min(100000, len(targets))]
        if np.any(np.isnan(t_sample)):
            issues.append("Targets contain NaN values")

    if not issues:
        issues.append("All checks passed")
    return issues

for issue in validate_mesh(data):
    print(f"  {issue}")
```

### 4. Subsample a mesh for quick experiments

```python
import numpy as np
import torch

VOLUME_PATH = "/Volumes/ml/transolver3/data"
data = np.load(f"{VOLUME_PATH}/drivaer_001.npz", mmap_mode='r')
coords = data["coordinates"]  # (N, 3)
targets = data["targets"]     # (N, out_dim)

N = coords.shape[0]
subset_size = 100_000  # 100K points for quick testing

rng = np.random.default_rng(seed=42)
indices = rng.choice(N, size=min(subset_size, N), replace=False)
indices.sort()  # Sorted for better mmap locality

x_sub = torch.tensor(coords[indices], dtype=torch.float32).unsqueeze(0)   # (1, n, 3)
y_sub = torch.tensor(targets[indices], dtype=torch.float32).unsqueeze(0)  # (1, n, out_dim)
print(f"Subsampled {subset_size:,} / {N:,} points ({100*subset_size/N:.1f}%)")
```

### 5. Compute normalization statistics

```python
import torch
from transolver3.normalizer import InputNormalizer, TargetNormalizer

# --- Input normalization (coordinates) ---
# Per-sample mode (default, no fitting needed):
input_norm = InputNormalizer(scale=1000.0, per_sample=True)
x_normalized = input_norm.encode(x_sub)  # (B, N, 3) -> [0, 1000]

# Dataset-level mode (fit once, reuse):
input_norm = InputNormalizer(scale=1000.0, per_sample=False)
input_norm.fit(x_sub)

# --- Target normalization (outputs) ---
target_norm = TargetNormalizer(out_dim=y_sub.shape[-1])
target_norm.fit(y_sub)
print(f"Target mean: {target_norm.mean.squeeze().tolist()}")
print(f"Target std:  {target_norm.std.squeeze().tolist()}")

# For large datasets, use streaming fit:
# target_norm.fit_incremental(target_iter)

# Encode for training, decode for evaluation:
y_encoded = target_norm.encode(y_sub)
y_decoded = target_norm.decode(predictions)
```

### 6. Register mesh metadata in Unity Catalog

```python
from transolver3.data_catalog import register_mesh_metadata, get_mesh_metadata

# Register a mesh dataset in the catalog
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

# Query all registered meshes
meshes = get_mesh_metadata(spark, catalog="ml", schema="transolver3")
display(meshes)
```

### 7. Preprocess with Spark (large-scale normalization stats)

```python
from transolver3.databricks_training import preprocess_with_spark

# Compute normalization stats across many mesh files using Spark
preprocess_with_spark(
    spark=spark,
    data_dir="/Volumes/ml/transolver3/data",
    catalog="ml",
    schema="transolver3",
)
# Writes normalization stats to Delta table: ml.transolver3.normalization_stats
```

### 8. Estimate GPU memory requirements

```python
N = 140_000_000  # 140M points (DrivAerML scale)
space_dim = 3
fun_dim = 22
out_dim = 4
bytes_per_float = 4

# Raw data memory
data_gb = N * (space_dim + fun_dim + out_dim) * bytes_per_float / 1024**3
print(f"Raw data: {data_gb:.1f} GB")

# With Transolver-3 cached inference (cache is only ~768 KB for 24 layers):
chunk_size = 100_000
chunk_gb = chunk_size * (space_dim + fun_dim) * bytes_per_float / 1024**3
print(f"Per-chunk working memory: {chunk_gb*1000:.1f} MB")
print(f"Cache size (24 layers): ~768 KB")
print(f"Total chunks needed: {N // chunk_size}")
```

### 9. Load data using AmortizedMeshSampler (training)

```python
from transolver3.amortized_training import AmortizedMeshSampler

sampler = AmortizedMeshSampler(subset_size=200_000, seed=42)

# In training loop:
N_total = x.shape[1]
indices = sampler.sample(N_total)  # Different random subset each call
x_sub = x[:, indices]
y_sub = y[:, indices]
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError` on Volume path | Volume not created or wrong path | Run `spark.sql("CREATE VOLUME IF NOT EXISTS ml.transolver3.data")` |
| `MemoryError` loading `.npz` | File loaded without mmap | Use `np.load(path, mmap_mode='r')` |
| Shape `(N,)` instead of `(N, D)` | Scalar output field | Reshape: `targets = targets.reshape(-1, 1)` |
| Very large coordinate ranges | Not normalized | Apply `InputNormalizer(scale=1000.0)` |
| Loss is NaN from step 1 | NaN in data or extreme target values | Run validation, check `TargetNormalizer` stats |
| `KeyError` on load | Different array key names | Check `data.files` for actual key names |
| Slow file reads from Volume | No mmap or large sequential reads | Use `mmap_mode='r'` and sorted index access |
