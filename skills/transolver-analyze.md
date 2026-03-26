# transolver-analyze

Interpret Transolver-3 results on Databricks — training metrics, prediction quality, physical bounds checking, and drift detection.

---

## When to use

Use this skill when the user wants to:
- Understand training loss curves and what "good" looks like
- Compare predictions against ground truth
- Check if predictions are physically plausible (bounds checking)
- Detect distribution drift between training and inference
- Profile GPU memory usage and latency on a Databricks GPU cluster
- Debug poor model performance
- Understand error metrics and what they mean for CFD

---

## Notebook setup (first cell)

```python
%pip install /Workspace/Repos/<user>/Transolver -q
dbutils.library.restartPython()
```

---

## Training loss interpretation

### Relative L2 loss

Transolver-3 uses **relative L2 loss** (paper Eq. 7):

```
L2_rel = ||pred - target||_2 / ||target||_2
```

This is scale-invariant: a loss of 0.01 means 1% relative error regardless of the physical units.

**What "good" looks like:**

| Loss value | Interpretation |
|-----------|----------------|
| > 1.0 | Model not learning (check data, LR, normalization) |
| 0.1 - 1.0 | Early training, model is learning |
| 0.01 - 0.1 | Reasonable accuracy for many applications |
| < 0.01 | Good accuracy (paper reports ~0.005 on DrivAerML) |
| < 0.001 | Excellent — diminishing returns beyond this |

**Healthy training curve characteristics:**
- Loss should drop sharply in first 10-20% of training (warmup + initial learning)
- Gradual decrease through middle of training (cosine schedule)
- Plateaus near the end — this is expected with cosine decay to `min_lr`
- Small fluctuations are normal with amortized sampling (different subsets each step)

### Review training metrics in MLflow

```python
import mlflow

# Find your experiment
experiment = mlflow.get_experiment_by_name("/Shared/transolver3-experiments")

# Load the best run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.eval_rel_l2 ASC"],
    max_results=5,
)
display(runs[["run_id", "params.n_layers", "params.total_steps",
              "metrics.train_loss", "metrics.eval_rel_l2"]])
```

### Plot training loss from MLflow

```python
import mlflow
import matplotlib.pyplot as plt

run_id = "<your_run_id>"
client = mlflow.tracking.MlflowClient()
history = client.get_metric_history(run_id, "train_loss")

steps = [m.step for m in history]
losses = [m.value for m in history]

plt.figure(figsize=(10, 4))
plt.semilogy(steps, losses)
plt.xlabel("Step")
plt.ylabel("Relative L2 Loss")
plt.title("Training Loss Curve")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Prediction quality analysis

### Per-channel error statistics

```python
import torch
from transolver3.amortized_training import relative_l2_loss

def analyze_predictions(pred, target, channel_names=None):
    """Compute per-channel error statistics.

    Args:
        pred: (B, N, out_dim) predictions
        target: (B, N, out_dim) ground truth
        channel_names: optional list of names like ["pressure", "vx", "vy", "vz"]
    """
    out_dim = pred.shape[-1]
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(out_dim)]

    rows = []
    for ch in range(out_dim):
        p = pred[..., ch]
        t = target[..., ch]
        err = (p - t).abs()
        rows.append({
            "channel": channel_names[ch],
            "MAE": err.mean().item(),
            "max_error": err.max().item(),
            "RMSE": ((p - t) ** 2).mean().sqrt().item(),
            "rel_L2": (torch.norm(p - t) / (torch.norm(t) + 1e-8)).item(),
            "pred_min": p.min().item(),
            "pred_max": p.max().item(),
            "target_min": t.min().item(),
            "target_max": t.max().item(),
        })

    # Display as Databricks table
    import pandas as pd
    display(pd.DataFrame(rows))

# Usage:
# analyze_predictions(pred, target, ["pressure", "vx", "vy", "vz"])
```

### Spatial error distribution

```python
def find_high_error_regions(pred, target, x_coords, percentile=99):
    """Find mesh regions with highest prediction errors.

    Useful for identifying where the model struggles — often sharp
    geometrical features, boundary layers, or flow separation zones.
    """
    err = (pred - target).norm(dim=-1)  # (B, N) per-point error
    threshold = torch.quantile(err.flatten(), percentile / 100)
    high_err_mask = err > threshold

    high_err_coords = x_coords[high_err_mask.squeeze()]
    print(f"Points above {percentile}th percentile error: {high_err_mask.sum().item()}")
    print(f"Error threshold: {threshold.item():.6f}")
    print(f"Spatial extent of high-error region:")
    print(f"  x: [{high_err_coords[:, 0].min():.3f}, {high_err_coords[:, 0].max():.3f}]")
    print(f"  y: [{high_err_coords[:, 1].min():.3f}, {high_err_coords[:, 1].max():.3f}]")
    if high_err_coords.shape[1] > 2:
        print(f"  z: [{high_err_coords[:, 2].min():.3f}, {high_err_coords[:, 2].max():.3f}]")

    return high_err_coords, high_err_mask
```

---

## Physical bounds checking

Validate that predictions fall within physically plausible ranges. Run this in a notebook cell after inference.

```python
from transolver3.monitoring import check_prediction_bounds

# Define physical bounds for your simulation
# Example: automotive CFD (DrivAerML)
physical_bounds = {
    0: (-50_000, 100_000),  # Pressure (Pa): ~[-50kPa, 100kPa]
    1: (-100, 400),          # Velocity x (m/s)
    2: (-100, 100),          # Velocity y (m/s)
    3: (-100, 100),          # Velocity z (m/s)
}

result = check_prediction_bounds(predictions, physical_bounds)

print(f"All predictions valid: {result['all_valid']}")
print(f"Total out-of-bounds: {result['total_out_of_bounds']}")

for ch, info in result["channels"].items():
    if info["out_of_bounds_count"] > 0:
        print(f"\n  Channel {ch}:")
        print(f"    Out of bounds: {info['out_of_bounds_count']} "
              f"({info['out_of_bounds_fraction']*100:.3f}%)")
        print(f"    Below min: {info['below_min']}, Above max: {info['above_max']}")
        print(f"    Pred range: [{info['min_value']:.2f}, {info['max_value']:.2f}]")
```

**Common physical bounds by domain:**

| Domain | Channel | Typical range |
|--------|---------|---------------|
| Automotive external aero | Pressure (Pa) | -50,000 to 100,000 |
| Automotive external aero | Velocity (m/s) | -100 to 400 |
| HVAC / building | Temperature (K) | 250 to 350 |
| HVAC / building | Pressure (Pa) | -1,000 to 1,000 |
| Turbomachinery | Pressure (Pa) | -200,000 to 500,000 |

---

## Drift detection

Monitor if inference-time data distributions have shifted from training. All drift data is persisted to Delta tables.

### Compute and log PSI to Delta

```python
from transolver3.monitoring import log_drift_metrics

# baseline_stats come from training data normalization:
baseline_stats = {
    "mean": target_norm.mean.squeeze().tolist(),
    "std": target_norm.std.squeeze().tolist(),
}

# Log drift metrics to Delta (spark is available in Databricks notebooks)
drift_results = log_drift_metrics(
    spark=spark,
    catalog="ml",
    schema="transolver3",
    predictions=predictions,
    baseline_stats=baseline_stats,
)

for ch, info in drift_results.items():
    print(f"Channel {ch}: PSI={info['psi']:.4f} ({info['drift']} drift)")
    print(f"  Baseline: mean={info['baseline_mean']:.4f}, std={info['baseline_std']:.4f}")
    print(f"  Current:  mean={info['current_mean']:.4f}, std={info['current_std']:.4f}")
```

### Query drift history from Delta

```python
# Review drift trends over time
drift_df = spark.read.table("ml.transolver3.drift_metrics")
display(drift_df.orderBy("logged_at", ascending=False))

# Find high-drift events
high_drift = drift_df.filter("drift_flag = 'high'")
display(high_drift)
```

### PSI interpretation

| PSI | Drift level | Action |
|-----|-------------|--------|
| < 0.1 | Low | No action needed |
| 0.1 - 0.2 | Medium | Monitor closely, investigate if persistent |
| > 0.2 | High | Retrain or investigate data pipeline changes |

**Common causes of drift in CFD:**
- New geometry variants not seen in training (different car body, building shape)
- Changed simulation boundary conditions (different inlet velocity, temperature)
- Different mesh resolution than training data
- Coordinate system or units changed upstream

---

## GPU memory and latency profiling

Run these on a Databricks GPU cluster to benchmark your configuration.

```python
from transolver3.profiling import (
    profile_memory, profile_latency, benchmark_scaling, format_benchmark_table
)

# Single measurement
mem = profile_memory(model, x, tile_size=100_000, mode='forward')
print(f"Peak memory: {mem.peak_mb:.1f} MB")

lat = profile_latency(model, x, tile_size=100_000, num_runs=10)
print(f"Latency: {lat.mean_ms:.1f} +/- {lat.std_ms:.1f} ms")

# Scaling benchmark (like paper Figure 6)
results = benchmark_scaling(
    model,
    mesh_sizes=[1_000, 5_000, 10_000, 50_000, 100_000],
    configs=[
        {'label': 'no_tiling', 'tile_size': 0},
        {'label': 'tile_50k', 'tile_size': 50_000},
        {'label': 'tile_10k', 'tile_size': 10_000},
        {'label': 'cached', 'mode': 'cached', 'cache_chunk_size': 10_000},
    ],
)
print(format_benchmark_table(results))
```

### Log profiling results to MLflow

```python
import mlflow

with mlflow.start_run(run_name="gpu-benchmark"):
    for i, config in enumerate(results['configs']):
        label = config.get('label', f'config_{i}')
        for j, N in enumerate(results['mesh_sizes']):
            if results['memory'] and results['memory'][i]:
                mr = results['memory'][i][j]
                mlflow.log_metric(f"peak_mb_{label}", mr.peak_mb, step=N)
            if results['latency'] and results['latency'][i]:
                lr = results['latency'][i][j]
                mlflow.log_metric(f"latency_ms_{label}", lr.mean_ms, step=N)
```

**What to look for:**
- Tiling should reduce peak memory significantly for large meshes
- Cached inference has constant memory regardless of total mesh size
- Latency grows linearly with mesh size (for tiled/cached modes)
- If memory grows quadratically, tiling is not engaged — check `tile_size` param

### Run the DAB GPU benchmark

```bash
databricks bundle deploy -t a10g
databricks bundle run gpu_memory_benchmark
```

This sweeps mesh sizes from 1K to 8M points on a single GPU and logs all results to MLflow.

---

## Troubleshooting poor results

| Symptom | Likely cause | Diagnostic |
|---------|-------------|------------|
| High error everywhere | Underfitting | Train longer, increase model size, check normalization |
| High error at boundaries | Boundary layer not resolved | Increase `slice_num`, check mesh density at walls |
| High error in wake regions | Complex flow not captured | More training steps, larger `n_hidden` |
| Predictions all near zero | Missing denormalization | Apply `target_norm.decode(pred)` |
| Predictions have wrong scale | Units mismatch | Check coordinate/target units match between train and eval |
| Out-of-bounds on new geometry | Distribution shift | Check drift metrics in `ml.transolver3.drift_metrics`, consider fine-tuning |
| Good train loss, bad eval loss | Overfitting | Increase `subset_size` in AmortizedMeshSampler, add regularization |
| PSI > 0.2 on all channels | Major data pipeline change | Compare new mesh stats against `ml.transolver3.normalization_stats` |
