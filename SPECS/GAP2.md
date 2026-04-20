# GAP2: Paper Reproduction Gap Analysis for DrivAerML

This document identifies gaps between the current implementation and the
Transolver-3 paper (Zhou et al., arXiv:2602.04940) training configuration,
specifically for reproducing the DrivAerML L2 errors reported in Table 4:

| Quantity | Paper L2 (%) |
|----------|-------------|
| Surface pressure (p_s) | 3.71 |
| Volume velocity (u) | 4.14 |
| Wall shear stress (τ) | 5.85 |
| Volume pressure (p_v) | 5.72 |

## Paper Training Configuration (Table 6, Appendix A.4)

| Parameter | Paper value |
|-----------|------------|
| Loss | Relative L2 |
| Epochs | 500 |
| Initial LR | 10⁻³ |
| Optimizer | AdamW |
| Subset size | 400K |
| Layers | 24 |
| Heads | 8 |
| Channels | 256 |
| Slices | 64 |
| Batch size | 1 |
| Weight decay | 0.05 |
| Grad clip | 1.0 |
| LR schedule | Cosine, 5% warmup, min LR 1×10⁻⁶ |
| Precision | float16 or bfloat16 mixed precision |
| Input normalization | Min-max to [0, 1] × 1000 |
| Target normalization | Zero-mean, unit-variance (z-score) |
| Data split | 400 train / 50 test / 34 validation |

---

## Gap Status

### GAP2-1: Mixed Precision (AMP) — NOT USED

**Paper** (Appendix A.4): "Training was conducted in either float16 or
bfloat16 precision."

**Current code**: The `train_step()` in `transolver3/amortized_training.py`
supports AMP via an optional `scaler` parameter and `torch.autocast`.
However, the distributed training script
(`Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py`) does **not**
use AMP at all. The `train_epoch()` function (line 163) runs entirely in
fp32 — no `autocast`, no `GradScaler`.

**Impact**: Higher memory usage (roughly 2× activations), slower training,
and potentially different convergence behavior. The paper numbers were
obtained with mixed precision, so reproducing them requires it.

**Fix**: Add `--amp` flag to the training script. Wrap the forward pass in
`torch.autocast(device_type="cuda", dtype=torch.float16)` and use
`torch.amp.GradScaler` for loss scaling. The plumbing already exists in
`train_step()` — the distributed script just needs to use it.

---

### GAP2-2: Target Normalization — NOT USED

**Paper** (Appendix A.3): "The target outputs, on the other hand, are
standardized to have zero mean and unit variance across the dataset."

**Current code**: `TargetNormalizer` exists in `transolver3/normalizer.py`
(line 167) and is supported by `train_step()`. However, the distributed
training script **never imports or uses it**. Targets are passed raw to
`relative_l2_loss`.

**Impact**: Without target normalization, the loss landscape is dominated
by high-magnitude output channels. This can slow convergence and shift
final L2 errors, especially for multi-channel outputs like wall shear
stress (3 components) vs pressure (1 component).

**Fix**: Before training, compute z-score statistics on the training set
(streaming via `TargetNormalizer.fit_incremental()` for large data).
Encode targets during training, decode predictions during evaluation.
Must ensure the normalizer's mean/std are saved with the model for
inference.

---

### GAP2-3: Input Normalization — OK

**Paper** (Appendix A.3): "Geometric features are typically first
normalized using min-max scaling and then optionally multiplied by a
constant scaling factor (e.g., 1000)."

**Current code**: `DrivAerMLDataset` applies per-sample min-max
normalization with `coord_scale=1000.0` by default (line 120, 160).

**Status**: Matches the paper.

---

### GAP2-4: LR Schedule — OK

**Paper** (Appendix A.4): "A cosine learning rate schedule was employed,
including a 5% warm-up phase and a minimum learning rate of 1×10⁻⁶."

**Current code**: `create_scheduler()` in `amortized_training.py` (line 93)
implements cosine decay with `warmup_fraction=0.05` and `min_lr=1e-6`.

**Status**: Matches the paper.

---

### GAP2-5: LR Scaling — NEEDS VERIFICATION

**Paper**: Does not mention linear LR scaling with world_size.

**Current code** (line 390): `scaled_lr = args.lr * world_size`. With 4
GPUs this produces an initial LR of 4×10⁻³ instead of 10⁻³.

**Impact**: Linear scaling is standard practice for DDP but is **not
mentioned in the paper**. If the paper trained on a single GPU (or did
not scale LR), this would produce a 4× higher learning rate, causing
divergence or different convergence.

**Fix**: Add a `--no-scale-lr` flag to disable LR scaling for paper
reproduction. Alternatively, verify with the paper authors whether LR
scaling was used.

---

### GAP2-6: Field-Specific Runs — PIPELINE NEEDS TWO RUNS

**Paper** (Table 4): Reports separate L2 errors for surface fields
(p_s, τ) and volume fields (u, p_v). These are trained as **separate
models** (Table 6 shows DrivAerML uses 24 layers for both).

**Current pipeline**: The `training_workflow.yml` trains with
`--field surface` (default). To reproduce all 4 metrics in Table 4,
you need two separate pipeline runs:

1. `--field surface` → produces p_s and τ errors
2. `--field volume` → produces u and p_v errors

**Fix**: Either duplicate the pipeline job for volume, or add a
parameter to the DAB workflow to select the field.

---

### GAP2-7: Data Split — NEEDS SPLIT FILES

**Paper** (Appendix A.2): "Since no predefined data split is available,
we randomly allocate 400 samples for training and 50 for testing, while
reserving 50 for validation. Of these validation cases, 16 lack
simulation results and are treated as hidden samples, yielding 34
effective validation samples."

**Current code**: `DrivAerMLDataset` reads from `train.txt`/`test.txt`
split files (line 140). If these files don't exist, it falls back to
listing all `.npz` files in the directory.

**Impact**: Reproducibility requires using the same 400/50/50 split.
Without the authors' split, results may differ due to train/test
composition.

**Fix**: Generate deterministic split files using a fixed seed (e.g.,
seed=42) to split the 500 samples into 400/50/50. Document the split
generation script.

---

### GAP2-8: Evaluation Metric Granularity — AGGREGATE ONLY

**Paper** (Table 4): Reports **per-quantity** L2 errors: p_s, τ, u, p_v
separately.

**Current code**: The `evaluate()` function (line 198) computes a single
`relative_l2_loss` over the entire concatenated target tensor. For
`--field surface`, this merges pressure (1 channel) and wall shear stress
(3 channels) into one number.

**Impact**: Cannot directly compare with the paper's per-quantity results.
The aggregate L2 is dominated by the 3-channel shear stress.

**Fix**: In `evaluate()`, split the prediction/target tensors by output
channel and compute L2 per quantity. For surface: channels [0] = p_s,
channels [1:4] = τ. For volume: channels [0:3] = u, channel [3] = p_v.
Log each to MLflow as separate metrics.

---

## Summary

| Gap | Status | Severity | Effort |
|-----|--------|----------|--------|
| GAP2-1: Mixed precision | **RESOLVED** — `--amp` flag added | High | Low |
| GAP2-2: Target normalization | **RESOLVED** — `TargetNormalizer` fitted + used | High | Medium |
| GAP2-3: Input normalization | OK | — | — |
| GAP2-4: LR schedule | OK | — | — |
| GAP2-5: LR scaling | **RESOLVED** — `--no-scale-lr` flag added | Medium | Low |
| GAP2-6: Volume field run | **RESOLVED** — `field` variable in DAB pipeline | Medium | Low |
| GAP2-7: Data split files | **RESOLVED** — `scripts/generate_split.py` | Medium | Low |
| GAP2-8: Per-quantity metrics | **RESOLVED** — per-quantity L2 in eval + MLflow | Medium | Medium |

All gaps have been addressed. To reproduce the paper's DrivAerML results:

```bash
# 1. Generate the data split
python scripts/generate_split.py --data_dir /path/to/data --seed 42

# 2. Train surface model (p_s, tau)
databricks bundle run transolver3_training_pipeline -t a10g \
  -v field=surface --profile DEFAULT

# 3. Train volume model (u, p_v)
databricks bundle run transolver3_training_pipeline -t a10g \
  -v field=volume --profile DEFAULT
```
