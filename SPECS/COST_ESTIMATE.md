# Cost Estimate: Reproducing DrivAerML Paper Results on Databricks

This document estimates the cost of reproducing the Transolver-3 paper's
DrivAerML L2 errors (Table 4) on Databricks, including data download,
training, evaluation, and storage.

## Paper Reference

Paper: Transolver-3 (Zhou et al., arXiv:2602.04940)

Target metrics (Table 4, DrivAerML):

| Quantity | Paper L2 (%) |
|----------|-------------|
| Surface pressure (p_s) | 3.71 |
| Volume velocity (u) | 4.14 |
| Wall shear stress (tau) | 5.85 |
| Volume pressure (p_v) | 5.72 |

Reproducing all 4 metrics requires **two separate training runs**:
one for `--field surface` (p_s, tau) and one for `--field volume` (u, p_v).

## Training Time Estimate

The paper reports **~7 hours on a single NVIDIA H100** for one field
(500 epochs, 400 training samples, batch_size=1).

| Target | Instance | GPUs | Est. time per field | Reasoning |
|--------|----------|------|---------------------|-----------|
| `a10g` | g5.12xlarge | 4x A10G | 10-12 hrs | 4x A10G ~ 0.6-0.7x H100 effective (lower FP16 FLOPS + DDP comm overhead) |
| `a100_40` | p4d.24xlarge | 8x A100-40G | 3-4 hrs | 8x A100-40G ~ 2x H100 effective |
| `a100_80` | p4de.24xlarge | 8x A100-80G | 3 hrs | Similar compute to a100_40, more memory headroom |

Note: Early stopping (`--patience 5`, eval every 10 epochs) may terminate
training well before epoch 500, reducing actual cost.

## Cost Breakdown: a10g Target (g5.12xlarge)

Cheapest option. Slower but sufficient for paper reproduction.

| Phase | Instance | Duration | AWS on-demand $/hr | DBU/hr | Est. cost |
|-------|----------|----------|--------------------|--------|-----------|
| Download raw VTP | Serverless | ~1 hr | -- | ~0.07/DBU | ~$5 |
| Preprocess | i3.xlarge | ~0.5 hr | $0.31 | 1 | ~$1 |
| Train (surface) | g5.12xlarge | ~11 hr | $5.67 | 8 | ~$97 |
| Train (volume) | g5.12xlarge | ~11 hr | $5.67 | 8 | ~$97 |
| Evaluate (x2) | g5.12xlarge | ~1 hr each | $5.67 | 8 | ~$18 |
| Register + Deploy | Serverless | ~5 min | -- | -- | ~$1 |
| **Total** | | **~26 hrs wall** | | | **~$220** |

Surface-only reproduction (p_s + tau): **~$120**

## Cost Breakdown: a100_40 Target (p4d.24xlarge)

Faster. Higher hourly rate but shorter training time.

| Phase | Instance | Duration | AWS on-demand $/hr | DBU/hr | Est. cost |
|-------|----------|----------|--------------------|--------|-----------|
| Download raw VTP | Serverless | ~1 hr | -- | ~0.07/DBU | ~$5 |
| Preprocess | i3.xlarge | ~0.5 hr | $0.31 | 1 | ~$1 |
| Train (surface) | p4d.24xlarge | ~3.5 hr | $32.77 | 28 | ~$170 |
| Train (volume) | p4d.24xlarge | ~3.5 hr | $32.77 | 28 | ~$170 |
| Evaluate (x2) | p4d.24xlarge | ~0.5 hr each | $32.77 | 28 | ~$50 |
| Register + Deploy | Serverless | ~5 min | -- | -- | ~$1 |
| **Total** | | **~10 hrs wall** | | | **~$400** |

Surface-only reproduction (p_s + tau): **~$225**

## Data Storage (UC Volumes)

| Data | Size | UC Volume cost/month |
|------|------|----------------------|
| Raw VTP files (500 runs) | ~150-200 GB | ~$4.50 |
| Surface NPZ (500 samples, ~8.8M pts each) | ~175 GB | ~$4.00 |
| Volume NPZ (500 samples, ~140M pts each) | ~2 TB | ~$46.00 |
| Checkpoints + models | ~5 GB | ~$0.12 |
| **Surface-only total** | **~380 GB** | **~$9/month** |
| **Full (surface + volume) total** | **~2.4 TB** | **~$55/month** |

## Pipeline Commands

```bash
# 1. Download raw data
databricks bundle run download_drivaer_raw -t a10g --profile DEFAULT

# 2. Run surface training pipeline
databricks bundle run transolver3_training_pipeline -t a10g \
  --profile DEFAULT

# 3. Run volume training pipeline
databricks bundle run transolver3_training_pipeline -t a10g \
  -v field=volume --profile DEFAULT
```

## Assumptions and Caveats

- AWS on-demand pricing as of early 2026; spot instances would reduce
  costs by 50-70% for training workloads.
- Databricks DBU rates are approximate and vary by pricing tier, region,
  and contract. Contact your Databricks account team for exact rates.
- Training time assumes the full 500 epochs. Early stopping with
  `--patience 5` (50 epoch lookback) typically converges by epoch
  200-300, which would reduce training cost by 40-60%.
- Evaluation cost depends on mesh size: surface eval (~8.8M pts) is
  fast; volume eval (~140M pts) takes significantly longer.
- VTP-to-NPZ conversion cost is not included above; it requires
  pyvista/VTK on a CPU node and takes ~2-4 hours for 500 samples.
