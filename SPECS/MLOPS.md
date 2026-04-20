# MLOps Gap Analysis

This document identifies MLOps best practices missing from the current
Transolver-3 training and deployment pipeline.

## Current State

What we already have:

- **Experiment tracking**: MLflow with per-epoch metrics, hyperparams, and model artifacts
- **Reproducibility**: Fixed seed, deterministic split files, all params logged
- **Model versioning**: UC Model Registry with field-specific model names
- **Model serving**: Scale-to-zero endpoint via `TransolverPyfunc` with input validation
- **Normalization artifacts**: Target normalizer saved to MLflow and restored in eval
- **Mixed precision**: AMP flag for paper-faithful training
- **Per-quantity metrics**: L2 errors logged per physical quantity (p_s, tau, u, p_v)

---

## Critical — Blocks Production Confidence

### MLOPS-1: No Quality Gate Between Evaluate and Register

**Problem**: The pipeline registers the model regardless of how bad the L2
error is. A diverged training run or a misconfigured experiment would still
be promoted to the Model Registry and deployed to serving.

**Fix**: Add a validation task between `evaluate` and `register` that reads
the MLflow run metrics and fails if they exceed a threshold. Example:

```python
# Fail if aggregate L2 > 10% or any per-quantity L2 > 15%
assert best_test_l2 < 0.10, f"Model too inaccurate: {best_test_l2:.4f}"
```

**Effort**: Low — small script + task dependency change.

---

### MLOPS-2: No Data Versioning

**Problem**: MLflow logs model hyperparameters but not which data produced
the model. If someone adds/removes NPZ files, regenerates splits, or
reprocesses VTP files, there is no way to trace which data version produced
which model. Reproducibility breaks silently.

**Fix**: Log data provenance as MLflow params at training time:
- Hash of `train.txt` and `test.txt` (SHA-256)
- Number of training / test samples
- Total mesh points in training set
- Data directory path

**Effort**: Low — add a few `mlflow.log_param()` calls.

---

### MLOPS-3: No Model Validation Test Before Serving

**Problem**: The `register` task promotes the model to UC Model Registry
without verifying it loads correctly and produces sane outputs. A corrupted
checkpoint or serialization bug would propagate to the serving endpoint.

**Fix**: Add a smoke test in the `register` task (or a new `validate` task):
1. Load the model from MLflow
2. Run inference on a small synthetic input
3. Check output shape matches expected `out_dim`
4. Check output values are finite (no NaN/Inf)
5. Check output magnitude is within a reasonable range

**Effort**: Low — extend `register_model.py` or add `validate_model.py`.

---

## Important — Standard MLOps Practice

### MLOPS-4: No Model Card / Metadata

**Problem**: No structured description of what the model does, its
limitations, expected input/output schema, or performance characteristics
is logged with the model in the registry.

**Fix**: Set `model_description` when registering the model version:
- Field type (surface / volume)
- Training dataset and split
- Best L2 error and per-quantity breakdown
- Input schema (space_dim, expected coordinate range)
- Output schema (out_dim, physical quantities)
- Known limitations (mesh size constraints, memory requirements)

**Effort**: Medium — update `register_model.py`.

---

### MLOPS-5: Dependencies Not Pinned

**Problem**: Pipeline specifies loose version ranges (`einops>=0.7`,
`timm>=1.0`, `torch>=2.0`). A training run today may use different
package versions than one next month, silently affecting reproducibility.

**Fix**:
- Pin exact versions in the DAB YAML environments and cluster libraries
- Log installed package versions as MLflow params at training start
- Consider using a `requirements.txt` or `uv.lock` snapshot per run

**Effort**: Low.

---

### MLOPS-6: No Alerting on Training Failures

**Problem**: If training diverges (loss goes NaN), early-stops too soon
(e.g., at epoch 20 instead of 200+), or the job fails silently, nobody
is notified. The pipeline just stops.

**Fix**:
- Add email/Slack webhook notifications on Databricks job failure
- Log a warning metric if training ends before a configurable minimum
  epoch count (e.g., `min_epochs=100`)
- Detect NaN loss in `train_epoch()` and fail fast with a clear message

**Effort**: Low (Databricks job notifications), Medium (NaN detection).

---

### MLOPS-7: No Checkpoint Cleanup

**Problem**: Periodic checkpoints (`training_checkpoint.pt`, `best_model.pt`,
`target_normalizer.pt`) accumulate in the UC Volume across runs. Old runs
leave stale artifacts that consume storage.

**Fix**:
- Add a cleanup step at the end of the pipeline that removes old checkpoints
- Or use run-specific subdirectories: `checkpoints/{mlflow_run_id}/`
- Set a retention policy on the UC Volume

**Effort**: Low.

---

### MLOPS-8: No Canary / Shadow Deployment

**Problem**: The `deploy` task creates or updates the serving endpoint
directly with the new model. In production, a bad model would immediately
serve all traffic.

**Fix**:
- Use Databricks Model Serving traffic splitting to route a percentage
  of traffic to the new model version
- Compare prediction quality on live traffic before full cutover
- Keep the previous model version as a rollback target

**Effort**: High — requires serving infrastructure changes.

---

## Nice to Have

### MLOPS-9: No Drift Monitoring

**Problem**: No tracking of input distribution or prediction distribution
over time on the serving endpoint. If inference inputs shift from the
training distribution, model quality degrades silently.

**Fix**:
- Enable Databricks Lakehouse Monitoring on the inference table
- Track input feature statistics (coordinate ranges, parameter distributions)
- Alert when prediction distributions shift significantly

**Effort**: Medium.

---

### MLOPS-10: No Cost Tagging

**Problem**: Jobs don't tag clusters with cost-tracking labels (team,
project, experiment). Makes it hard to attribute GPU spend across
experiments and fields.

**Fix**: Add `custom_tags` to all cluster definitions:
```yaml
custom_tags:
  Project: "transolver3"
  Field: "${var.field}"
  Team: "cfd-research"
```

**Effort**: Low.

---

### MLOPS-11: No Experiment Comparison

**Problem**: No automated comparison between the new training run and the
previous best run. Users must manually check MLflow to see if the new
model is better.

**Fix**:
- In the validation task, query MLflow for the previous best run
- Log `delta_vs_previous` metric (improvement or regression)
- Fail the pipeline if the new model regresses significantly

**Effort**: Medium.

---

## Priority Matrix

| Gap | Severity | Effort | Priority |
|-----|----------|--------|----------|
| MLOPS-1: Quality gate | Critical | Low | **P0** |
| MLOPS-2: Data versioning | Critical | Low | **P0** |
| MLOPS-3: Model smoke test | Critical | Low | **P0** |
| MLOPS-4: Model card | Important | Medium | P1 |
| MLOPS-5: Pin dependencies | Important | Low | **P1** |
| MLOPS-6: Alerting | Important | Low-Med | P1 |
| MLOPS-7: Checkpoint cleanup | Important | Low | P1 |
| MLOPS-8: Canary deployment | Important | High | P2 |
| MLOPS-9: Drift monitoring | Nice to have | Medium | P2 |
| MLOPS-10: Cost tagging | Nice to have | Low | P2 |
| MLOPS-11: Experiment comparison | Nice to have | Medium | P2 |
