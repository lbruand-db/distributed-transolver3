# Why Databricks for Transolver v3 Distributed

This document outlines the top differentiators for running Transolver v3's
distributed mesh-sharded training and inference on Databricks, compared to
generic cloud ML platforms.

---

## 1. Integrated GPU Cluster Management with Spark-Native Distributed Launch

Transolver v3's mesh-sharded training requires `torchrun` with DDP, precise
GPU-to-shard mapping, and NCCL all-reduce — all of which demand careful
process orchestration. On standalone ML platforms, this means writing custom
launch scripts, managing `MASTER_ADDR`/`MASTER_PORT`, babysitting process
groups, and configuring GPU compute separately from the rest of your stack.

Databricks combines GPU cluster provisioning, driver management, and
distributed launch into a single managed environment. The project uses
**TorchDistributor** (an open-source PySpark API, contributed by Databricks
to Apache Spark 3.4):

```python
TorchDistributor(num_processes=8, local_mode=True, use_gpu=True).run(main)
```

While TorchDistributor itself is available on any Spark 3.4+ cluster, the
Databricks advantage is the **surrounding platform**: GPU-aware autoscaling,
pre-configured NCCL/CUDA in the ML Runtime, spot instance management with
automatic checkpointing, and a unified control plane that ties compute
lifecycle to the same workspace where data and experiments live. On standalone
ML platforms, the distributed launch, the GPU cluster, the storage, and the
experiment tracker are four separate services with four separate failure modes.

---

## 2. Unity Catalog Volumes for 100 GB+ Mesh Data Lifecycle

This is the killer differentiator for this specific workload. Each DrivAer ML
sample is ~12 GB (140M volume cells x 22 features x 4 bytes). A realistic
training dataset is 500+ samples = 6+ TB of `.npz` files. The project uses
memory-mapped I/O (`np.load(..., mmap_mode='r')`) with byte-range reads for
shard-aware loading.

On generic ML platforms, you manage blob storage mounts, fight FUSE filesystem
latency, and build your own data versioning. Databricks gives you:

- **Unity Catalog Volumes** as a governed, versioned namespace for datasets and
  checkpoints — already configured in the DAB bundle.
- **NVMe-backed local SSD** on p4d/p4de instances (24 GB/s aggregate bandwidth)
  for hot data, with seamless spill to cloud storage.
- **Delta Lake lineage** to track which mesh partitions trained which model
  version — critical when the CFD team iterates on geometry variants (the
  16-dim parametric deformations in each sample).

No other platform gives you governed data + bare-metal I/O performance in the
same abstraction.

---

## 3. MLflow + Unity Catalog: Unified Lineage from Mesh Data to Deployed Model

Transolver v3's memory behaviour is extremely hardware-sensitive — the project
ships an entire benchmark suite (`benchmarks/gpu_memory_benchmark.py`) because
2.9M cells fits on A100-80GB but OOMs on A10G. This makes experiment tracking
non-optional: every run must record not just loss curves, but GPU profile,
mesh size, shard count, peak memory, and communication overhead.

MLflow itself is open-source (created by Databricks, like TorchDistributor)
and available on other platforms — including managed MLflow tracking on
competing services. Auto-logging, run comparison, and basic model registry
are not Databricks-specific.

**The differentiator is Unity Catalog as the single governance layer across
data, experiments, and models.** On Databricks, the 6 TB of mesh `.npz` files
(UC Volumes), the MLflow experiment runs, and the registered model checkpoints
all live in the same UC namespace with unified access control and lineage:

- **End-to-end traceability**: from a deployed 768 KB inference cache back to
  the exact training data partition, the GPU tier it ran on, and the experiment
  run that produced it — in one queryable graph.
- **Model Registry in Unity Catalog** versions checkpoints alongside the
  hardware profile and data snapshot that produced them — essential when the
  same architecture behaves differently across GPU tiers (A10G vs A100-80).
- **Run comparison** across the full parameter space: "does 8-way sharding on
  A10G match the accuracy of 1-GPU A100-80?" by overlaying loss curves,
  relative L2 error, and R² — with each run linked to its data lineage.

On competing platforms, MLflow tracking works fine, but the model registry,
the blob storage holding mesh data, and the experiment store are three
separate services with separate ACLs and no automatic lineage between them.
For Transolver — where the team iterates across hardware profiles, mesh sizes
(400K to 160M cells), and geometric variants (16-dim parametric deformations)
— this fragmentation means manually reconstructing provenance that Databricks
provides out of the box.

---

## 4. DABs: Research Prototype to Reproducible Pipeline in One Artifact

The project already ships a `databricks.yml` with target profiles (`a10g`,
`a100_40`, `a100_80`) and job definitions for GPU memory benchmarks and
distributed validation tests. This is a **deployable asset bundle** that pins:

- Cluster specs (instance type, GPU count, Spark/DBR version)
- Job DAGs (benchmark -> train -> evaluate)
- Environment (Python deps, CUDA version via DBR ML Runtime)

A new team member runs `databricks bundle deploy --target a100_80` and gets an
identical environment. On competing platforms, reproducing a multi-GPU training
run means stitching together a compute definition, an environment definition,
a pipeline definition, and a data reference — four separate concepts with four
separate versioning stories. DABs collapse this into one Git-tracked artifact.

---

## Summary

| Concern | Databricks | Generic ML Platform |
|---------|-----------|-------------------|
| Multi-GPU launch | Managed GPU clusters + TorchDistributor + ML Runtime | Separate compute, launch, and CUDA config |
| 6+ TB mesh data | Unity Catalog Volumes + NVMe | Blob mounts + FUSE + manual versioning |
| Experiment tracking | MLflow + Unity Catalog unified lineage | MLflow tracking works, but data/model/experiment are separate silos |
| Reproducibility | DAB bundle — one `deploy` command | 4 separate config systems to align |