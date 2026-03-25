# Scaling Transolver v3 to Huge Meshes (>100 GB)

## Problem Statement

DrivAerML volume meshes are ~140M cells per sample. At 22 input features × 4 bytes,
a single sample is **~12 GB** of coordinates alone. With targets, normals, and multiple
samples, the full dataset can exceed **100 GB**. Future automotive/aerospace meshes
will be even larger.

The bottleneck is **not the model** (~300M params = 1.2 GB) — it's the **mesh data**:

| Component | Size per sample (140M volume) | Fits in... |
|-----------|---:|---|
| Coordinates (140M × 3 × 4B) | 1.7 GB | GPU (barely) |
| Full input (140M × 22 × 4B) | 12.3 GB | CPU RAM only |
| Targets (140M × 4 × 4B) | 2.2 GB | CPU RAM |
| Preprocessed features (140M × 256 × 4B) | 143 GB | **Nothing** |
| Model + optimizer | 4.2 GB | GPU |
| Cache per layer: (1, 8, 64, 32 × 4B) | **32 KB** | Anything |

The key architectural insight: Transolver v3's slice accumulators `s_raw (B, H, M, C)`
and `d (B, H, M)` are **additive** — they can be independently computed from disjoint
mesh partitions and summed. This makes mesh sharding mathematically exact.

---

## Databricks Runtime

| Instance | GPUs | GPU RAM | CPU RAM | NVLink | Storage |
|----------|:----:|--------:|--------:|--------|---------|
| g5.xlarge | 1× A10G | 24 GB | 16 GB | N/A | EBS |
| g5.12xlarge | 4× A10G | 96 GB | 192 GB | NVSwitch | EBS |
| p4d.24xlarge | 8× A100 | 320 GB | 1.1 TB | 600 GB/s | 8 TB NVMe |
| p4de.24xlarge | 8× A100-80 | 640 GB | 1.1 TB | 600 GB/s | 8 TB NVMe |

Multi-GPU launches via `TorchDistributor` or `torchrun`.
Checkpoints to Unity Catalog Volumes.

---

## Three Options for Huge Meshes

### Option A: Mesh-Sharded Training

**Each GPU/node owns a disjoint partition of the mesh. The model is replicated.**

#### Concept

Instead of loading the full 140M-point mesh into one process and subsampling,
**shard the mesh files** across K workers. Each worker loads only its N/K partition
(e.g., 17.5M points on 8 GPUs). Each worker independently:

1. Subsamples its local partition (amortized training: 400K/K = 50K points)
2. Runs forward/backward on its local subset
3. All-reduces gradients (model is replicated, DDP-style)

Because each worker's subset comes from a different spatial region of the mesh,
the combined gradient covers more geometry per step — exactly like the current
`AmortizedMeshSampler` but with the I/O distributed.

```
               Full mesh (140M points, 12 GB)
              /        |         |         \
         Shard 0    Shard 1   Shard 2    Shard 3     ← disk/mmap
         (35M pts)  (35M pts) (35M pts)  (35M pts)
            |          |         |          |
        GPU 0       GPU 1     GPU 2      GPU 3       ← local subset
        (50K pts)   (50K pts) (50K pts)  (50K pts)
            |          |         |          |
        forward     forward   forward    forward
            |          |         |          |
            \__________|_________|__________/
                    all-reduce gradients              ← 600MB, NVLink ~1ms
```

#### Data Loading

The dataset loader must support **range-based loading** from memory-mapped files:

```python
# Each worker reads only its shard from the mmap'd file
data = np.load(path, mmap_mode='r')
my_start = rank * (N // world_size)
my_end = (rank + 1) * (N // world_size)
coords = data['volume_coords'][my_start:my_end]  # reads only this slice from disk
```

This requires the `.npz` files to be stored with coordinates in a single contiguous
array (already the case in DrivAerML format). No data duplication needed — every
worker mmap's the same file but reads a different range.

#### Memory per GPU

| Component | Single GPU | 8-way sharded |
|-----------|----------:|-------------:|
| Mesh data in CPU | 14.5 GB | **1.8 GB** |
| Training subset on GPU | ~400K pts = 40 MB | ~50K pts = 5 MB |
| Model + optimizer | 4.2 GB | 4.2 GB |
| Activations (24 layers) | ~200 MB | ~25 MB |

A single A10G (24 GB GPU, 16 GB CPU) **cannot** load a 140M mesh today.
With 4-way sharding on g5.12xlarge, each A10G handles 35M points with 4 GB CPU.

#### Communication

- Gradient all-reduce: ~600 MB per step (NVLink: ~1 ms)
- No mesh data transferred between GPUs
- Each worker reads independently from shared filesystem (DBFS/NVMe)

#### Complexity: Low-Medium (~200 LOC)

Files to modify:
- `dataset/drivaer_ml.py` — add `shard_id`/`num_shards` params for range-based mmap loading
- `exp_drivaer_ml.py` — DDP wrapping, per-rank shard assignment
- New `transolver3/distributed.py` — setup/cleanup utilities

---

### Option B: Mesh-Sharded Cache Build (Distributed Inference)

**Shard the mesh across workers for Phase 1 (cache build), all-reduce the tiny
accumulators, then shard query points for Phase 2 (decode).**

#### Concept

The cache build loop in `_cache_chunked` iterates:

```python
for layer in blocks:
    for chunk in mesh_chunks:
        s_raw_accum += compute_physical_state(chunk)   # (B, H, M, C) = 128 KB
        d_accum += ...                                  # (B, H, M)    = 2 KB
```

Since `s_raw` and `d` are **additive**, different workers can process different
chunks and all-reduce the accumulators:

```
          Full mesh (140M points)
         /         |          \
    Worker 0    Worker 1    Worker 2      ← each streams its partition
    (47M pts)   (47M pts)   (47M pts)
        |          |           |
    local s_raw  local s_raw  local s_raw  ← (B, H, M, C) = 128 KB each
        |          |           |
        \__________|___________/
           all-reduce s_raw, d             ← 130 KB total, instant
                   |
            compute_cached_state           ← s_out (B, H, M, dim_head) = 32 KB
                   |
          broadcast s_out to all
                   |
    ┌──────────┬──────────┬──────────┐
    Worker 0   Worker 1   Worker 2         ← advance local chunks through layer
    (on CPU)   (on CPU)   (on CPU)
```

After all layers, each worker holds the full cache (24 × 32 KB = 768 KB).
Then Phase 2 (decode) shards query points embarrassingly parallel.

#### The key numbers

- **All-reduce per layer**: `s_raw` is (1, 8, 64, 256) × 4 bytes = **512 KB** + `d` is (1, 8, 64) × 4 bytes = **2 KB**. Total: **~514 KB per layer, 24 layers = 12 MB total**.
- **Decode shard**: Each worker decodes N/K query points independently. For 140M volume mesh on 8 GPUs: 17.5M points each, chunked at 50K.
- **Result gather**: 140M × 4 outputs × 4 bytes = 2.2 GB. Done once, streamed.

#### Memory per GPU

| Phase | Single GPU | 8-way sharded |
|-------|----------:|-------------:|
| Cache build: mesh in CPU | 143 GB (features) | **18 GB** |
| Cache build: GPU peak | chunk_size × 256 | Same (unchanged) |
| Cache result | 768 KB | 768 KB (replicated) |
| Decode: query points | 140M × 256 = 143 GB | **18 GB** |
| Decode: GPU peak | decode_chunk × 256 | Same |

**This is the only way to run inference on a 140M-point mesh without 143 GB of CPU RAM.**

#### Complexity: Medium (~300 LOC)

Files to modify:
- `transolver3/model.py` — new `_cache_chunked_distributed` method with all-reduce on `s_raw`, `d`
- `transolver3/inference.py` — new `DistributedCachedInference` class
- `exp_drivaer_ml.py` — distributed evaluate function

---

### Option C: Streaming from Disk (Out-of-Core)

**The mesh never fully loads into RAM. Chunks are streamed from disk (NVMe/DBFS)
directly to GPU, one at a time.**

#### Concept

For meshes that don't fit in CPU RAM of even a multi-GPU node (future meshes
>1 TB, or training on smaller instances), stream chunks from disk:

```
NVMe/DBFS ──mmap──► CPU page cache ──pin──► GPU
                    (OS-managed)       (one chunk at a time)
```

Training:
1. mmap the full file with `np.load(path, mmap_mode='r')`
2. Random-access index into the mmap for amortized sampling
3. OS page cache handles the I/O transparently — no explicit buffering needed
4. Only the accessed 400K-point subset is paged into RAM

Inference (cache build):
1. Iterate chunks sequentially through the mmap
2. Each chunk is loaded, preprocessed, accumulated into `s_raw`/`d`, then evicted
3. CPU peak: 1 chunk × features, not full mesh

Inference (decode):
1. Same streaming — iterate query points in chunks from mmap
2. Decode against the cache (768 KB, already on GPU)

#### The insight

**This already nearly works.** The existing `_cache_chunked` method streams chunks
and offloads to CPU. The missing piece is that `chunks_fx_cpu` stores ALL
preprocessed features on CPU (O(N × hidden_dim) = 143 GB for volume mesh).

The fix: **don't store** `chunks_fx_cpu`. Instead, re-read and re-preprocess
each chunk from the mmap when advancing through layers. This trades compute
for memory: each chunk is preprocessed L times (once per layer) instead of 1 time.

```python
# Current: O(N × hidden_dim) CPU memory
chunks_fx_cpu = [preprocess(chunk_k).cpu() for k in range(num_chunks)]

# Proposed: O(chunk_size × hidden_dim) CPU memory
for layer in blocks:
    for k in range(num_chunks):
        chunk_raw = load_chunk_from_mmap(k)         # re-read from disk
        chunk_fx = preprocess_through_layers(chunk_raw, layers[:layer])  # recompute
        s_raw += compute_physical_state(chunk_fx)
```

#### Trade-offs

| Aspect | Store-all (current) | Stream from disk |
|--------|---:|---:|
| CPU RAM | O(N × C) = 143 GB | O(chunk_size × C) = 100 MB |
| Disk I/O | 1× read | L× read (24× for 24 layers) |
| Compute | 1× preprocess | L× preprocess per chunk |
| NVMe bandwidth (p4d: 8 TB) | N/A | ~3 GB/s per SSD, 8 SSDs |

For 140M points × 24 layers: re-reading is 24 × 12 GB = 288 GB of I/O.
On p4d NVMe (24 GB/s aggregate): ~12 seconds. Acceptable for inference.

#### Complexity: Medium (~250 LOC)

Files to modify:
- `transolver3/model.py` — new `_cache_streaming` method that recomputes from mmap
- `transolver3/inference.py` — accept a `data_source` (mmap or tensor) instead of requiring full `x` tensor
- `dataset/drivaer_ml.py` — expose mmap handle for streaming access

---

## Comparison Matrix

| Criterion | A: Mesh-Sharded Training | B: Mesh-Sharded Inference | C: Streaming from Disk |
|-----------|:---:|:---:|:---:|
| **Solves >100 GB mesh** | Yes (training) | Yes (inference) | Yes (both) |
| Training speedup (8 GPU) | ~7x | N/A | ~1x (I/O bound) |
| Inference speedup (8 GPU) | N/A | ~7x | ~1x |
| CPU RAM per worker | N/K | N/K | O(chunk_size) |
| Communication overhead | 600 MB gradients | 12 MB accumulators | None |
| Disk I/O | 1× read per epoch | 1× read | 24× read |
| Databricks feasibility | p4d/p4de (NVLink) | p4d/p4de (NVLink) | Any (even single GPU) |
| Complexity | Low-Medium | Medium | Medium |
| Can combine with others | A+B together | A+B together | C+A, C+B |

---

## Recommended Implementation Order

### Phase 1: A + B together (Mesh-Sharded Training + Inference) — 3-5 days

This is the primary strategy for Databricks on p4d/p4de instances. Sharding the mesh
across 8 A100s gives each GPU only 1/8 of the data while the all-reduce on slice
accumulators is negligible (514 KB per layer).

**Implementation steps:**

1. **`transolver3/distributed.py`** (new, ~40 LOC)
   - `setup_distributed()`, `cleanup()`, `is_main_process()`, `get_device()`
   - `mesh_shard_range(N, rank, world_size)` → `(start, end)` index range

2. **`dataset/drivaer_ml.py`** (~50 LOC changes)
   - Add `shard_id` and `num_shards` parameters
   - Use `mmap_mode='r'` and read only `[start:end]` slice from arrays
   - Subsample only within the local shard

3. **`exp_drivaer_ml.py`** (~80 LOC changes)
   - DDP wrapping, per-rank shard assignment
   - `AmortizedMeshSampler(seed=base_seed + rank)` for different subsets
   - Scale LR by `world_size`
   - Only rank 0 saves/logs
   - Distributed evaluate with sharded cache build + decode

4. **`transolver3/model.py`** (~60 LOC)
   - New `_cache_chunked_distributed()` that all-reduces `s_raw`, `d` per layer
   - Workers process their local chunks, then `dist.all_reduce(s_raw_accum)`

5. **`transolver3/inference.py`** (~80 LOC)
   - New `DistributedCachedInference` class
   - Sharded `build_cache()`: each rank processes local mesh partition, all-reduces accumulators
   - Sharded `decode()`: each rank decodes its partition of query points, gathers results

6. **Databricks entry point** `train_distributed.py` (~20 LOC)
   ```python
   from pyspark.ml.torch.distributor import TorchDistributor
   TorchDistributor(num_processes=8, local_mode=True, use_gpu=True).run(main)
   ```

### Phase 2: Option C (Streaming from Disk) — 2-3 days

For meshes that exceed even the aggregate CPU RAM of a multi-GPU node (>1 TB),
or for running on smaller instances. Can be combined with A+B.

**Implementation steps:**

1. **`transolver3/model.py`** (~100 LOC)
   - New `_cache_streaming()`: re-reads and recomputes chunks from an mmap source
     rather than storing `chunks_fx_cpu` in memory
   - Accepts a callable `chunk_loader(k) → (x_chunk, fx_chunk)` instead of full tensor

2. **`transolver3/inference.py`** (~50 LOC)
   - `StreamingCachedInference` that accepts mmap file paths instead of tensors
   - Streams chunks from disk for both cache build and decode

3. **`dataset/drivaer_ml.py`** (~30 LOC)
   - Expose `get_mmap_handle()` for streaming access to raw arrays

---

## Files to Modify

| File | Phase 1 (Sharded) | Phase 2 (Streaming) |
|------|:---:|:---:|
| `transolver3/distributed.py` (new) | X | |
| `transolver3/model.py` | X | X |
| `transolver3/inference.py` | X | X |
| `Industrial-Scale-Benchmarks/exp_drivaer_ml.py` | X | |
| `Industrial-Scale-Benchmarks/dataset/drivaer_ml.py` | X | X |
| `Industrial-Scale-Benchmarks/train_distributed.py` (new) | X | |

## Verification

- **Phase 1 (Sharded)**: Compare cache tensors from sharded (8 GPU) vs single-GPU on the same mesh. `s_out` should match to FP32 precision (additive accumulation is associative). Verify inference outputs are identical. Measure training loss curve convergence.
- **Phase 2 (Streaming)**: Compare `_cache_streaming` output to `_cache_chunked` on a mesh that fits in RAM. Outputs must match exactly. Profile disk I/O throughput vs compute time to verify I/O is not the bottleneck on NVMe.