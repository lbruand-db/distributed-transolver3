# Transolver-3 Pipeline

## 1. Training

**Input:** mesh coordinates `x (B, N, space_dim)`, optional features `fx (B, N, fun_dim)`, ground-truth targets `target (B, N, out_dim)`
**Output:** trained model weights (saved checkpoint)

- Uses `AmortizedMeshSampler` to randomly subsample large meshes (100K-400K points per iteration)
- `train_step()` runs forward pass → relative L2 loss → backprop
- Supports mixed precision via `GradScaler`
- Optional `InputNormalizer` (min-max on coordinates) and `TargetNormalizer` (zero-mean/unit-variance on targets)

## 2. Cache Building (Inference Phase 1)

**Input:** full mesh coordinates `x (B, N, space_dim)`, optional `fx`, trained model
**Output:** `cache` — a list of L tensors `s_out (B, H, M, C)`, one per layer

- `model.cache_physical_states(x, fx, chunk_size=100_000)`
- Streams the full mesh in chunks through each layer, accumulating slice-domain physical states (`s_raw`, `d`)
- Only slice tokens (M << N) are kept on GPU; chunks are offloaded to CPU
- This is what makes 160M+ cell inference possible — you never need the full N on GPU at once

## 3. Decoding (Inference Phase 2)

**Input:** query point coordinates `x_query (B, N_q, space_dim)`, `cache` from step 2
**Output:** predictions `(B, N_q, out_dim)` — e.g. pressure, velocity fields

- `model.decode_from_cache(x_query, cache, fx_query)`
- For each layer: computes slice weights for query points, multiplies by cached `s_out`
- Can decode in chunks too (`decode_chunk_size`)
- Query points can be the full mesh or any arbitrary subset

The convenience wrapper `CachedInference.predict(x)` or `model.full_mesh_inference(x)` does steps 2+3 together.
