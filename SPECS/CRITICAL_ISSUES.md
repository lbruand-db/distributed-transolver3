# Critical Issues — Transolver-3 Implementation vs Paper

## Status: All Critical Issues Fixed

All 5 critical bugs have been fixed and verified with 22 tests (9 original + 10 fix-specific + 3 tile_size).

| Issue | Status | Fix |
|-------|--------|-----|
| #1 Head aggregation (mean → rearrange) | **FIXED** | `slice_linear3` is now `dim_head→dim_head`, heads concatenated via rearrange |
| #2 einsum materializes (B,H,N,C) | **FIXED** | Replaced with `_slice_aggregate` / `_deslice` using `torch.matmul` with broadcast |
| #3 MLP/LN not tiled | **FIXED** | Added `mlp_chunk_size` param + `_pointwise_chunked` helper |
| #4 `_cache_chunked` stores all features on GPU | **FIXED** | CPU offloading — only one chunk on GPU at a time |
| #5 `get_grid` chunk safety | **N/A** | Uses fixed [0,1] linspace — already chunk-safe by design |
| #6 Mixed precision | **SUPPORTED** | Works with `torch.autocast`; user wraps training loop |

---

## Fix Details

### Fix #1: Head aggregation — `rearrange` (concat) not `mean`

**File:** `transolver3/physics_attention_v3.py`

- `slice_linear3` changed from `Linear(dim_head, dim)` to `Linear(dim_head, dim_head)`
- After deslice: `rearrange(x_out, 'b h n d -> b n (h d)')` instead of `x_out.mean(dim=1)`
- Added `assert inner_dim == dim` to catch misconfiguration
- **Test:** `test_fix1_head_concat_not_mean` — verifies shapes, linear dimensions, head diversity

### Fix #2: Memory-efficient slice/deslice via matmul

**File:** `transolver3/physics_attention_v3.py`

- `_slice_aggregate(w, x, heads)`: `w.transpose(2,3) @ x.unsqueeze(1)` — broadcasts x without materializing (B,H,N,C)
- `_deslice(s_out, w)`: `w @ s_out` — direct matmul
- Applied in standard, tiled, and cached paths
- **Test:** `test_fix2_no_large_intermediate` — verifies matmul matches einsum reference

### Fix #3: MLP and LayerNorm tiling

**File:** `transolver3/transolver3_block.py`

- Added `mlp_chunk_size` parameter to `Transolver3Block`
- `_pointwise_chunked(fn, x, chunk_size)` processes LN+MLP in chunks along dim=1
- Applied to both `forward` and `forward_from_cache` paths
- **Tests:** `test_fix3_mlp_chunking`, `test_fix3_pointwise_chunked_helper`, `test_model_with_mlp_chunking`

### Fix #4: Streaming cache with CPU offloading

**File:** `transolver3/model.py`

- `_cache_chunked` now stores intermediate chunk features on CPU (`chunks_fx_cpu`)
- Each chunk is moved to GPU one at a time for processing, then result goes back to CPU
- GPU memory: O(chunk_size × hidden_dim) instead of O(N × hidden_dim)
- CPU memory: O(N × hidden_dim) — fits in system RAM for 160M points (~40GB in fp32)
- **Tests:** `test_fix4_streaming_cache`, `test_fix4_memory_pattern`

### Fix #5: Chunk-safe preprocessing (no fix needed)

**File:** `transolver3/model.py`

- `get_grid` uses `np.linspace(0, 1, ref)` — fixed reference grid independent of input range
- Positional encoding is chunk-safe by design
- Added documentation comment noting this

### Fix #6: Mixed precision support

**File:** `transolver3/model.py` (usage pattern)

- Model is compatible with `torch.autocast(device_type, dtype=torch.bfloat16)`
- User wraps their training/inference loop with autocast
- Gradient scaler recommended for float16 on CUDA
- **Tests:** `test_fix6_mixed_precision`, `test_fix6_autocast_numerics`

---

## Remaining Improvements (Non-Critical)

### ~~7. No target normalization~~ — DONE

Paper (Appendix A.3): "Target outputs are standardized to have zero mean and unit variance across the dataset."

**Fixed:** Added `TargetNormalizer` in `transolver3/normalizer.py`. Supports `fit()` (full tensor) and `fit_incremental()` (streaming Welford's for large datasets). `encode()`/`decode()` for training/inference. Registered as `nn.Module` for `state_dict` save/load. Integrated with `train_step` via `normalizer=` parameter.

**Tests:** `test_target_normalizer_fit`, `test_target_normalizer_incremental`, `test_target_normalizer_2d_input`, `test_target_normalizer_state_dict`, `test_target_normalizer_device`, `test_train_step_with_normalizer`

### ~~8. Tile size configuration~~ — DONE

Paper (Table 5): "A tile size of 100k serves as an ideal choice."

**Fixed:** Added `tile_size` parameter to `PhysicsAttentionV3`, `Transolver3Block`, `Transolver3`, and `train_step`. When `tile_size > 0`, `num_tiles = ceil(N / tile_size)` is auto-computed via `_resolve_num_tiles()`. Usage: `model = Transolver3(..., tile_size=100_000)`.

**Tests:** `test_resolve_num_tiles`, `test_tile_size_attention`, `test_tile_size_model`

### ~~9. Input preprocessing~~ — DONE

Paper (Appendix A.3): "Geometric features are typically first normalized using min-max scaling and then optionally multiplied by a constant scaling factor (e.g., 1000)."

**Fixed:** Added `InputNormalizer` in `transolver3/normalizer.py`. Two modes: `per_sample=True` (default, per-sample min-max, no fitting) and `per_sample=False` (dataset-level, with `fit()`/`fit_incremental()`). Supports configurable `scale` factor. `encode()`/`decode()` roundtrip for dataset-level mode.

**Tests:** `test_input_normalizer_per_sample`, `test_input_normalizer_dataset_level`, `test_input_normalizer_incremental`, `test_input_normalizer_scale_factor`, `test_input_normalizer_2d`, `test_input_normalizer_decode_per_sample_raises`

### ~~10. No memory profiling / benchmarking~~ — DONE

The paper's Figure 6 shows precise memory consumption curves.

**Fixed:** Added `transolver3/profiling.py` with:
- `profile_memory(model, x, ...)` — peak memory for forward or cached inference (CUDA via `torch.cuda` stats, CPU via `tracemalloc`)
- `profile_latency(model, x, ...)` — wall-clock timing with warmup
- `benchmark_scaling(model, mesh_sizes, configs)` — sweep mesh sizes × tiling configs, returns structured `MemoryResult`/`LatencyResult` data
- `format_benchmark_table(results)` — readable text output for comparison

**Tests:** `test_profile_memory_forward`, `test_profile_memory_cached`, `test_profile_latency`, `test_benchmark_scaling`, `test_format_benchmark_table`, `test_tiling_reduces_memory_relative`
