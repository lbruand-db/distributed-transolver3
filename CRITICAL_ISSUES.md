# Critical Issues — Transolver-3 Implementation vs Paper

## Confidence Assessment

**We cannot replicate the paper's memory scalability claims as-is.** The critical blockers are listed below.

| Issue | Impact | Paper's 2.9M single-GPU claim |
|-------|--------|-------------------------------|
| einsum may materialize (B,H,N,C) | Blows up memory at ~3M points | **Blocks it** |
| MLP/LN not tiled | ~3GB per layer for 2.9M points | Tight but possible on A100 80GB |
| `_cache_chunked` stores all features | 163GB for 160M points | **Blocks inference** |
| No mixed precision | 2× memory overhead | Reduces capacity by ~2× |
| Head aggregation bug | Wrong model behavior | Wrong results |

To replicate the paper's claims, we need to fix issues **#1, #2, #3, #4**, and add **#6**.

---

## Critical Issues — Bugs That Would Break Scalability

### 1. Head aggregation is wrong (`mean` instead of `rearrange`)

**File:** `transolver3/physics_attention_v3.py:138`

This is a **functional bug**. In v1:
```python
out_x = rearrange(out_x, 'b h n d -> b n (h d)')  # concat heads → B,N,inner_dim
return self.to_out(out_x)                           # Linear(inner_dim, dim)
```

In our v3 we do `x_out.mean(dim=1)` — averaging heads. But in the paper, `slice_linear3` absorbs the `to_out` linear. Since in the standard config `inner_dim == dim` (256 = 8×32), the correct approach is:

- `slice_linear3` should map `dim_head → dim_head` (not `dim_head → dim`)
- After deslice: `rearrange(x_out, 'b h n d -> b n (h d)')` to get `B, N, dim`

This changes model capacity and breaks the mathematical equivalence with the paper.

### 2. `s_raw` accumulates raw `x` in full `dim`-space per head — hidden O(HMC) memory

**File:** `transolver3/physics_attention_v3.py:117`

`s_raw = einsum("bhnm,bnc->bhmc", w, x)` produces shape `(B, H, M, C)`. With H=8, M=64, C=256 this is tiny. But the einsum itself requires PyTorch to broadcast `x` (B,N,C) against `w` (B,H,N,M). **PyTorch's einsum may materialize a (B,H,N,C) intermediate** depending on the backend. For N=2.9M, H=8, C=256, that's 2.9M × 8 × 256 × 4 bytes = **23.6 GB** — exceeding A100 40GB.

The paper avoids this by doing `wᵀx` as a matrix multiply, not an einsum. The correct memory-efficient approach is:
```python
# For each head, do: s_raw[h] = w[h].T @ x   (N_t,M).T @ (N_t,C) = (M,C)
# This never materializes (H,N,C)
```

### 3. MLP and LayerNorm in blocks are NOT tiled

**File:** `transolver3/transolver3_block.py:38-39`

The feed-forward MLP operates on **all N points** — `self.mlp(self.ln_2(fx))` where `fx` is `(B, N, hidden_dim)`. For N=2.9M, hidden=256, mlp_ratio=1:
- MLP intermediate: 2.9M × 256 × 4 bytes = **2.96 GB** per activation
- LayerNorm: same

The paper handles this during **training** via amortized subsets (400K), but during **cached inference decode** (`forward_from_cache`), we run the MLP on the full decode chunk. This is not tiled/checkpointed, so each decode chunk must be small enough for the MLP.

### 4. `_cache_chunked` stores all chunk features simultaneously

**File:** `transolver3/model.py:212-218`

```python
chunks_fx = []
for k in range(num_chunks):
    chunks_fx.append(self._preprocess(x_k, fx_k, T))
```

This stores **all N preprocessed features in memory** (just split into a list). For 160M points × 256 dims × 4 bytes = **163 GB**. This defeats the purpose of chunking. The paper processes one chunk at a time and only keeps the accumulators.

### 5. Preprocessing with `unified_pos` and `get_grid` isn't chunk-safe

**File:** `transolver3/model.py:83-95`

`get_grid` computes distances from each point to a reference grid. If you preprocess chunk-by-chunk, the reference grid min/max may differ per chunk vs. the full mesh, creating inconsistent positional encodings.

---

## Missing Features From Paper

### 6. Mixed precision (float16/bfloat16) not implemented

Paper (Appendix A.4): "Training was conducted in either float16 or bfloat16 precision." This is critical for memory — halves the footprint of all activations.

### 7. No target normalization

Paper (Appendix A.3): "Target outputs are standardized to have zero mean and unit variance across the dataset." Our experiment scripts don't compute or apply dataset-level normalization statistics.

### 8. Tile size configuration

Paper (Table 5): "A tile size of 100k serves as an ideal choice." We expose `num_tiles` but don't auto-compute the tile count from a target tile size.

### 9. Input preprocessing

Paper (Appendix A.3): "Geometric features are typically first normalized using min-max scaling and then optionally multiplied by a constant scaling factor (e.g., 1000)." Our dataset loaders implement this, but the model preprocessing doesn't enforce it.

### 10. No memory profiling / benchmarking

The paper's Figure 6 shows precise memory consumption curves. We have no tooling to measure/verify memory savings from tiling.
