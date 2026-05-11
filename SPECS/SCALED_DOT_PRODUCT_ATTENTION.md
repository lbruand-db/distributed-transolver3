# Scaled Dot-Product Attention: v1 vs v3

A note on a small but consequential implementation difference between the
reference [Transolver v1](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
and our `distributed-transolver3` reimplementation.

## The difference

**Transolver v1** (`Physics_Attention.py`) computes attention manually:

```python
dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
attn = self.softmax(dots)
attn = self.dropout(attn)
out_slice_token = torch.matmul(attn, v_slice_token)
```

**Transolver-3** (`transolver3/physics_attention_v3.py`) delegates to PyTorch's
fused kernel:

```python
s_out = F.scaled_dot_product_attention(
    q, k, v, dropout_p=self.dropout_p if self.training else 0.0
)
```

Mathematically the two are equivalent: both implement
`softmax(QK^T / sqrt(d_head)) V` with dropout on the attention weights.
The scale factor `self.scale = dim_head ** -0.5` in v1 is exactly the default
SDPA uses internally.

## Why v1 stayed with manual attention

There is no single "best" answer in the v1 source or paper, but several
plausible reasons converge on the same choice:

### 1. Historical timing

`torch.nn.functional.scaled_dot_product_attention` shipped in **PyTorch 2.0
(March 2023)**. Transolver was first released in late 2023 / early 2024 around
the ICML 2024 submission cycle. Targeting PyTorch ≥2.0 would have shut out a
large fraction of academic users still on 1.13. Manual attention is a two-line
construction that works back to PyTorch 1.0 and on any device (CPU, CUDA, MPS).

### 2. The attention is already tiny

Physics-Attention does not run attention over the mesh — it runs it over the
**slice tokens**, where `M = slice_num = 64` by default
(`physics_attention_v3.py:115`). The attention matrix is `(B, H, 64, 64)`.

SDPA's biggest wins (Flash Attention, memory-efficient attention) come from
fusing the softmax with the matmul to avoid materializing the `(B, H, N, N)`
attention matrix. At `N = 64` the full attention matrix is ~16 KB per head per
batch element — already cache-resident. There is essentially no Flash Attention
speedup to capture, and PyTorch will most likely dispatch to the "math" backend
anyway, which is the same two-matmul-and-a-softmax that v1 wrote by hand.

In other words: v1 wasn't leaving performance on the table by skipping SDPA.
The expensive work in Physics-Attention is the `O(N·M)` slice/deslice on the
**mesh** side, not the `O(M^2)` attention on the slice side.

### 3. Pedagogical alignment with the paper

The v1 file mirrors Algorithm 1 of the paper line-for-line. Writing out
`matmul → scale → softmax → dropout → matmul` makes the code a transcription of
the math. SDPA collapses those five operations into a single opaque kernel
call, which is harder to read alongside the paper and harder to instrument
(e.g. printing attention entropy, inspecting attention maps for diagnostics).

### 4. Deterministic, backend-agnostic behavior

SDPA selects a backend at runtime (math, mem-efficient, flash, cudnn) based on
input shapes, dtype, and `torch.backends.cuda.sdp_kernel(...)` flags. Two
otherwise-identical runs on different GPUs (A100 vs H100, fp32 vs bf16) can
take different paths and produce bit-different outputs. For a research codebase
where reviewers re-run experiments, manual attention removes that source of
variance.

### 5. Easy to extend

Custom masks, learned attention biases (ALiBi-style), explicit temperatures,
or attention-map visualization all attach trivially to the manual form. With
SDPA they require either passing `attn_mask` (which forces the math backend
and erases the speedup motivation) or giving up the fused kernel entirely.

## Why v3 switched

Three things changed for the v3 reimplementation:

1. **PyTorch ≥2.0 is the floor.** `pyproject.toml` already requires `torch >=
   2.0`, and Databricks ML Runtime ships much newer than that. No compatibility
   cost.
2. **`torch.compile` and SDPA pair well.** Even when SDPA dispatches to the
   math backend at `M = 64`, compiling the surrounding module fuses the
   softmax, scale, and dropout more aggressively than a hand-written sequence
   of ops.
3. **It costs nothing to use.** The forward function shrinks by four lines,
   dropout is parameterized via the existing `dropout_p`, and the numerical
   result matches v1 to fp32 round-off. The test suite verifies cached vs
   direct equivalence at `~1e-7` (`CLAUDE.md` Style Guidelines), and SDPA does
   not regress that tolerance at `M = 64`.

So v3's choice is not "SDPA is faster here" — at `M = 64` it largely isn't —
but rather "SDPA is the idiomatic PyTorch 2.x form, the surrounding stack
(`torch.compile`, autocast, future backends) will keep improving it for free,
and there is no longer a compatibility reason to avoid it."

## Where the real performance work lives

For completeness, the optimizations that *do* matter in v3 are all in the
mesh-domain operations, not the attention:

- **Slice/deslice associativity**: moving `in_project_x` and `to_out` from the
  `O(N)` mesh domain to the `O(M)` slice domain (`physics_attention_v3.py:99`
  docstring, items 1-2).
- **Per-head matmul** for slice aggregation, avoiding a materialized
  `(B, H, N, C)` intermediate (`_slice_aggregate`, fix #2).
- **Geometry slice tiling** with gradient checkpointing, lowering peak memory
  from `O(N·M)` to `O(N_t·M)` (`_forward_tiled`).
- **Physical-state caching** for inference, so decoding any query subset is
  `O(N_q·M)` after a one-time `O(N·M)` cache build (`compute_physical_state`,
  `decode_from_cache`).

The attention call itself is a rounding error in the total FLOP and memory
budget. v1's choice to write it out and v3's choice to delegate to SDPA are
both defensible — the difference is stylistic and forward-looking, not a
correctness or performance fix.
