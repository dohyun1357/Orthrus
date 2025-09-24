"""
Triton segment-sum/mean over variable-length sequences.

Given a 2D tensor `hidden_states` with shape [N, D] (N rows of token
hidden-states, D hidden size) and a batch of B sequences represented by
`offsets` (start row per sequence) and `lengths` (token count per sequence),
this module computes per-sequence sums over the token dimension, producing
`sums` with shape [B, D]. Means are obtained by dividing each row by the
corresponding length.

This is used to accelerate mean pooling in HybridMeanPooler without creating
Python loops or materializing per-sequence slices.
"""

from typing import Tuple

import torch

from vllm.triton_utils import triton, tl


@triton.jit
def _segment_sum_kernel(
    H_ptr,           # *f16/f32 [N, D]
    OFF_ptr,         # *i32 [B]
    LEN_ptr,         # *i32 [B]
    OUT_ptr,         # *f16/f32 [B, D]
    N: tl.constexpr,   # total rows in H (not used, for safety)
    D: tl.constexpr,   # hidden size
    MAX_LEN: tl.constexpr,  # max length across this batch
    BLOCK_D: tl.constexpr,
    DTYPE_IS_FP16: tl.constexpr,
):
    bid = tl.program_id(0)  # sequence id in [0, B)
    did = tl.program_id(1)  # block id along hidden dim

    # hidden-dim offsets for this program
    d_offsets = did * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # load segment start + length
    start = tl.load(OFF_ptr + bid)
    seqlen = tl.load(LEN_ptr + bid)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Iterate up to MAX_LEN with mask per sequence
    # Each step loads one token row vector slice and accumulates
    t = 0
    while t < MAX_LEN:
        active = t < seqlen
        # Compute row index and pointer into H
        row = start + t
        # base pointer for row: row * D + d_offsets
        h_ptrs = H_ptr + row * D + d_offsets
        vals = tl.load(h_ptrs, mask=d_mask & active, other=0.0)
        acc += vals.to(tl.float32)
        t += 1

    # Write back accumulated sum for this [seq, hidden-slice]
    out_ptrs = OUT_ptr + bid * D + d_offsets
    out_vals = acc.to(tl.float16) if DTYPE_IS_FP16 else acc
    tl.store(out_ptrs, out_vals, mask=d_mask)


def segment_sum(hidden_states: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence sums along token dimension using Triton.

    Args:
        hidden_states: [N, D] fp16/fp32 tensor (contiguous in memory).
        lengths: [B] int32/int64 tensor of token counts per sequence.

    Returns:
        sums: [B, D] tensor with per-sequence sums.
    """
    assert hidden_states.ndim == 2, "hidden_states must be [N, D]"
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Ensure proper dtypes and contiguity
    H = hidden_states.contiguous()
    B = int(lengths.numel())
    D = int(H.shape[1])

    lens_i32 = lengths.to(device=device, dtype=torch.int32, non_blocking=True)
    # Offsets: [0, cumsum(lens)[:-1]]
    offsets = torch.empty_like(lens_i32)
    if B > 0:
        torch.cumsum(lens_i32, dim=0, out=offsets)
        offsets = torch.roll(offsets, shifts=1, dims=0)
        offsets[0] = 0

    # Output buffer
    OUT = torch.zeros((B, D), device=device, dtype=dtype)

    if B == 0:
        return OUT

    max_len = int(lens_i32.max().item()) if B > 0 else 0
    if max_len == 0:
        return OUT

    BLOCK_D = 128
    grid = (B, (D + BLOCK_D - 1) // BLOCK_D)

    _segment_sum_kernel[grid](
        H,
        offsets,
        lens_i32,
        OUT,
        H.shape[0],
        D,
        max_len,
        BLOCK_D,
        int(dtype == torch.float16),
        num_warps=4,
        num_stages=2,
    )

    return OUT


def segment_mean(hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence means using Triton segment_sum.

    Args:
        hidden_states: [N, D]
        lengths: [B]

    Returns:
        means: [B, D]
    """
    sums = segment_sum(hidden_states, lengths)
    # Avoid division by zero
    denom = lengths.to(device=hidden_states.device, dtype=sums.dtype).clamp_min(1)
    return sums / denom.unsqueeze(1)
