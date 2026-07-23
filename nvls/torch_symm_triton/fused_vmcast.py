# Copyright (c) 2026. Fused per-token variable-multicast dispatch/combine for the one-buffer vmcast.
"""
Fused kernels for the one-buffer variable-size multicast MoE dispatcher (bench_vmcast_onebuf.py).

Dispatch and combine are each ONE launch + ONE global barrier:

  - one CTA per token (persistent grid-stride loop over this rank's local tokens);
  - each token multicast-stores (dispatch) / multimem.ld_reduce-loads (combine) to/from ITS OWN
    group's multicast VA. The ONLY routing-dependent per-token value is tok_size_idx (which of the
    NSZ nested groups it picked); the kernel gathers the actual VA from the per-size menus mc_ptrs_*
    by that index. Tokens on one rank targeting different-size groups run in the same kernel -- no
    per-size passes.
  - the buffer row is the COMPACTED global row = rank_token_offset + t (t = the token loop index),
    computed IN-KERNEL from the once-per-step scalar rank_token_offset -- exactly like NVLS's
    AllGather-V (global_offsets = rank_token_offset + token_offset). No per-token row array.
  - a SINGLE global (size-P) symm_mem_sync barrier. A size-P barrier is a correct superset of every
    nested subgroup's barrier (each subgroup is a subset of all P ranks), so one global sync safely
    orders all the per-token multicasts. The grid is min(per_rank_cap, MAX_BLOCKS), identical on
    every rank, so all CTAs pair up by block_id.

The four transport buffers (hidden bf16, routing int64, probs fp32, rsv bf16) are the SAME allocations
NVLS uses; vmcast just rendezvouses each over every nested group to get one aliased multicast VA per
group (the mc_ptrs_* menus). Byte movement matches flat NVLS.
"""
from unittest.mock import MagicMock

import torch

from ._compat import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

from .barrier import symm_mem_sync
from .multimem_asm import ld_64, ld_128, st_64, st_128
from .utils import is_device_nvls_capable, sync_threads

MAX_NUM_BLOCKS = 148


@triton.jit
def _vmcast_size_kernel(
    in_routing_ptr, size_idx_ptr, n_tokens,
    K: tl.constexpr, EPR: tl.constexpr, RANK: tl.constexpr,
    MIN_GROUP: tl.constexpr, NSZ: tl.constexpr, BLOCK: tl.constexpr, KK: tl.constexpr,
):
    """Per-token group-size selection for the ONE-BUFFER design. Reduce the token's K experts to the
    dest-rank span [lo,hi] (source RANK forced in), then emit its group-size index size_idx = smallest
    aligned pow2 block spanning it (bit trick: diff=lo^hi -> smallest i with MIN_GROUP<<i > diff).
    That single index is the only routing-dependent per-token value; dispatch/combine gather the actual
    VAs from the per-size menus (mc_ptrs_*) by it. No VA gather / atomic / counts here."""
    pid = tl.program_id(0)
    if pid * BLOCK >= n_tokens:          # whole block is past the valid tokens (mask all-0): skip an over-provisioned grid
        return
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tokens
    rp = in_routing_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    si_out = size_idx_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    # Vectorized: load the whole [BLOCK, K] routing tile in ONE coalesced 2D load (KK = K padded up to a
    # power of 2 for tl.arange; pad columns read RANK*EPR -> dest rank RANK, harmless to the span), then
    # reduce experts -> dest ranks and take the min/max dest-rank over the token's K experts.
    kk = tl.arange(0, KK)
    col = kk < K
    e = tl.load(rp + offs[:, None] * K + kk[None, :], mask=mask[:, None] & col[None, :], other=RANK * EPR)
    d = e // EPR                                       # [BLOCK, K] dest ranks
    lo = tl.minimum(tl.min(d, axis=1), RANK)           # smallest dest rank (source RANK folded in)
    hi = tl.maximum(tl.max(d, axis=1), RANK)           # largest  dest rank (source RANK folded in)
    diff = lo ^ hi
    size_idx = tl.full((BLOCK,), NSZ - 1, tl.int32)
    for i in range(NSZ - 1, -1, -1):
        size_idx = tl.where(diff < (MIN_GROUP << i), i, size_idx)
    tl.store(si_out + offs, size_idx, mask=mask)


def vmcast_compute_size(in_routing, size_idx, epr, rank, min_group, nsz):
    """Fill size_idx [n] int32 = each token's group-size index (its choice among the NSZ nested groups).
    in_routing [n,K] int64. Dispatch/combine gather the VAs from the menus by this index."""
    assert HAVE_TRITON
    n, K = in_routing.shape
    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK, 1, 1)
    _vmcast_size_kernel[grid](
        in_routing.data_ptr(), size_idx.data_ptr(),
        n, K=K, EPR=epr, RANK=rank, MIN_GROUP=min_group, NSZ=nsz, BLOCK=BLOCK,
        KK=triton.next_power_of_2(K),
    )


@triton.jit
def _fused_dispatch_kernel(
    in_h_ptr, in_r_ptr, in_p_ptr,
    mc_ptrs_h_ptr, mc_ptrs_r_ptr, mc_ptrs_p_ptr,      # [NSZ] int64 per-size VA menus
    tok_size_idx_ptr,                                 # [n] int32 per-token group index
    rank_token_offset_ptr,                            # scalar int32: this rank's compacted-row base (NVLS)
    signal_pad_ptrs,
    n_tokens,
    NPT_H: tl.constexpr,
    NPT_R: tl.constexpr,
    NPT_P: tl.constexpr,
    BITS_R: tl.constexpr,
    BITS_P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Per-token variable-multicast all-gather (dispatch). One CTA per token, then a global barrier.

    Each token multicast-stores hidden(128b)/routing(BITS_R)/probs(BITS_P) into its own group's
    buffer at row (rank_token_offset + t), reading this rank's local input row t directly (no compaction).
    """
    pid = tl.program_id(axis=0)

    # Required Triton-3.6 fix: widen raw pointer int args to i64 (tt.int_to_ptr needs i64).
    in_h_ptr = in_h_ptr.to(tl.int64)
    in_r_ptr = in_r_ptr.to(tl.int64)
    in_p_ptr = in_p_ptr.to(tl.int64)
    mc_ptrs_h_ptr = mc_ptrs_h_ptr.to(tl.int64)
    mc_ptrs_r_ptr = mc_ptrs_r_ptr.to(tl.int64)
    mc_ptrs_p_ptr = mc_ptrs_p_ptr.to(tl.int64)
    tok_size_idx_ptr = tok_size_idx_ptr.to(tl.int64)
    rank_token_offset_ptr = rank_token_offset_ptr.to(tl.int64)

    tid = tl.arange(0, BLOCK_SIZE)
    num_prog = tl.num_programs(axis=0)
    offset = tl.load(rank_token_offset_ptr.to(tl.pointer_type(tl.int32))).to(tl.int64)  # NVLS: scalar, once

    for t in range(pid, n_tokens, num_prog):
        slot = offset + t                                                   # compacted row = offset + t
        sidx = tl.load(tok_size_idx_ptr.to(tl.pointer_type(tl.int32)) + t)   # token's group index (menu idx)

        # --- hidden (128-bit) ---  gather this size's hidden-buffer VA from the menu
        mc_h = tl.load(mc_ptrs_h_ptr.to(tl.pointer_type(tl.int64)) + sidx).to(tl.pointer_type(tl.uint64))
        src_h = in_h_ptr.to(tl.pointer_type(tl.uint64))
        for co in range(0, NPT_H, BLOCK_SIZE):
            ch = co + tid
            m = ch < NPT_H
            (x, y, z, w) = ld_128(src_h + (t * NPT_H + ch) * 2, mask=m, multicast_op=False)
            st_128(mc_h + (slot * NPT_H + ch) * 2, x, y, z, w, mask=m, multicast_op=True)

        # --- routing (BITS_R) ---
        mc_r = tl.load(mc_ptrs_r_ptr.to(tl.pointer_type(tl.int64)) + sidx).to(tl.pointer_type(tl.uint64))
        src_r = in_r_ptr.to(tl.pointer_type(tl.uint64))
        for co in range(0, NPT_R, BLOCK_SIZE):
            ch = co + tid
            m = ch < NPT_R
            if BITS_R == 128:
                (x, y, z, w) = ld_128(src_r + (t * NPT_R + ch) * 2, mask=m, multicast_op=False)
                st_128(mc_r + (slot * NPT_R + ch) * 2, x, y, z, w, mask=m, multicast_op=True)
            else:
                (x, y) = ld_64(src_r + (t * NPT_R + ch), mask=m)
                st_64(mc_r + (slot * NPT_R + ch), x, y, mask=m, multicast_op=True)

        # --- probs (BITS_P) ---
        mc_p = tl.load(mc_ptrs_p_ptr.to(tl.pointer_type(tl.int64)) + sidx).to(tl.pointer_type(tl.uint64))
        src_p = in_p_ptr.to(tl.pointer_type(tl.uint64))
        for co in range(0, NPT_P, BLOCK_SIZE):
            ch = co + tid
            m = ch < NPT_P
            if BITS_P == 128:
                (x, y, z, w) = ld_128(src_p + (t * NPT_P + ch) * 2, mask=m, multicast_op=False)
                st_128(mc_p + (slot * NPT_P + ch) * 2, x, y, z, w, mask=m, multicast_op=True)
            else:
                (x, y) = ld_64(src_p + (t * NPT_P + ch), mask=m)
                st_64(mc_p + (slot * NPT_P + ch), x, y, mask=m, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=True, hasSubsequentMemAccess=True,
    )


@triton.jit
def _fused_combine_kernel(
    out_ptr,
    mc_ptrs_v_ptr, tok_size_idx_ptr,                 # [NSZ] rsv-VA menu + [n] per-token group index
    rank_token_offset_ptr,                           # scalar int32: compacted-row base (NVLS)
    signal_pad_ptrs,
    n_tokens,
    NPT_H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    REDUCE_F32: tl.constexpr,
):
    """Per-token variable-multicast reduce-scatter (combine). Global barrier, then one CTA per token.

    Each token multimem.ld_reduce-loads its row ((rank_token_offset + t)) from its own group's RSv buffer --
    summing over that group's members -- and writes the result to local out[t]."""
    pid = tl.program_id(axis=0)
    out_ptr = out_ptr.to(tl.int64)
    mc_ptrs_v_ptr = mc_ptrs_v_ptr.to(tl.int64)
    tok_size_idx_ptr = tok_size_idx_ptr.to(tl.int64)
    rank_token_offset_ptr = rank_token_offset_ptr.to(tl.int64)

    # Wait for all ranks to have written their expert outputs before any read.
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=False, hasSubsequentMemAccess=False,
    )
    sync_threads()

    tid = tl.arange(0, BLOCK_SIZE)
    num_prog = tl.num_programs(axis=0)
    dst = out_ptr.to(tl.pointer_type(tl.uint64))
    offset = tl.load(rank_token_offset_ptr.to(tl.pointer_type(tl.int32))).to(tl.int64)  # NVLS: scalar, once

    for t in range(pid, n_tokens, num_prog):
        slot = offset + t
        sidx = tl.load(tok_size_idx_ptr.to(tl.pointer_type(tl.int32)) + t)
        mc_v = tl.load(mc_ptrs_v_ptr.to(tl.pointer_type(tl.int64)) + sidx).to(tl.pointer_type(tl.uint64))
        for co in range(0, NPT_H, BLOCK_SIZE):
            ch = co + tid
            m = ch < NPT_H
            (x, y, z, w) = ld_128(
                mc_v + (slot * NPT_H + ch) * 2, mask=m, multicast_op=True, reduce_f32=REDUCE_F32
            )
            st_128(dst + (t * NPT_H + ch) * 2, x, y, z, w, mask=m, multicast_op=False)


def _bits_npt(row_elems, elem_bytes):
    """(bits, numel_per_token) for a row of `row_elems` elements of `elem_bytes` each."""
    row_bytes = row_elems * elem_bytes
    assert row_bytes % 8 == 0, f"row {row_bytes}B not 8-byte aligned"
    bits = 128 if row_bytes % 16 == 0 else 64
    numel_per_thread = bits // (elem_bytes * 8)
    return bits, (row_elems + numel_per_thread - 1) // numel_per_thread


def fused_vmcast_dispatch(
    in_h, in_r, in_p, mc_ptrs_h, mc_ptrs_r, mc_ptrs_p, tok_size_idx, rank_token_offset,
    n_tokens, signal_pad_ptrs, rank, world_size, per_rank_cap,
    max_num_blocks=MAX_NUM_BLOCKS,
):
    """Fused variable-multicast dispatch. in_* are this rank's local [n, *] inputs; mc_ptrs_* are the
    [NSZ] per-size VA menus (one per buffer); tok_size_idx [n] picks each token's group; rank_token_offset
    (scalar) sets the compacted row = offset + t. Each CTA gathers its VA = mc_ptrs[tok_size_idx[t]].
    One launch, one global barrier."""
    assert HAVE_TRITON, "Triton required for fused vmcast dispatch."
    assert is_device_nvls_capable(in_h.device), "fused vmcast needs SM>=9 NVLink."
    H, K = in_h.shape[1], in_r.shape[1]
    _, npt_h = _bits_npt(H, in_h.element_size())          # bf16 -> always 128-bit
    bits_r, npt_r = _bits_npt(K, in_r.element_size())     # int64
    bits_p, npt_p = _bits_npt(K, in_p.element_size())     # fp32
    block_size = min(triton.next_power_of_2(max(npt_h, npt_r, npt_p)), 1024)
    num_warps = max(1, block_size // 32)
    num_blocks = min(per_rank_cap, max_num_blocks)
    _fused_dispatch_kernel[(num_blocks, 1, 1)](
        in_h.data_ptr(), in_r.data_ptr(), in_p.data_ptr(),
        mc_ptrs_h.data_ptr(), mc_ptrs_r.data_ptr(), mc_ptrs_p.data_ptr(),
        tok_size_idx.data_ptr(), rank_token_offset.data_ptr(),
        signal_pad_ptrs,
        n_tokens,
        NPT_H=npt_h, NPT_R=npt_r, NPT_P=npt_p,
        BITS_R=bits_r, BITS_P=bits_p,
        BLOCK_SIZE=block_size,
        RANK=rank, WORLD_SIZE=world_size,
        num_warps=num_warps,
    )


def fused_vmcast_combine(
    out, mc_ptrs_v, tok_size_idx, rank_token_offset, n_tokens, signal_pad_ptrs, rank, world_size, per_rank_cap,
    max_num_blocks=MAX_NUM_BLOCKS,
):
    """Fused variable-multicast combine. mc_ptrs_v [NSZ] = the rsv-VA menu; each CTA gathers its VA =
    mc_ptrs_v[tok_size_idx[t]], ld_reduces row (rank_token_offset + t) over that group (bf16, f32 accum) into
    out[t]. Global barrier first."""
    assert HAVE_TRITON, "Triton required for fused vmcast combine."
    H = out.shape[1]
    _, npt_h = _bits_npt(H, out.element_size())
    block_size = min(triton.next_power_of_2(npt_h), 1024)
    num_warps = max(1, block_size // 32)
    num_blocks = min(per_rank_cap, max_num_blocks)
    reduce_f32 = out.dtype == torch.float32
    _fused_combine_kernel[(num_blocks, 1, 1)](
        out.data_ptr(),
        mc_ptrs_v.data_ptr(), tok_size_idx.data_ptr(), rank_token_offset.data_ptr(),
        signal_pad_ptrs,
        n_tokens,
        NPT_H=npt_h,
        BLOCK_SIZE=block_size,
        RANK=rank, WORLD_SIZE=world_size, REDUCE_F32=reduce_f32,
        num_warps=num_warps,
    )
