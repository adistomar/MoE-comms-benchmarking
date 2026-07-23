# Copyright (c) 2026. Fused per-token variable-multicast dispatch/combine for vmcast.
"""
Fused kernels for the variable-size multicast MoE dispatcher (bench_vmcast.py).

The un-fused vmcast prototype issues ONE multimem AllGather-V / ReduceScatter-V collective
PER distinct active group size, run sequentially -> N group barriers per layer. Under a mixed
routing distribution every nested size is populated, so the cost is dominated by
(#active sizes) x (one cross-node barrier each). See the mixed-routing measurement.

These kernels collapse that to ONE launch + ONE global barrier:

  - one CTA per token (persistent grid-stride loop over this rank's local tokens);
  - each token multicast-stores (dispatch) / multimem.ld_reduce-loads (combine) to/from ITS OWN
    group's multicast VA, looked up from a per-token pointer array (tok_mc_*), at its slot in
    that group's buffer (tok_slot). Tokens on the same rank targeting different-size groups are
    handled by the same kernel -- no per-size passes.
  - a SINGLE global (size-P) symm_mem_sync barrier. A size-P barrier is a correct superset of
    every nested subgroup's barrier (each subgroup is a subset of all P ranks), so one global
    sync safely orders all the per-token multicasts. The grid is min(per_rank_cap, MAX_BLOCKS),
    identical on every rank, so all CTAs pair up by block_id (no ep_max early-exit needed).

Layout matches the per-group AGv/RSv exactly: token t (source rank r, group g) lives at row
tok_slot[t] = rank_token_offset_g(r) + j of group g's [gcap, H] buffer, so the vendored mask
(_mask_into_rsv) and this combine read the same slots. Byte movement matches flat NVLS (all
three tensors: hidden bf16, routing int64, probs fp32).
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
def _vmcast_layout_kernel(
    in_routing_ptr,                                   # [n, K] int64 topk expert ids
    mc_h_arr, mc_r_arr, mc_p_arr, mc_v_arr,           # [NSZ] int64 per-size group multicast VAs
    counts_ptr,                                       # [NSZ] int32 atomic accumulator (pre-zeroed)
    size_idx_ptr, intra_ptr,                          # [n] int32 out
    tok_mc_h_ptr, tok_mc_r_ptr, tok_mc_p_ptr, tok_mc_v_ptr,   # [n] int64 out
    n_tokens,
    K: tl.constexpr, EPR: tl.constexpr, RANK: tl.constexpr,
    MIN_GROUP: tl.constexpr, NSZ: tl.constexpr, BLOCK: tl.constexpr,
):
    """Per-token group layout in ONE launch (one lane per token, grid-stride). Replaces ~15 tiny torch
    ops. For each token: reduce its K experts to the dest-rank span [lo,hi] (source RANK forced in),
    tok_size = smallest aligned pow2 block spanning it (via diff=lo^hi: smallest MIN_GROUP<<i > diff),
    take an atomic intra-rank position within that size's bucket (also fills the per-size count), and
    gather the size's four multicast VAs. Slot = rank_token_offset (metadata) + intra is added later."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tokens

    # Widen raw int pointer args to i64 then type them (Triton-3.6: tt.int_to_ptr needs i64).
    rp = in_routing_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mch = mc_h_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mcr = mc_r_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mcp = mc_p_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mcv = mc_v_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    cnt = counts_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    si_out = size_idx_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    in_out = intra_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    th = tok_mc_h_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    tr = tok_mc_r_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    tp = tok_mc_p_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    tv = tok_mc_v_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))

    lo = tl.full((BLOCK,), RANK, tl.int64)
    hi = tl.full((BLOCK,), RANK, tl.int64)
    for k in range(K):
        e = tl.load(rp + offs * K + k, mask=mask, other=RANK * EPR)
        d = e // EPR
        lo = tl.minimum(lo, d)
        hi = tl.maximum(hi, d)
    diff = lo ^ hi                                     # 0 iff single dest rank == source
    size_idx = tl.full((BLOCK,), NSZ - 1, tl.int32)    # size-P always fits
    for i in range(NSZ - 1, -1, -1):                   # keep the SMALLEST fitting size
        size_idx = tl.where(diff < (MIN_GROUP << i), i, size_idx)

    # atomic per-size counter: returns this token's 0-based position in its bucket AND accumulates count.
    intra = tl.atomic_add(cnt + size_idx, 1, mask=mask)

    mh = tl.load(mch + size_idx)
    mr = tl.load(mcr + size_idx)
    mp = tl.load(mcp + size_idx)
    mv = tl.load(mcv + size_idx)
    tl.store(si_out + offs, size_idx, mask=mask)
    tl.store(in_out + offs, intra, mask=mask)
    tl.store(th + offs, mh, mask=mask)
    tl.store(tr + offs, mr, mask=mask)
    tl.store(tp + offs, mp, mask=mask)
    tl.store(tv + offs, mv, mask=mask)


def vmcast_compute_group(in_routing, mc_by_size, counts, size_idx, intra,
                         tok_mc_h, tok_mc_r, tok_mc_p, tok_mc_v,
                         epr, rank, min_group, nsz):
    """Launch the layout kernel. in_routing [n,K] int64; mc_by_size = (h,r,p,v) [NSZ] int64 tensors;
    counts [NSZ] int32 must be pre-zeroed. Fills size_idx/intra [n] int32 and tok_mc_* [n] int64."""
    assert HAVE_TRITON
    n, K = in_routing.shape
    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK, 1, 1)
    _vmcast_layout_kernel[grid](
        in_routing.data_ptr(),
        mc_by_size[0].data_ptr(), mc_by_size[1].data_ptr(),
        mc_by_size[2].data_ptr(), mc_by_size[3].data_ptr(),
        counts.data_ptr(), size_idx.data_ptr(), intra.data_ptr(),
        tok_mc_h.data_ptr(), tok_mc_r.data_ptr(), tok_mc_p.data_ptr(), tok_mc_v.data_ptr(),
        n, K=K, EPR=epr, RANK=rank, MIN_GROUP=min_group, NSZ=nsz, BLOCK=BLOCK,
    )


@triton.jit
def _vmcast_layout_slotless_kernel(
    in_routing_ptr,
    mc_h_arr, mc_r_arr, mc_p_arr, mc_v_arr,           # [NSZ] int64 per-size group multicast VAs
    size_idx_ptr,                                     # [n] int32 out (debug/hist only)
    tok_mc_h_ptr, tok_mc_r_ptr, tok_mc_p_ptr, tok_mc_v_ptr,   # [n] int64 out
    n_tokens,
    K: tl.constexpr, EPR: tl.constexpr, RANK: tl.constexpr,
    MIN_GROUP: tl.constexpr, NSZ: tl.constexpr, BLOCK: tl.constexpr,
):
    """Slotless per-token layout for the ONE-BUFFER design: tok_size (bit trick on the dest-rank span)
    -> gather the size's four multicast VAs. NO atomic / counts / intra -- the global row is static
    (rank*cap + i), so there's no compaction to scan. Removes the cross-token atomic serialization."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tokens
    rp = in_routing_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mch = mc_h_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mcr = mc_r_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mcp = mc_p_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    mcv = mc_v_arr.to(tl.int64).to(tl.pointer_type(tl.int64))
    si_out = size_idx_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    th = tok_mc_h_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    tr = tok_mc_r_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    tp = tok_mc_p_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
    tv = tok_mc_v_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))

    lo = tl.full((BLOCK,), RANK, tl.int64)
    hi = tl.full((BLOCK,), RANK, tl.int64)
    for k in range(K):
        e = tl.load(rp + offs * K + k, mask=mask, other=RANK * EPR)
        d = e // EPR
        lo = tl.minimum(lo, d)
        hi = tl.maximum(hi, d)
    diff = lo ^ hi
    size_idx = tl.full((BLOCK,), NSZ - 1, tl.int32)
    for i in range(NSZ - 1, -1, -1):
        size_idx = tl.where(diff < (MIN_GROUP << i), i, size_idx)

    tl.store(si_out + offs, size_idx, mask=mask)
    tl.store(th + offs, tl.load(mch + size_idx), mask=mask)
    tl.store(tr + offs, tl.load(mcr + size_idx), mask=mask)
    tl.store(tp + offs, tl.load(mcp + size_idx), mask=mask)
    tl.store(tv + offs, tl.load(mcv + size_idx), mask=mask)


def vmcast_compute_group_slotless(in_routing, mc_by_size, size_idx,
                                  tok_mc_h, tok_mc_r, tok_mc_p, tok_mc_v,
                                  epr, rank, min_group, nsz):
    """Lean layout for the one-buffer bencher: fills size_idx [n] int32 and tok_mc_* [n] int64.
    No counts/intra/atomic (static slots). in_routing [n,K] int64; mc_by_size = (h,r,p,v) [NSZ] int64."""
    assert HAVE_TRITON
    n, K = in_routing.shape
    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK, 1, 1)
    _vmcast_layout_slotless_kernel[grid](
        in_routing.data_ptr(),
        mc_by_size[0].data_ptr(), mc_by_size[1].data_ptr(),
        mc_by_size[2].data_ptr(), mc_by_size[3].data_ptr(),
        size_idx.data_ptr(),
        tok_mc_h.data_ptr(), tok_mc_r.data_ptr(), tok_mc_p.data_ptr(), tok_mc_v.data_ptr(),
        n, K=K, EPR=epr, RANK=rank, MIN_GROUP=min_group, NSZ=nsz, BLOCK=BLOCK,
    )


@triton.jit
def _fused_dispatch_kernel(
    in_h_ptr, in_r_ptr, in_p_ptr,
    tok_mc_h_ptr, tok_mc_r_ptr, tok_mc_p_ptr, tok_slot_ptr,
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
    buffer at row tok_slot[t], reading this rank's local input row t directly (no compaction).
    """
    pid = tl.program_id(axis=0)

    # Required Triton-3.6 fix: widen raw pointer int args to i64 (tt.int_to_ptr needs i64).
    in_h_ptr = in_h_ptr.to(tl.int64)
    in_r_ptr = in_r_ptr.to(tl.int64)
    in_p_ptr = in_p_ptr.to(tl.int64)
    tok_mc_h_ptr = tok_mc_h_ptr.to(tl.int64)
    tok_mc_r_ptr = tok_mc_r_ptr.to(tl.int64)
    tok_mc_p_ptr = tok_mc_p_ptr.to(tl.int64)
    tok_slot_ptr = tok_slot_ptr.to(tl.int64)

    tid = tl.arange(0, BLOCK_SIZE)
    num_prog = tl.num_programs(axis=0)

    for t in range(pid, n_tokens, num_prog):
        slot = tl.load(tok_slot_ptr.to(tl.pointer_type(tl.int32)) + t).to(tl.int64)

        # --- hidden (128-bit) ---
        mc_h = tl.load(tok_mc_h_ptr.to(tl.pointer_type(tl.int64)) + t).to(tl.pointer_type(tl.uint64))
        src_h = in_h_ptr.to(tl.pointer_type(tl.uint64))
        for co in range(0, NPT_H, BLOCK_SIZE):
            ch = co + tid
            m = ch < NPT_H
            (x, y, z, w) = ld_128(src_h + (t * NPT_H + ch) * 2, mask=m, multicast_op=False)
            st_128(mc_h + (slot * NPT_H + ch) * 2, x, y, z, w, mask=m, multicast_op=True)

        # --- routing (BITS_R) ---
        mc_r = tl.load(tok_mc_r_ptr.to(tl.pointer_type(tl.int64)) + t).to(tl.pointer_type(tl.uint64))
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
        mc_p = tl.load(tok_mc_p_ptr.to(tl.pointer_type(tl.int64)) + t).to(tl.pointer_type(tl.uint64))
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
    tok_mc_v_ptr, tok_slot_ptr,
    signal_pad_ptrs,
    n_tokens,
    NPT_H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    REDUCE_F32: tl.constexpr,
):
    """Per-token variable-multicast reduce-scatter (combine). Global barrier, then one CTA per token.

    Each token multimem.ld_reduce-loads its row (tok_slot[t]) from its own group's RSv buffer --
    summing over that group's members -- and writes the result to local out[t]."""
    pid = tl.program_id(axis=0)
    out_ptr = out_ptr.to(tl.int64)
    tok_mc_v_ptr = tok_mc_v_ptr.to(tl.int64)
    tok_slot_ptr = tok_slot_ptr.to(tl.int64)

    # Wait for all ranks to have written their expert outputs before any read.
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=False, hasSubsequentMemAccess=False,
    )
    sync_threads()

    tid = tl.arange(0, BLOCK_SIZE)
    num_prog = tl.num_programs(axis=0)
    dst = out_ptr.to(tl.pointer_type(tl.uint64))

    for t in range(pid, n_tokens, num_prog):
        slot = tl.load(tok_slot_ptr.to(tl.pointer_type(tl.int32)) + t).to(tl.int64)
        mc_v = tl.load(tok_mc_v_ptr.to(tl.pointer_type(tl.int64)) + t).to(tl.pointer_type(tl.uint64))
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
    in_h, in_r, in_p, tok_mc_h, tok_mc_r, tok_mc_p, tok_slot,
    n_tokens, signal_pad_ptrs, rank, world_size, per_rank_cap,
    max_num_blocks=MAX_NUM_BLOCKS,
):
    """Fused variable-multicast dispatch. in_* are this rank's local [n, *] inputs; tok_mc_*/tok_slot
    are per-token int64 group multicast VAs / int32 dest rows. One launch, one global barrier."""
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
        tok_mc_h.data_ptr(), tok_mc_r.data_ptr(), tok_mc_p.data_ptr(), tok_slot.data_ptr(),
        signal_pad_ptrs,
        n_tokens,
        NPT_H=npt_h, NPT_R=npt_r, NPT_P=npt_p,
        BITS_R=bits_r, BITS_P=bits_p,
        BLOCK_SIZE=block_size,
        RANK=rank, WORLD_SIZE=world_size,
        num_warps=num_warps,
    )


def fused_vmcast_combine(
    out, tok_mc_v, tok_slot, n_tokens, signal_pad_ptrs, rank, world_size, per_rank_cap,
    max_num_blocks=MAX_NUM_BLOCKS,
):
    """Fused variable-multicast combine. Reads each token's row from its group's RSv buffer via
    multimem.ld_reduce (bf16 with f32 accumulation) into local out[t]. Global barrier first."""
    assert HAVE_TRITON, "Triton required for fused vmcast combine."
    H = out.shape[1]
    _, npt_h = _bits_npt(H, out.element_size())
    block_size = min(triton.next_power_of_2(npt_h), 1024)
    num_warps = max(1, block_size // 32)
    num_blocks = min(per_rank_cap, max_num_blocks)
    reduce_f32 = out.dtype == torch.float32
    _fused_combine_kernel[(num_blocks, 1, 1)](
        out.data_ptr(),
        tok_mc_v.data_ptr(), tok_slot.data_ptr(),
        signal_pad_ptrs,
        n_tokens,
        NPT_H=npt_h,
        BLOCK_SIZE=block_size,
        RANK=rank, WORLD_SIZE=world_size, REDUCE_F32=reduce_f32,
        num_warps=num_warps,
    )
