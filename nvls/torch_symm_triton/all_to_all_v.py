# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Variable-count all-to-all dispatch / combine over NVLink (unicast push, pull combine).

Mirrors the AGV-V / RSV-V kernels in variable_collectives.py and uses the same per-step
metadata (valid_tokens, rank_token_offset, ep_max_tokens) from metadata.py.

Dispatch
  * Same logical/physical indexing as AGV-V: source rank s writes its local token t to row
    rank_token_offset[s] + t of the [global_max, *] symmetric buffers.
  * Hidden rows are UNICAST, only to the distinct destination ranks of the token
    (dst = expert_id // num_local_experts, deduplicated, -1 entries skipped), through the peer
    pointers in buffer_ptrs_dev. Each row is read from HBM once and stored D times.
  * Routing ids and probs stay MULTICAST to every rank (as in AGV-V), so every rank has a
    complete, fresh routing map for rows [0, valid_tokens). Rows of the hidden buffer that were
    not written this layer ("holes") are exactly the rows with no local expert, which the expert
    kernels already skip.
  * Same per-CTA early exit on ep_max_tokens and trailing per-CTA-pair barrier as AGV-V.

Combine
  * Pull: each source rank reads, for each of its local tokens t, row rank_token_offset + t of
    the bf16 partial-output buffer on each of t's destination ranks (recomputed from its own
    routing), sums the partials in fp32 in a fixed per-token order, and stores bf16 locally.
  * Leading per-CTA-pair barrier with release/acquire semantics (plain peer loads, unlike
    multimem.ld_reduce), so every destination's partials are visible before any read.

Destinations are enumerated by walking the token's top-k entries with a "sent" bitmask, so the
work scales with top-k rather than the EP size (WORLD_SIZE <= 64). Peer stores and loads are
predicated rather than branched, so the per-destination accesses stay in straight-line code.

Buffer reuse follows the AGV-V / RSV-V barrier chain: a source cannot re-dispatch into a
destination before that destination's experts finished (combine barrier), and a destination
cannot overwrite its partials before every source finished pulling (next dispatch barrier).
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

try:
    from torch._C._distributed_c10d import _SymmetricMemory
except ImportError:
    _SymmetricMemory = MagicMock()

from .barrier import symm_mem_sync
from .multimem_asm import ld_64, ld_128, st_64, st_128
from .utils import is_device_nvls_capable, sync_threads

_MAX_BLOCK_SIZE = 1024
_WARP_SIZE = 32
_MAX_WORLD_SIZE = 64  # destinations are tracked in an int64 bitmask


@triton.jit
def _ld_128_or_zero(ptr, mask):
    """128-bit load into four 32-bit registers that are zero when `mask` is false.

    Unlike multimem_asm.ld_128 (which branches around the load), the load is predicated, so
    consecutive loads to different peers stay in one basic block and can all be in flight.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            mov.u32 $2, 0;
            mov.u32 $3, 0;
            @%p0 ld.global.v4.u32 {$0, $1, $2, $3}, [$4];
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16_lo_to_f32(packed):
    """Low bf16 of a bf16x2 word (element 2i) as fp32 (exact)."""
    return (packed << 16).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_hi_to_f32(packed):
    """High bf16 of a bf16x2 word (element 2i+1) as fp32 (exact)."""
    return ((packed >> 16) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_bf16x2(lo, hi):
    """Round two fp32 values to bf16 (round-to-nearest-even) and pack them as bf16x2."""
    lo_bits = lo.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    hi_bits = hi.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    return lo_bits | (hi_bits << 16)


@triton.jit
def _next_destination(
    routing_ids, token_offset, sent, K: tl.constexpr, TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
):
    """Top-k entry K of a token -> (dst_rank, first_entry_for_dst, updated sent bitmask).

    The -1 test must be on the expert id: -1 // NUM_LOCAL_EXPERTS truncates to 0 (rank 0).
    """
    expert = tl.load(routing_ids + token_offset * TOPK + K)
    valid = expert >= 0
    dst = tl.where(valid, expert // NUM_LOCAL_EXPERTS, 0)
    first = valid & (((sent >> dst) & 1) == 0)
    sent = sent | tl.where(first, tl.full((), 1, tl.int64) << dst, 0)
    return dst, first, sent


@triton.jit
def _multicast_row(
    src_ptr,
    mc_ptr,
    byte_offset,
    token_offset,
    global_row,
    tid,
    PACKS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """AGV-V copy of one row: local row `token_offset` -> multicast row `global_row`."""
    for channel_offset in range(0, PACKS, BLOCK_SIZE):
        cols = channel_offset + tid
        mask = cols < PACKS
        local_offsets = token_offset * PACKS + cols
        global_offsets = global_row * PACKS + cols
        if BITS == 128:
            mc_ptrs = mc_ptr.to(tl.pointer_type(tl.uint64)) + byte_offset // 8 + global_offsets * 2
            src_ptrs = src_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
            (x, y, z, w) = ld_128(src_ptrs, mask=mask, multicast_op=False)
            st_128(mc_ptrs, x, y, z, w, mask=mask, multicast_op=True)
        else:
            mc_ptrs = mc_ptr.to(tl.pointer_type(tl.uint64)) + byte_offset // 8 + global_offsets
            src_ptrs = src_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets
            (x, y) = ld_64(src_ptrs, mask=mask)
            st_64(mc_ptrs, x, y, mask=mask, multicast_op=True)


@triton.jit
def _a2a_dispatch_v_kernel(
    hidden_ptr,
    hidden_peer_ptrs,
    hidden_byte_offset,
    routing_ptr,
    routing_mc_ptr,
    routing_byte_offset,
    probs_ptr,
    probs_mc_ptr,
    probs_byte_offset,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    HIDDEN_PACKS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    ROUTING_PACKS: tl.constexpr,
    ROUTING_BITS: tl.constexpr,
    PROBS_PACKS: tl.constexpr,
    PROBS_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """All-to-all-V dispatch. One CTA per token (persistent over local_tokens).

    Args:
        hidden_ptr: local hidden [local_tokens, HIDDEN_PACKS * 16 B] (raw pointer int).
        hidden_peer_ptrs: device array of WORLD_SIZE uint64 base addresses of every rank's
            hidden receive buffer (the handle's buffer_ptrs_dev).
        hidden_byte_offset: byte offset of the hidden tensor inside its symmetric buffer.
        routing_ptr / probs_ptr: local [local_tokens, TOPK] int64 ids / fp32 probs.
        routing_mc_ptr / probs_mc_ptr: multicast pointers of their receive buffers.
        signal_pad_ptrs: signal pads of the hidden buffer (one barrier covers all three).
        local_tokens: this rank's token count.
        rank_token_offset_ptr / ep_max_tokens_ptr: per-step metadata (see AGV-V).
        HIDDEN_PACKS: 16-byte packs per hidden row.
        ROUTING_PACKS / PROBS_PACKS: packs per routing / probs row, of ROUTING_BITS / PROBS_BITS.
        BLOCK_SIZE: threads per CTA (>= every *_PACKS up to the 1024 cap; wider rows loop).
    """
    pid = tl.program_id(axis=0)

    # Same rank-consistent early exit as AGV-V: CTAs >= ep_max_tokens have no token on any rank.
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Triton-3.6 fix (as in variable_collectives.py): widen raw pointer ints to i64.
    hidden_ptr = hidden_ptr.to(tl.int64)
    hidden_peer_ptrs = hidden_peer_ptrs.to(tl.int64)
    routing_ptr = routing_ptr.to(tl.int64)
    routing_mc_ptr = routing_mc_ptr.to(tl.int64)
    probs_ptr = probs_ptr.to(tl.int64)
    probs_mc_ptr = probs_mc_ptr.to(tl.int64)

    hidden_u64 = hidden_ptr.to(tl.pointer_type(tl.uint64))
    peer_table = hidden_peer_ptrs.to(tl.pointer_type(tl.uint64))
    routing_ids = routing_ptr.to(tl.pointer_type(tl.int64))

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        # Same row on every destination: AGV-V indexing.
        global_row = (rank_token_offset + token_offset).to(tl.int64)

        # Hidden: read the row from HBM once, store it to each distinct destination rank.
        for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
            cols = channel_offset + tid
            mask = cols < HIDDEN_PACKS
            (x, y, z, w) = ld_128(
                hidden_u64 + (token_offset * HIDDEN_PACKS + cols) * 2, mask=mask, multicast_op=False
            )
            sent = tl.full((), 0, tl.int64)
            for k in tl.static_range(TOPK):
                dst, first, sent = _next_destination(
                    routing_ids, token_offset, sent, k, TOPK, NUM_LOCAL_EXPERTS
                )
                peer = tl.load(peer_table + dst).to(tl.pointer_type(tl.uint64))
                st_128(
                    peer + hidden_byte_offset // 8 + (global_row * HIDDEN_PACKS + cols) * 2,
                    x, y, z, w,
                    mask=mask & first,
                    multicast_op=False,
                )

        # Routing ids and probs: multicast to every rank, exactly as AGV-V does, so each rank can
        # tell received rows from holes.
        _multicast_row(
            routing_ptr, routing_mc_ptr, routing_byte_offset, token_offset, global_row, tid,
            ROUTING_PACKS, ROUTING_BITS, BLOCK_SIZE,
        )
        _multicast_row(
            probs_ptr, probs_mc_ptr, probs_byte_offset, token_offset, global_row, tid,
            PROBS_PACKS, PROBS_BITS, BLOCK_SIZE,
        )

    # The release in the barrier covers this CTA's unicast and multicast stores.
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


@triton.jit
def _a2a_combine_v_kernel(
    output_ptr,
    partial_peer_ptrs,
    partial_byte_offset,
    routing_ptr,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    HIDDEN_PACKS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """All-to-all-V combine (pull). One CTA per token (persistent over local_tokens).

    Args:
        output_ptr: local bf16 output [local_tokens, HIDDEN_PACKS * 8] (raw pointer int).
        partial_peer_ptrs: device array of WORLD_SIZE uint64 base addresses of every rank's bf16
            partial-output buffer [global_max, HIDDEN_PACKS * 8] (the handle's buffer_ptrs_dev).
        partial_byte_offset: byte offset of the partial tensor inside its symmetric buffer.
        routing_ptr: this rank's LOCAL routing [local_tokens, TOPK] int64, used to recompute
            which ranks each token was dispatched to.
        signal_pad_ptrs: signal pads of the partial buffer.
        HIDDEN_PACKS: 16-byte packs (8 bf16) per row.
    """
    pid = tl.program_id(axis=0)

    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Every destination's partials (written by earlier kernels) must be visible before any pull:
    # release on the signal side, acquire on the wait side (these are plain peer loads).
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )
    sync_threads()

    # Triton-3.6 fix (as in variable_collectives.py): widen raw pointer ints to i64.
    output_ptr = output_ptr.to(tl.int64)
    partial_peer_ptrs = partial_peer_ptrs.to(tl.int64)
    routing_ptr = routing_ptr.to(tl.int64)

    output_u64 = output_ptr.to(tl.pointer_type(tl.uint64))
    peer_table = partial_peer_ptrs.to(tl.pointer_type(tl.uint64))
    routing_ids = routing_ptr.to(tl.pointer_type(tl.int64))

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_row = (rank_token_offset + token_offset).to(tl.int64)

        for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
            cols = channel_offset + tid
            mask = cols < HIDDEN_PACKS
            # Eight fp32 accumulators: (lo, hi) halves of each of the four bf16x2 words.
            acc_xl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_xh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_yl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_yh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_zl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_zh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_wl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_wh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            sent = tl.full((), 0, tl.int64)
            # Fixed per-token order (top-k entry order of first occurrences): deterministic.
            for k in tl.static_range(TOPK):
                dst, first, sent = _next_destination(
                    routing_ids, token_offset, sent, k, TOPK, NUM_LOCAL_EXPERTS
                )
                peer = tl.load(peer_table + dst).to(tl.pointer_type(tl.uint64))
                (x, y, z, w) = _ld_128_or_zero(
                    peer + partial_byte_offset // 8 + (global_row * HIDDEN_PACKS + cols) * 2,
                    mask & first,
                )
                acc_xl += _bf16_lo_to_f32(x)
                acc_xh += _bf16_hi_to_f32(x)
                acc_yl += _bf16_lo_to_f32(y)
                acc_yh += _bf16_hi_to_f32(y)
                acc_zl += _bf16_lo_to_f32(z)
                acc_zh += _bf16_hi_to_f32(z)
                acc_wl += _bf16_lo_to_f32(w)
                acc_wh += _bf16_hi_to_f32(w)

            st_128(
                output_u64 + (token_offset * HIDDEN_PACKS + cols) * 2,
                _f32_to_bf16x2(acc_xl, acc_xh),
                _f32_to_bf16x2(acc_yl, acc_yh),
                _f32_to_bf16x2(acc_zl, acc_zh),
                _f32_to_bf16x2(acc_wl, acc_wh),
                mask=mask,
                multicast_op=False,
            )
    # No trailing barrier: the next dispatch's barrier protects the partial buffer (see module
    # docstring).


def _row_params(t: torch.Tensor):
    """(bits, packs per row) for a 2-D tensor row: 128-bit packs if 16-B aligned, else 64-bit."""
    row_bytes = t.shape[1] * t.element_size()
    assert row_bytes % 8 == 0, f"row of {row_bytes} bytes is not 8-byte aligned"
    bits = 128 if row_bytes % 16 == 0 else 64
    return bits, row_bytes // (bits // 8)


def a2a_dispatch_v(
    output_hidden: torch.Tensor,
    output_routing: torch.Tensor,
    output_probs: torch.Tensor,
    input_hidden: torch.Tensor,
    input_routing: torch.Tensor,
    input_probs: torch.Tensor,
    symm_mem_hdl_hidden: _SymmetricMemory,
    symm_mem_hdl_routing: _SymmetricMemory,
    symm_mem_hdl_probs: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    num_local_experts: int,
    hidden_byte_offset: int = 0,
    routing_byte_offset: int = 0,
    probs_byte_offset: int = 0,
    **kwargs,
):
    """All-to-all-V dispatch: unicast hidden rows to their destination ranks, multicast routing
    ids and probs to every rank, all at AGV-V row rank_token_offset + t.

    Args:
        output_hidden / output_routing / output_probs: symmetric receive buffers
            [global_tokens, *]. Hidden rows with no local expert are left untouched (holes).
        input_hidden: local [local_tokens, hidden] (16-byte rows, e.g. bf16 with hidden % 8 == 0).
        input_routing: local [local_tokens, topk] int64 expert ids (-1 = no expert).
        input_probs: local [local_tokens, topk] probs.
        symm_mem_hdl_*: symmetric memory handles of the three receive buffers (same EP group).
        rank_token_offset / ep_max_tokens: per-step metadata tensors (see AGV-V).
        per_rank_max_tokens: static; grid = min(per_rank_max_tokens, max_num_blocks).
        num_local_experts: experts per rank; dst rank = expert_id // num_local_experts.
    """
    assert HAVE_TRITON, "Triton is required for a2a_dispatch_v."
    for name, inp, out in (
        ("hidden", input_hidden, output_hidden),
        ("routing", input_routing, output_routing),
        ("probs", input_probs, output_probs),
    ):
        assert inp.ndim == 2 and out.ndim == 2, f"{name}: tensors must be 2-D"
        assert inp.shape[1] == out.shape[1], f"{name}: row width mismatch"
        assert inp.is_contiguous(), f"{name}: input must be contiguous"
    local_tokens = input_hidden.shape[0]
    assert input_routing.shape[0] == local_tokens and input_probs.shape[0] == local_tokens
    assert input_routing.dtype == torch.int64, "routing ids must be int64"
    assert input_routing.shape == input_probs.shape, "routing and probs must both be [n, topk]"
    assert is_device_nvls_capable(input_hidden.device), "needs a Hopper+ GPU with NVLink"
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."
    hdls = (symm_mem_hdl_hidden, symm_mem_hdl_routing, symm_mem_hdl_probs)
    assert len({h.rank for h in hdls}) == 1 and len({h.world_size for h in hdls}) == 1
    world_size = symm_mem_hdl_hidden.world_size
    assert world_size <= _MAX_WORLD_SIZE, f"world size {world_size} > {_MAX_WORLD_SIZE}"

    hidden_bits, hidden_packs = _row_params(input_hidden)
    assert hidden_bits == 128, "hidden rows must be 16-byte aligned"
    routing_bits, routing_packs = _row_params(input_routing)
    probs_bits, probs_packs = _row_params(input_probs)

    max_num_blocks = kwargs.get("max_num_blocks", 148)
    block_size = min(
        max(triton.next_power_of_2(p) for p in (hidden_packs, routing_packs, probs_packs)),
        _MAX_BLOCK_SIZE,
    )
    num_warps = max(1, block_size // _WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _a2a_dispatch_v_kernel[(num_blocks, 1, 1)](
        input_hidden.data_ptr(),
        symm_mem_hdl_hidden.buffer_ptrs_dev,
        hidden_byte_offset,
        input_routing.data_ptr(),
        symm_mem_hdl_routing.multicast_ptr,
        routing_byte_offset,
        input_probs.data_ptr(),
        symm_mem_hdl_probs.multicast_ptr,
        probs_byte_offset,
        symm_mem_hdl_hidden.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        HIDDEN_PACKS=hidden_packs,
        TOPK=input_routing.shape[1],
        NUM_LOCAL_EXPERTS=num_local_experts,
        ROUTING_PACKS=routing_packs,
        ROUTING_BITS=routing_bits,
        PROBS_PACKS=probs_packs,
        PROBS_BITS=probs_bits,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl_hidden.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output_hidden, output_routing, output_probs


def a2a_combine_v(
    output: torch.Tensor,
    partial: torch.Tensor,
    symm_mem_hdl_partial: _SymmetricMemory,
    routing: torch.Tensor,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    num_local_experts: int,
    partial_byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """All-to-all-V combine: for each local token, pull its bf16 partial rows from the ranks it
    was dispatched to (row rank_token_offset + t), sum in fp32, store bf16 into `output`.

    Args:
        output: local [local_tokens, hidden] bf16.
        partial: symmetric [global_tokens, hidden] bf16 partial-output buffer; on each rank only
            rows with a local expert are read.
        symm_mem_hdl_partial: symmetric memory handle of `partial`.
        routing: this rank's local [local_tokens, topk] int64 routing (as passed to dispatch).
        rank_token_offset / ep_max_tokens / per_rank_max_tokens / num_local_experts: as dispatch.
    """
    assert HAVE_TRITON, "Triton is required for a2a_combine_v."
    assert output.ndim == 2 and partial.ndim == 2 and routing.ndim == 2
    assert output.dtype == partial.dtype == torch.bfloat16, "combine supports bf16 partials only"
    assert output.shape[1] == partial.shape[1], "output / partial hidden mismatch"
    assert output.is_contiguous() and routing.is_contiguous()
    assert routing.dtype == torch.int64 and routing.shape[0] == output.shape[0]
    assert is_device_nvls_capable(output.device), "needs a Hopper+ GPU with NVLink"
    world_size = symm_mem_hdl_partial.world_size
    assert world_size <= _MAX_WORLD_SIZE, f"world size {world_size} > {_MAX_WORLD_SIZE}"
    bits, hidden_packs = _row_params(output)
    assert bits == 128, "hidden rows must be 16-byte aligned"

    max_num_blocks = kwargs.get("max_num_blocks", 148)
    block_size = min(triton.next_power_of_2(hidden_packs), _MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // _WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _a2a_combine_v_kernel[(num_blocks, 1, 1)](
        output.data_ptr(),
        symm_mem_hdl_partial.buffer_ptrs_dev,
        partial_byte_offset,
        routing.data_ptr(),
        symm_mem_hdl_partial.signal_pad_ptrs_dev,
        local_tokens=output.shape[0],
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        HIDDEN_PACKS=hidden_packs,
        TOPK=routing.shape[1],
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl_partial.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output
