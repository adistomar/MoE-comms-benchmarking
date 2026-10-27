# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid dispatch / combine: multicast (AGV-V / RSV-V) or unicast (A2A-V), chosen on device.

The AGV-V / RSV-V kernels (variable_collectives.py) and the A2A-V kernels (all_to_all_v.py)
share the row layout (source rank s writes local token t to row rank_token_offset[s] + t of
[global_max, *] symmetric buffers), the per-step metadata, the pid >= ep_max_tokens early exit,
the grid and the per-CTA-pair barrier. They differ only in how a row moves, so each hybrid
kernel is the A2A-V kernel with one branch on a per-step mode flag that the metadata kernel
writes (metadata.py, COMPUTE_MODES):

Dispatch (flag step_metadata[3])
  * 1: the hidden row is multicast to every rank (multimem.st), as AGV-V.
  * 0: the hidden row is unicast to the token's distinct destination ranks, as A2A-V.
  * Routing ids and probs are multicast in both modes, so every rank has the full routing map
    for rows [0, valid_tokens). The expert reads only rows routed to its rank, and those arrive
    in either mode: the dispatch mode needs no agreement across ranks.

Combine (flag step_metadata[4])
  * 1: multimem.ld_reduce of row rank_token_offset + t over every rank (bf16x2 operands, fp32
    accumulation), as RSV-V. Every rank's partial row must be valid, so ranks without an
    expert for the token must hold a zero row (expert write_unrouted_rows=True).
  * 0: pull the bf16 partial rows from the token's destination ranks and sum in fp32 in a fixed
    order, as A2A-V. Rows of ranks without an expert for the token are never read.
  * The flag may differ per rank, and (with mc_split) per token: the expert always writes zero
    rows for tokens it did not process, so any owner can multicast-reduce any of its tokens.

The flag is read on device, so the mode can change on any step without host sync or graph
recapture; the branch is uniform across the grid. Buffer reuse and barrier pairing do not
depend on the mode (see docs/hybrid_dispatcher_design.md).
"""

from typing import Optional
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

from .all_to_all_v import (
    _MAX_BLOCK_SIZE,
    _MAX_WORLD_SIZE,
    _WARP_SIZE,
    _bf16_hi_to_f32,
    _bf16_lo_to_f32,
    _f32_to_bf16x2,
    _ld_128_or_zero,
    _multicast_row,
    _next_destination,
    _row_params,
)
from .barrier import symm_mem_sync
from .multimem_asm import ld_128, st_128
from .utils import device_num_sms, is_device_nvls_capable, sync_threads


@triton.jit
def _hybrid_dispatch_v_kernel(
    hidden_ptr,
    hidden_mc_ptr,
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
    mode_ptr,
    split_ptr,
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
    HAS_SPLIT: tl.constexpr = False,
):
    """Hybrid dispatch. One CTA per token (persistent over local_tokens).

    Args as _a2a_dispatch_v_kernel, plus:
        hidden_mc_ptr: multicast pointer of the hidden receive buffer (multicast mode).
        mode_ptr: scalar int32, 1 = multicast the hidden row, 0 = unicast it to its
            destination ranks. Read once per launch.
        split_ptr: with HAS_SPLIT, scalar int32 m: in multicast mode only CTAs pid < m
            multicast; the others unicast. Tokens go to CTAs round-robin, so this splits the
            rank's tokens by index (per-rank / per-token mixing is legal, see the module doc).
    """
    pid = tl.program_id(axis=0)

    # Same rank-consistent early exit as AGV-V / A2A-V.
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return
    use_multicast = tl.load(mode_ptr) != 0
    if HAS_SPLIT:
        use_multicast = use_multicast & (pid < tl.load(split_ptr))

    # Triton-3.6 fix (as in variable_collectives.py): widen raw pointer ints to i64.
    hidden_ptr = hidden_ptr.to(tl.int64)
    hidden_mc_ptr = hidden_mc_ptr.to(tl.int64)
    hidden_peer_ptrs = hidden_peer_ptrs.to(tl.int64)
    routing_ptr = routing_ptr.to(tl.int64)
    routing_mc_ptr = routing_mc_ptr.to(tl.int64)
    probs_ptr = probs_ptr.to(tl.int64)
    probs_mc_ptr = probs_mc_ptr.to(tl.int64)

    hidden_u64 = hidden_ptr.to(tl.pointer_type(tl.uint64))
    hidden_mc_u64 = hidden_mc_ptr.to(tl.pointer_type(tl.uint64))
    peer_table = hidden_peer_ptrs.to(tl.pointer_type(tl.uint64))
    routing_ids = routing_ptr.to(tl.pointer_type(tl.int64))

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_row = (rank_token_offset + token_offset).to(tl.int64)

        for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
            cols = channel_offset + tid
            mask = cols < HIDDEN_PACKS
            (x, y, z, w) = ld_128(
                hidden_u64 + (token_offset * HIDDEN_PACKS + cols) * 2, mask=mask, multicast_op=False
            )
            row_u64 = hidden_byte_offset // 8 + (global_row * HIDDEN_PACKS + cols) * 2
            if use_multicast:
                # AGV-V: one store, the switch writes the row to every rank.
                st_128(hidden_mc_u64 + row_u64, x, y, z, w, mask=mask, multicast_op=True)
            else:
                # A2A-V: one store per distinct destination rank.
                sent = tl.full((), 0, tl.int64)
                for k in tl.static_range(TOPK):
                    dst, first, sent = _next_destination(
                        routing_ids, token_offset, sent, k, TOPK, NUM_LOCAL_EXPERTS
                    )
                    peer = tl.load(peer_table + dst).to(tl.pointer_type(tl.uint64))
                    st_128(peer + row_u64, x, y, z, w, mask=mask & first, multicast_op=False)

        # Routing ids and probs: multicast in both modes.
        _multicast_row(
            routing_ptr, routing_mc_ptr, routing_byte_offset, token_offset, global_row, tid,
            ROUTING_PACKS, ROUTING_BITS, BLOCK_SIZE,
        )
        _multicast_row(
            probs_ptr, probs_mc_ptr, probs_byte_offset, token_offset, global_row, tid,
            PROBS_PACKS, PROBS_BITS, BLOCK_SIZE,
        )

    # The release covers this CTA's multicast and unicast stores.
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
def _reduce_row(
    output_u64, partial_mc_u64, partial_byte_offset, token_offset, global_row, tid,
    HIDDEN_PACKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """One token, multicast-reduce: the switch sums row global_row over every rank (fp32
    accumulation). The body of the RSV-V token loop, verbatim."""
    for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
        cols = channel_offset + tid
        mask = cols < HIDDEN_PACKS
        (x, y, z, w) = ld_128(
            partial_mc_u64 + partial_byte_offset // 8 + (global_row * HIDDEN_PACKS + cols) * 2,
            mask=mask,
            multicast_op=True,
        )
        st_128(
            output_u64 + (token_offset * HIDDEN_PACKS + cols) * 2,
            x, y, z, w,
            mask=mask,
            multicast_op=False,
        )


@triton.jit
def _pull_row(
    output_u64, peer_table, partial_byte_offset, routing_ids, token_offset, global_row, tid,
    HIDDEN_PACKS: tl.constexpr, TOPK: tl.constexpr, NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One token, pull: read row global_row from each distinct destination rank and sum in fp32
    in a fixed order. The body of the _a2a_combine_v_kernel token loop, verbatim: each peer
    load's address is written out in full (see the kernel docstring)."""
    for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
        cols = channel_offset + tid
        mask = cols < HIDDEN_PACKS
        acc_xl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_xh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_yl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_yh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_zl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_zh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_wl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_wh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        sent = tl.full((), 0, tl.int64)
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


@triton.jit
def _hybrid_combine_v_kernel(
    output_ptr,
    partial_mc_ptr,
    partial_peer_ptrs,
    partial_byte_offset,
    routing_ptr,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    mode_ptr,
    split_ptr,
    HIDDEN_PACKS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HAS_SPLIT: tl.constexpr = False,
):
    """Hybrid combine. One CTA per token (persistent over local_tokens).

    Args as _a2a_combine_v_kernel, plus:
        partial_mc_ptr: multicast pointer of the bf16 partial buffer (multicast-reduce mode).
        mode_ptr: scalar int32, 1 = multimem.ld_reduce over every rank, 0 = pull from the
            token's destination ranks. May differ per rank: the expert writes zero rows for
            tokens it did not process, so any owner can multicast-reduce.
        split_ptr: with HAS_SPLIT, scalar int32 m: in multicast-reduce mode only CTAs pid < m
            multicast-reduce; the others pull (per-token mixing, as in dispatch).

    One token loop; per token, the branch picks a helper whose body is the stand-alone kernel's
    token-loop body verbatim (_reduce_row: RSV-V, _pull_row: A2A-V). Keep each peer load's
    address written out in full: precomputing one row offset and adding it to all 22 peer
    pointers compiled to a pull loop 5-17% slower than A2A-V's (some peer-pointer loads were
    scheduled between the row loads). Where the branch sits (outside or inside the token loop)
    and where the flag is loaded make no measurable difference
    (profiles/combine_branch_probe.py, docs/hybrid_dispatcher_design.md §3.4).
    """
    pid = tl.program_id(axis=0)

    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    use_reduce = tl.load(mode_ptr) != 0
    if HAS_SPLIT:
        use_reduce = use_reduce & (pid < tl.load(split_ptr))
    if pid >= ep_max_tokens:
        return

    # Triton-3.6 fix (as in variable_collectives.py): widen raw pointer ints to i64.
    output_u64 = output_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
    partial_mc_u64 = partial_mc_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
    peer_table = partial_peer_ptrs.to(tl.int64).to(tl.pointer_type(tl.uint64))
    routing_ids = routing_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))

    # Relaxed, as RSV-V and A2A-V: the partials were written by earlier kernels.
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False,
    )
    sync_threads()

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)
    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_row = (rank_token_offset + token_offset).to(tl.int64)
        if use_reduce:
            _reduce_row(
                output_u64, partial_mc_u64, partial_byte_offset, token_offset, global_row, tid,
                HIDDEN_PACKS, BLOCK_SIZE,
            )
        else:
            _pull_row(
                output_u64, peer_table, partial_byte_offset, routing_ids, token_offset, global_row,
                tid, HIDDEN_PACKS, TOPK, NUM_LOCAL_EXPERTS, BLOCK_SIZE,
            )
    # No trailing barrier: the next dispatch's barrier protects the partial buffer.


def _check_scalar_int32(name: str, t: torch.Tensor):
    assert t.numel() == 1 and t.dtype == torch.int32 and t.is_cuda, (
        f"{name} must be a scalar int32 CUDA tensor")


def hybrid_dispatch_v(
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
    dispatch_mode: torch.Tensor,
    per_rank_max_tokens: int,
    num_local_experts: int,
    hidden_byte_offset: int = 0,
    routing_byte_offset: int = 0,
    probs_byte_offset: int = 0,
    mc_split: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Hybrid dispatch: hidden rows multicast to every rank (dispatch_mode == 1) or unicast to
    their destination ranks (0); routing ids and probs multicast; all at row
    rank_token_offset + t.

    Args as a2a_dispatch_v, plus:
        dispatch_mode: scalar int32 CUDA tensor (step_metadata[3:4] from
            fused_metadata_update with mode_params), read on device each launch.
        mc_split: optional scalar int32 CUDA tensor m (prototype): in multicast mode only CTAs
            pid < m multicast, the rest of this rank's tokens are unicast.
    """
    assert HAVE_TRITON, "Triton is required for hybrid_dispatch_v."
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
    _check_scalar_int32("rank_token_offset", rank_token_offset)
    _check_scalar_int32("dispatch_mode", dispatch_mode)
    if mc_split is not None:
        _check_scalar_int32("mc_split", mc_split)
    hdls = (symm_mem_hdl_hidden, symm_mem_hdl_routing, symm_mem_hdl_probs)
    assert len({h.rank for h in hdls}) == 1 and len({h.world_size for h in hdls}) == 1
    world_size = symm_mem_hdl_hidden.world_size
    assert world_size <= _MAX_WORLD_SIZE, f"world size {world_size} > {_MAX_WORLD_SIZE}"

    hidden_bits, hidden_packs = _row_params(input_hidden)
    assert hidden_bits == 128, "hidden rows must be 16-byte aligned"
    routing_bits, routing_packs = _row_params(input_routing)
    probs_bits, probs_packs = _row_params(input_probs)

    max_num_blocks = kwargs.get("max_num_blocks", device_num_sms(input_hidden.device))
    block_size = min(
        max(triton.next_power_of_2(p) for p in (hidden_packs, routing_packs, probs_packs)),
        _MAX_BLOCK_SIZE,
    )
    num_warps = max(1, block_size // _WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _hybrid_dispatch_v_kernel[(num_blocks, 1, 1)](
        input_hidden.data_ptr(),
        symm_mem_hdl_hidden.multicast_ptr,
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
        mode_ptr=dispatch_mode,
        split_ptr=mc_split if mc_split is not None else dispatch_mode,  # unread without split
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
        HAS_SPLIT=mc_split is not None,
        num_warps=num_warps,
    )
    return output_hidden, output_routing, output_probs


def hybrid_combine_v(
    output: torch.Tensor,
    partial: torch.Tensor,
    symm_mem_hdl_partial: _SymmetricMemory,
    routing: torch.Tensor,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    combine_mode: torch.Tensor,
    per_rank_max_tokens: int,
    num_local_experts: int,
    partial_byte_offset: int = 0,
    mc_split: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Hybrid combine: for each local token, sum its bf16 partial rows (row rank_token_offset + t)
    with multimem.ld_reduce over every rank (combine_mode == 1; every rank's row must be valid,
    zero where the rank has no expert for the token) or by pulling them from its destination
    ranks (0). fp32 accumulation either way; stores bf16 into `output`.

    Args as a2a_combine_v, plus:
        combine_mode: scalar int32 CUDA tensor (step_metadata[4:5] from fused_metadata_update
            with mode_params), read on device each launch. May differ per rank (zero rows are
            always written).
        mc_split: optional scalar int32 CUDA tensor m (prototype): in multicast-reduce mode only
            CTAs pid < m multicast-reduce, the rest of this rank's tokens are pulled.
    """
    assert HAVE_TRITON, "Triton is required for hybrid_combine_v."
    assert output.ndim == 2 and partial.ndim == 2 and routing.ndim == 2
    assert output.dtype == partial.dtype == torch.bfloat16, "combine supports bf16 partials only"
    assert output.shape[1] == partial.shape[1], "output / partial hidden mismatch"
    assert output.is_contiguous() and routing.is_contiguous()
    assert routing.dtype == torch.int64 and routing.shape[0] == output.shape[0]
    assert is_device_nvls_capable(output.device), "needs a Hopper+ GPU with NVLink"
    _check_scalar_int32("rank_token_offset", rank_token_offset)
    _check_scalar_int32("combine_mode", combine_mode)
    if mc_split is not None:
        _check_scalar_int32("mc_split", mc_split)
    world_size = symm_mem_hdl_partial.world_size
    assert world_size <= _MAX_WORLD_SIZE, f"world size {world_size} > {_MAX_WORLD_SIZE}"
    bits, hidden_packs = _row_params(output)
    assert bits == 128, "hidden rows must be 16-byte aligned"

    max_num_blocks = kwargs.get("max_num_blocks", device_num_sms(output.device))
    block_size = min(triton.next_power_of_2(hidden_packs), _MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // _WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _hybrid_combine_v_kernel[(num_blocks, 1, 1)](
        output.data_ptr(),
        symm_mem_hdl_partial.multicast_ptr,
        symm_mem_hdl_partial.buffer_ptrs_dev,
        partial_byte_offset,
        routing.data_ptr(),
        symm_mem_hdl_partial.signal_pad_ptrs_dev,
        local_tokens=output.shape[0],
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        mode_ptr=combine_mode,
        split_ptr=mc_split if mc_split is not None else combine_mode,  # unread without split
        HIDDEN_PACKS=hidden_packs,
        TOPK=routing.shape[1],
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl_partial.rank,
        WORLD_SIZE=world_size,
        HAS_SPLIT=mc_split is not None,
        num_warps=num_warps,
    )
    return output
