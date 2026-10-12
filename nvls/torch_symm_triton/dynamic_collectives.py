# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic dispatch / combine: AGV-V / RSV-V or all-to-all-V, chosen on device each step.

Both modes share the AGV-V row layout (source rank s writes its local token t to row
rank_token_offset[s] + t of the [global_max, *] symmetric buffers), the CTA grid, the per-step
metadata from metadata.py, and the barrier placement, so a kernel can switch between them without
any extra synchronization. The mode is

    use_a2a = valid_tokens >= a2a_threshold

where valid_tokens (step_metadata[0]) is the global token count, identical on every rank, so every
rank takes the same branch. a2a_threshold = FORCE_A2A always takes the all-to-all path and
FORCE_AG_RS never does.

Dispatch
  * AG mode: hidden rows are multicast to every rank (multimem.st), as in AGV-V.
  * A2A mode: hidden rows are unicast to the token's distinct destination ranks, as in
    a2a_dispatch_v.
  * Both modes multicast routing ids and probs and end with the same release/acquire per-CTA-pair
    barrier.

Combine (bf16 partials, fp32 accumulation, bf16 output in both modes)
  * RS mode: multimem.ld_reduce of the owned rows, as in RSV-V. The reduce sums every rank's row,
    so rows of tokens with no local expert must hold zeros.
  * A2A mode: pull the owned rows from the token's destination ranks and sum them in fp32, as in
    a2a_combine_v. Rows of tokens with no local expert are never read.
  * Both modes start with the same relaxed per-CTA-pair barrier: the partials were written by
    earlier kernels.
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

# a2a_threshold values that pin the mode: valid_tokens >= 0 always holds, and valid_tokens (int32)
# never exceeds 2**31 - 1.
FORCE_A2A = 0
FORCE_AG_RS = 2**31 - 1

# Most recently launched compiled kernel per wrapper, for register / spill reporting.
_LAST_COMPILED = {}


@triton.jit
def _unicast_pack(
    peer_table, offsets_u64, x, y, z, w, mask, routing_ids, token_offset,
    TOPK: tl.constexpr, NUM_LOCAL_EXPERTS: tl.constexpr,
):
    """A2A-V store of one 16-byte pack per thread to each distinct destination rank of the token.
    offsets_u64 is the same on every rank (AGV-V row layout)."""
    sent = tl.full((), 0, tl.int64)
    for k in tl.static_range(TOPK):
        dst, first, sent = _next_destination(
            routing_ids, token_offset, sent, k, TOPK, NUM_LOCAL_EXPERTS
        )
        peer = tl.load(peer_table + dst).to(tl.pointer_type(tl.uint64))
        st_128(peer + offsets_u64, x, y, z, w, mask=mask & first, multicast_op=False)


@triton.jit
def _pull_sum_pack(
    peer_table, offsets_u64, mask, routing_ids, token_offset,
    BLOCK_SIZE: tl.constexpr, TOPK: tl.constexpr, NUM_LOCAL_EXPERTS: tl.constexpr,
):
    """A2A-V pull of one 16-byte bf16 pack per thread from each distinct destination rank of the
    token, summed in fp32 in top-k order (deterministic) and rounded once to bf16x2."""
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
        (x, y, z, w) = _ld_128_or_zero(peer + offsets_u64, mask & first)
        acc_xl += _bf16_lo_to_f32(x)
        acc_xh += _bf16_hi_to_f32(x)
        acc_yl += _bf16_lo_to_f32(y)
        acc_yh += _bf16_hi_to_f32(y)
        acc_zl += _bf16_lo_to_f32(z)
        acc_zh += _bf16_hi_to_f32(z)
        acc_wl += _bf16_lo_to_f32(w)
        acc_wh += _bf16_hi_to_f32(w)
    return (
        _f32_to_bf16x2(acc_xl, acc_xh),
        _f32_to_bf16x2(acc_yl, acc_yh),
        _f32_to_bf16x2(acc_zl, acc_zh),
        _f32_to_bf16x2(acc_wl, acc_wh),
    )


@triton.jit
def _dynamic_dispatch_v_kernel(
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
    valid_tokens_ptr,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    a2a_threshold,
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
    """Dynamic dispatch. One CTA per token (persistent over local_tokens).

    Args beyond a2a_dispatch_v's:
        hidden_mc_ptr: multicast pointer of the hidden receive buffer (AG mode).
        hidden_peer_ptrs: the hidden handle's buffer_ptrs_dev (A2A mode).
        valid_tokens_ptr: per-step global token count (step_metadata[0]).
        a2a_threshold: take the A2A path when valid_tokens >= a2a_threshold.
    """
    pid = tl.program_id(axis=0)

    # Same rank-consistent early exit as AGV-V: CTAs >= ep_max_tokens have no token on any rank.
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

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

    # valid_tokens is the same on every rank, so every rank (and CTA) takes the same branch.
    use_a2a = tl.load(valid_tokens_ptr) >= a2a_threshold

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_row = (rank_token_offset + token_offset).to(tl.int64)

        # Hidden: read the row from HBM once, then multicast it (AG) or unicast it to each
        # distinct destination rank (A2A). Same row on every rank either way.
        for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
            cols = channel_offset + tid
            mask = cols < HIDDEN_PACKS
            (x, y, z, w) = ld_128(
                hidden_u64 + (token_offset * HIDDEN_PACKS + cols) * 2, mask=mask, multicast_op=False
            )
            offsets_u64 = hidden_byte_offset // 8 + (global_row * HIDDEN_PACKS + cols) * 2
            if use_a2a:
                _unicast_pack(
                    peer_table, offsets_u64, x, y, z, w, mask, routing_ids, token_offset,
                    TOPK, NUM_LOCAL_EXPERTS,
                )
            else:
                st_128(hidden_mc_u64 + offsets_u64, x, y, z, w, mask=mask, multicast_op=True)

        # Routing ids and probs: multicast to every rank in both modes.
        _multicast_row(
            routing_ptr, routing_mc_ptr, routing_byte_offset, token_offset, global_row, tid,
            ROUTING_PACKS, ROUTING_BITS, BLOCK_SIZE,
        )
        _multicast_row(
            probs_ptr, probs_mc_ptr, probs_byte_offset, token_offset, global_row, tid,
            PROBS_PACKS, PROBS_BITS, BLOCK_SIZE,
        )

    # The release in the barrier covers this CTA's multicast and unicast stores.
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
def _dynamic_combine_v_kernel(
    output_ptr,
    partial_mc_ptr,
    partial_peer_ptrs,
    partial_byte_offset,
    routing_ptr,
    signal_pad_ptrs,
    local_tokens,
    valid_tokens_ptr,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    a2a_threshold,
    HIDDEN_PACKS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Dynamic combine. One CTA per token (persistent over local_tokens).

    Args beyond a2a_combine_v's:
        partial_mc_ptr: multicast pointer of the bf16 partial buffer (RS mode).
        valid_tokens_ptr: per-step global token count (step_metadata[0]).
        a2a_threshold: take the A2A path when valid_tokens >= a2a_threshold.
    """
    pid = tl.program_id(axis=0)

    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Same barrier in both modes: the partials were written by earlier kernels, so a relaxed
    # barrier makes them visible (torch symmetric memory "Pattern 0", as in RSV-V).
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False,
    )
    sync_threads()

    # Triton-3.6 fix (as in variable_collectives.py): widen raw pointer ints to i64.
    output_ptr = output_ptr.to(tl.int64)
    partial_mc_ptr = partial_mc_ptr.to(tl.int64)
    partial_peer_ptrs = partial_peer_ptrs.to(tl.int64)
    routing_ptr = routing_ptr.to(tl.int64)

    output_u64 = output_ptr.to(tl.pointer_type(tl.uint64))
    partial_mc_u64 = partial_mc_ptr.to(tl.pointer_type(tl.uint64))
    peer_table = partial_peer_ptrs.to(tl.pointer_type(tl.uint64))
    routing_ids = routing_ptr.to(tl.pointer_type(tl.int64))

    # valid_tokens is the same on every rank, so every rank (and CTA) takes the same branch.
    use_a2a = tl.load(valid_tokens_ptr) >= a2a_threshold

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_row = (rank_token_offset + token_offset).to(tl.int64)

        for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
            cols = channel_offset + tid
            mask = cols < HIDDEN_PACKS
            offsets_u64 = partial_byte_offset // 8 + (global_row * HIDDEN_PACKS + cols) * 2
            if use_a2a:
                (x, y, z, w) = _pull_sum_pack(
                    peer_table, offsets_u64, mask, routing_ids, token_offset,
                    BLOCK_SIZE, TOPK, NUM_LOCAL_EXPERTS,
                )
            else:
                # bf16 operands, fp32 accumulation in the switch (acc::f32), bf16 result.
                (x, y, z, w) = ld_128(
                    partial_mc_u64 + offsets_u64, mask=mask, multicast_op=True, reduce_f32=False
                )
            st_128(
                output_u64 + (token_offset * HIDDEN_PACKS + cols) * 2,
                x, y, z, w,
                mask=mask,
                multicast_op=False,
            )
    # No trailing barrier: the next dispatch's barrier protects the partial buffer.


def _check_scalar_int32(name: str, t: torch.Tensor):
    assert t.numel() == 1 and t.dtype == torch.int32 and t.is_cuda, (
        f"{name} must be a scalar int32 CUDA tensor."
    )


def dynamic_dispatch_v(
    output_hidden: torch.Tensor,
    output_routing: torch.Tensor,
    output_probs: torch.Tensor,
    input_hidden: torch.Tensor,
    input_routing: torch.Tensor,
    input_probs: torch.Tensor,
    symm_mem_hdl_hidden: _SymmetricMemory,
    symm_mem_hdl_routing: _SymmetricMemory,
    symm_mem_hdl_probs: _SymmetricMemory,
    valid_tokens: torch.Tensor,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    num_local_experts: int,
    a2a_threshold: int,
    hidden_byte_offset: int = 0,
    routing_byte_offset: int = 0,
    probs_byte_offset: int = 0,
    **kwargs,
):
    """Dynamic dispatch: AGV-V when valid_tokens < a2a_threshold, else all-to-all-V (see
    a2a_dispatch_v). Routing ids and probs are multicast in both modes.

    Args (beyond a2a_dispatch_v's):
        valid_tokens: per-step scalar int32 CUDA tensor with the global token count
            (step_metadata[0]); identical on every rank.
        a2a_threshold: static int; FORCE_A2A / FORCE_AG_RS pin the mode.
    """
    assert HAVE_TRITON, "Triton is required for dynamic_dispatch_v."
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
    for name, t in (
        ("valid_tokens", valid_tokens),
        ("rank_token_offset", rank_token_offset),
        ("ep_max_tokens", ep_max_tokens),
    ):
        _check_scalar_int32(name, t)
    assert FORCE_A2A <= a2a_threshold <= FORCE_AG_RS, f"a2a_threshold {a2a_threshold} out of range"
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

    _LAST_COMPILED["dispatch"] = _dynamic_dispatch_v_kernel[(num_blocks, 1, 1)](
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
        valid_tokens_ptr=valid_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        a2a_threshold=a2a_threshold,
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


def dynamic_combine_v(
    output: torch.Tensor,
    partial: torch.Tensor,
    symm_mem_hdl_partial: _SymmetricMemory,
    routing: torch.Tensor,
    valid_tokens: torch.Tensor,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    num_local_experts: int,
    a2a_threshold: int,
    partial_byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Dynamic combine: RSV-V when valid_tokens < a2a_threshold, else all-to-all-V pull (see
    a2a_combine_v). bf16 partials and output in both modes, accumulated in fp32.

    In RS mode the multicast reduce sums every rank's partial row, so rows of tokens with no local
    expert must hold zeros; in A2A mode those rows are never read.

    Args (beyond a2a_combine_v's):
        valid_tokens: per-step scalar int32 CUDA tensor with the global token count
            (step_metadata[0]); identical on every rank.
        a2a_threshold: static int; FORCE_A2A / FORCE_AG_RS pin the mode.
    """
    assert HAVE_TRITON, "Triton is required for dynamic_combine_v."
    assert output.ndim == 2 and partial.ndim == 2 and routing.ndim == 2
    assert output.dtype == partial.dtype == torch.bfloat16, "combine supports bf16 partials only"
    assert output.shape[1] == partial.shape[1], "output / partial hidden mismatch"
    assert output.is_contiguous() and routing.is_contiguous()
    assert routing.dtype == torch.int64 and routing.shape[0] == output.shape[0]
    assert is_device_nvls_capable(output.device), "needs a Hopper+ GPU with NVLink"
    for name, t in (
        ("valid_tokens", valid_tokens),
        ("rank_token_offset", rank_token_offset),
        ("ep_max_tokens", ep_max_tokens),
    ):
        _check_scalar_int32(name, t)
    assert FORCE_A2A <= a2a_threshold <= FORCE_AG_RS, f"a2a_threshold {a2a_threshold} out of range"
    world_size = symm_mem_hdl_partial.world_size
    assert world_size <= _MAX_WORLD_SIZE, f"world size {world_size} > {_MAX_WORLD_SIZE}"
    bits, hidden_packs = _row_params(output)
    assert bits == 128, "hidden rows must be 16-byte aligned"

    max_num_blocks = kwargs.get("max_num_blocks", device_num_sms(output.device))
    block_size = min(triton.next_power_of_2(hidden_packs), _MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // _WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _LAST_COMPILED["combine"] = _dynamic_combine_v_kernel[(num_blocks, 1, 1)](
        output.data_ptr(),
        symm_mem_hdl_partial.multicast_ptr,
        symm_mem_hdl_partial.buffer_ptrs_dev,
        partial_byte_offset,
        routing.data_ptr(),
        symm_mem_hdl_partial.signal_pad_ptrs_dev,
        local_tokens=output.shape[0],
        valid_tokens_ptr=valid_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        a2a_threshold=a2a_threshold,
        HIDDEN_PACKS=hidden_packs,
        TOPK=routing.shape[1],
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl_partial.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output


def kernel_stats() -> dict:
    """{"dispatch" / "combine": (registers per thread, spill bytes)} of the most recently launched
    dynamic kernels (None until a kernel has been launched)."""
    return {
        name: (getattr(k, "n_regs", None), getattr(k, "n_spills", None))
        for name, k in _LAST_COMPILED.items()
    }
