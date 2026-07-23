# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused NVLS metadata update kernel for MoE expert parallelism.

Replaces the multi-kernel sequence:
    dist.all_gather_into_tensor(...)   # NCCL
    local_tokens_per_rank.sum()        # kernel
    local_tokens_per_rank[:rank].sum() # kernel
    local_tokens_per_rank.max()        # kernel
    _step_metadata.copy_(...)          # kernel

with a single Triton kernel that:
    1. Multicast-stores this rank's local_tokens to the symmetric memory buffer.
    2. Barrier (all ranks have written).
    3. Reads all ranks' counts, computes sum / prefix-sum / max.
    4. Writes the 3-element step_metadata tensor in-place.
"""

from unittest.mock import MagicMock

import torch

from .torch_symm_triton._compat import null_decorator

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

from .torch_symm_triton.barrier import symm_mem_sync
from .torch_symm_triton.multimem_asm import st_32
from .torch_symm_triton.utils import sync_threads


@triton.jit
def _fused_metadata_kernel(
    local_tokens,
    local_buf_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    step_metadata_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Fused allgather + reduce kernel for MoE step metadata.

    Single CTA. Writes this rank's local_tokens to the symmetric buffer
    via multicast store, barriers, then reads all ranks' values from the
    local buffer and computes [valid_tokens, rank_token_offset, ep_max_tokens].

    Args:
        local_tokens: scalar int32, this rank's token count.
        local_buf_ptr: pointer to the local symmetric memory buffer (for reads).
        multicast_ptr: multicast pointer to the symmetric memory buffer (for writes).
        signal_pad_ptrs: signal pads for barrier synchronization.
        step_metadata_ptr: pointer to the 3-element int32 output tensor.
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
    """

    tid = tl.program_id(0)
    if tid > 0:
        return

    # Required Triton-3.6 fix (NOT diagnostic): widen the raw multicast pointer int
    # to i64 (tt.int_to_ptr requires i64; a low VA gets specialized as i32). Without
    # this, the metadata kernel does not compile and NVLS setup fails.
    multicast_ptr = multicast_ptr.to(tl.int64)

    # 1. Multicast-store local_tokens to buffer[RANK].
    mc_ptr = multicast_ptr.to(tl.pointer_type(tl.uint32)) + RANK
    mask = tl.full([], 1, dtype=tl.int1)
    val = tl.full([], local_tokens, dtype=tl.uint32)
    st_32(mc_ptr, val, mask, multicast_op=True)

    # 2. Barrier — wait for all ranks to have written.
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # 3. Load all ranks' values, reduce, and write metadata.
    offsets = tl.arange(0, WORLD_SIZE)
    vals = tl.load(local_buf_ptr + offsets)

    total = tl.sum(vals)
    prefix = tl.sum(tl.where(offsets < RANK, vals, tl.zeros_like(vals)))
    max_val = tl.max(vals)

    tl.store(step_metadata_ptr, total)
    tl.store(step_metadata_ptr + 1, prefix)
    tl.store(step_metadata_ptr + 2, max_val)


@triton.jit
def _fused_metadata_kernel_dev(
    local_tokens_ptr,
    local_buf_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    step_metadata_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Device-count variant of _fused_metadata_kernel: reads this rank's token count from a device
    int32 scalar (local_tokens_ptr) instead of a launch-time Python int. This is what makes the
    metadata graph-capturable with a count that changes each replay -- required when the routing
    (hence the count) is produced per layer inside the model graph, not known at capture time."""
    tid = tl.program_id(0)
    if tid > 0:
        return
    multicast_ptr = multicast_ptr.to(tl.int64)
    local_tokens = tl.load(local_tokens_ptr)            # <-- device read (vs the baked scalar arg)

    mc_ptr = multicast_ptr.to(tl.pointer_type(tl.uint32)) + RANK
    mask = tl.full([], 1, dtype=tl.int1)
    st_32(mc_ptr, local_tokens.to(tl.uint32), mask, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=True, hasSubsequentMemAccess=True,
    )

    offsets = tl.arange(0, WORLD_SIZE)
    vals = tl.load(local_buf_ptr + offsets)
    total = tl.sum(vals)
    prefix = tl.sum(tl.where(offsets < RANK, vals, tl.zeros_like(vals)))
    max_val = tl.max(vals)
    tl.store(step_metadata_ptr, total)
    tl.store(step_metadata_ptr + 1, prefix)
    tl.store(step_metadata_ptr + 2, max_val)


def fused_metadata_update_dev(
    local_tokens: torch.Tensor,
    local_buf: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    step_metadata: torch.Tensor,
) -> None:
    """Graph-capturable NVLS metadata allgather+reduce reading the count from device.

    Identical to fused_metadata_update but `local_tokens` is a [>=1] int32 CUDA tensor (only element 0
    is read) rather than a Python int, so the value is picked up at replay time. Writes
    step_metadata = [valid_tokens, rank_token_offset, ep_max_tokens]."""
    assert HAVE_TRITON, "Triton is required for fused_metadata_update_dev."
    _fused_metadata_kernel_dev[(1, 1, 1)](
        local_tokens,
        local_buf,
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        step_metadata,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=min(max(1, (symm_mem_hdl.world_size + 31) // 32), 8),
    )


@triton.jit
def _fused_group_metadata_kernel(
    counts_ptr,
    local_buf_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    offsets_ptr,
    valid_ptr,
    MIN_GROUP: tl.constexpr,
    NSZ: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """ONE collective computing rank_token_offset (and per-group valid total) for ALL nested group
    sizes at once (vs one metadata kernel per size). Each rank multicast-stores its [NSZ] per-size
    count vector into a [P, NSZ] symm buffer, barriers once, then for each size i computes the prefix
    sum over the members of its aligned size-(MIN_GROUP<<i) group that precede it (offsets) and the
    full-group sum (valid). Writes offsets_ptr[NSZ], valid_ptr[NSZ]."""
    tid = tl.program_id(0)
    if tid > 0:
        return
    multicast_ptr = multicast_ptr.to(tl.int64)
    si = tl.arange(0, NSZ)

    # 1. multicast-store this rank's count vector to buffer[RANK, 0:NSZ].
    mc = multicast_ptr.to(tl.pointer_type(tl.uint32)) + RANK * NSZ + si
    cnt = tl.load(counts_ptr + si)
    st_32(mc, cnt.to(tl.uint32), si < NSZ, multicast_op=True)

    # 2. barrier.
    sync_threads()
    symm_mem_sync(signal_pad_ptrs, None, RANK, WORLD_SIZE,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    # 3. per-size prefix (offset) and full-group sum (valid).
    ranks = tl.arange(0, WORLD_SIZE)
    for i in range(NSZ):
        s = MIN_GROUP << i
        gstart = (RANK // s) * s
        col = tl.load(local_buf_ptr + ranks * NSZ + i)
        in_group = (ranks >= gstart) & (ranks < gstart + s)
        before = in_group & (ranks < RANK)
        tl.store(offsets_ptr + i, tl.sum(tl.where(before, col, tl.zeros_like(col))))
        tl.store(valid_ptr + i, tl.sum(tl.where(in_group, col, tl.zeros_like(col))))


def fused_group_metadata_update(
    counts: torch.Tensor,
    local_buf: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    offsets: torch.Tensor,
    valid: torch.Tensor,
    min_group: int,
    nsz: int,
) -> None:
    """One graph-capturable collective -> rank_token_offset + valid total per nested size. counts/
    offsets/valid are [NSZ] int32 CUDA tensors; local_buf is a [P*NSZ] int32 symm buffer; symm_mem_hdl
    spans all P ranks."""
    assert HAVE_TRITON, "Triton is required for fused_group_metadata_update."
    _fused_group_metadata_kernel[(1, 1, 1)](
        counts, local_buf, symm_mem_hdl.multicast_ptr, symm_mem_hdl.signal_pad_ptrs_dev,
        offsets, valid,
        MIN_GROUP=min_group, NSZ=nsz,
        RANK=symm_mem_hdl.rank, WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=min(max(1, (symm_mem_hdl.world_size + 31) // 32), 8),
    )


def fused_metadata_update(
    local_tokens: int,
    local_buf: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    step_metadata: torch.Tensor,
) -> None:
    """Fused NVLS allgather + reduce for MoE step metadata.

    Args:
        local_tokens: number of tokens on this rank this step.
        local_buf: the local symmetric memory buffer tensor ([WORLD_SIZE] int32).
            Used for reads after the barrier.
        symm_mem_hdl: symmetric memory handle for the metadata buffer.
            Provides the multicast pointer for writes and signal pads for barrier.
        step_metadata: [3] int32 CUDA tensor to write
            [valid_tokens, rank_token_offset, ep_max_tokens] into.
    """
    assert HAVE_TRITON, "Triton is required for fused_metadata_update."

    _fused_metadata_kernel[(1, 1, 1)](
        local_tokens,
        local_buf,
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        step_metadata,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=min(max(1, (symm_mem_hdl.world_size + 31) // 32), 8),
    )
