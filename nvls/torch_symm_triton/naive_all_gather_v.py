# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Variable-count all-gather over NVLink with plain unicast stores (no multicast).

Same contract as multimem_all_gatherv_3tensor in variable_collectives.py: source rank s writes
its local token t to row rank_token_offset[s] + t of the [global_max, *] symmetric buffers on
EVERY rank, using the same per-step metadata (rank_token_offset, ep_max_tokens), the same
one-token-per-CTA grid with early exit, and the same trailing per-CTA-pair barrier.

The only difference is the transport. AGV-V issues one multimem.st per 16-byte chunk and the
NVSwitch replicates it to all ranks; here each chunk is read from HBM once and stored
WORLD_SIZE times through the peer pointers in buffer_ptrs_dev (WORLD_SIZE - 1 NVLink stores
plus one local store). Each GPU therefore pushes WORLD_SIZE x the bytes AGV-V does, so this is
the "naive" baseline that isolates what NVLink multicast buys.

Destinations are visited starting at RANK + 1 (wrapping), so ranks do not all hit the same peer
at the same time. Buffer reuse is identical to AGV-V: every rank's buffer receives the same rows
either way, so the AGV-V / RSV-V barrier chain still protects it.
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

from .all_to_all_v import _row_params
from .barrier import symm_mem_sync
from .multimem_asm import ld_64, ld_128, st_64, st_128
from .utils import device_num_sms, is_device_nvls_capable, sync_threads

_MAX_BLOCK_SIZE = 1024
_WARP_SIZE = 32


@triton.jit
def _unicast_row(
    src_ptr,
    peer_table,
    byte_offset,
    token_offset,
    global_row,
    tid,
    PACKS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Copy local row `token_offset` to row `global_row` of every rank's buffer.

    peer_table: uint64 pointer to the WORLD_SIZE base addresses of the receive buffers.

    The last destination (i == WORLD_SIZE) is RANK itself: the input is a separate local
    tensor, so this rank's own rows must also be copied into its gathered buffer (multimem.st
    does the same). That store hits local HBM, not NVLink.
    """
    for channel_offset in range(0, PACKS, BLOCK_SIZE):
        cols = channel_offset + tid
        mask = cols < PACKS
        local_offsets = token_offset * PACKS + cols
        global_offsets = global_row * PACKS + cols
        if BITS == 128:
            src_ptrs = src_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
            (x, y, z, w) = ld_128(src_ptrs, mask=mask, multicast_op=False)
            for i in tl.static_range(1, WORLD_SIZE + 1):
                peer = tl.load(peer_table + (RANK + i) % WORLD_SIZE).to(tl.pointer_type(tl.uint64))
                st_128(
                    peer + byte_offset // 8 + global_offsets * 2,
                    x, y, z, w,
                    mask=mask,
                    multicast_op=False,
                )
        else:
            src_ptrs = src_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets
            (x, y) = ld_64(src_ptrs, mask=mask)
            for i in tl.static_range(1, WORLD_SIZE + 1):
                peer = tl.load(peer_table + (RANK + i) % WORLD_SIZE).to(tl.pointer_type(tl.uint64))
                st_64(peer + byte_offset // 8 + global_offsets, x, y, mask=mask, multicast_op=False)


@triton.jit
def _naive_all_gatherv_3tensor_kernel(
    src_ptr_0,
    peer_ptrs_0,
    byte_offset_0,
    src_ptr_1,
    peer_ptrs_1,
    byte_offset_1,
    src_ptr_2,
    peer_ptrs_2,
    byte_offset_2,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    PACKS_0: tl.constexpr,
    BITS_0: tl.constexpr,
    PACKS_1: tl.constexpr,
    BITS_1: tl.constexpr,
    PACKS_2: tl.constexpr,
    BITS_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Unicast all-gather-V of three tensors. One CTA per token (persistent over local_tokens).

    Args:
        src_ptr_0/1/2: local inputs [local_tokens, *] (raw pointer ints).
        peer_ptrs_0/1/2: device arrays of WORLD_SIZE uint64 base addresses of every rank's
            receive buffer (each handle's buffer_ptrs_dev).
        byte_offset_0/1/2: byte offset of each tensor inside its symmetric buffer.
        signal_pad_ptrs: signal pads of buffer 0 (one barrier covers all three).
        local_tokens / rank_token_offset_ptr / ep_max_tokens_ptr: as AGV-V.
        PACKS_i / BITS_i: packs per row and pack width (128 or 64 bits) of tensor i.
        BLOCK_SIZE: threads per CTA (>= every PACKS_i up to the 1024 cap; wider rows loop).
    """
    pid = tl.program_id(axis=0)

    # Same rank-consistent early exit as AGV-V: CTAs >= ep_max_tokens have no token on any rank.
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Triton-3.6 fix (as in variable_collectives.py): widen raw pointer ints to i64.
    src_ptr_0 = src_ptr_0.to(tl.int64)
    src_ptr_1 = src_ptr_1.to(tl.int64)
    src_ptr_2 = src_ptr_2.to(tl.int64)
    peer_table_0 = peer_ptrs_0.to(tl.int64).to(tl.pointer_type(tl.uint64))
    peer_table_1 = peer_ptrs_1.to(tl.int64).to(tl.pointer_type(tl.uint64))
    peer_table_2 = peer_ptrs_2.to(tl.int64).to(tl.pointer_type(tl.uint64))

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_row = (rank_token_offset + token_offset).to(tl.int64)
        _unicast_row(
            src_ptr_0, peer_table_0, byte_offset_0, token_offset, global_row, tid,
            PACKS_0, BITS_0, BLOCK_SIZE, RANK, WORLD_SIZE,
        )
        _unicast_row(
            src_ptr_1, peer_table_1, byte_offset_1, token_offset, global_row, tid,
            PACKS_1, BITS_1, BLOCK_SIZE, RANK, WORLD_SIZE,
        )
        _unicast_row(
            src_ptr_2, peer_table_2, byte_offset_2, token_offset, global_row, tid,
            PACKS_2, BITS_2, BLOCK_SIZE, RANK, WORLD_SIZE,
        )

    # The release in the barrier covers this CTA's peer stores.
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def naive_all_gatherv_3tensor(
    output_tensor_0: torch.Tensor,
    output_tensor_1: torch.Tensor,
    output_tensor_2: torch.Tensor,
    input_tensor_0: torch.Tensor,
    input_tensor_1: torch.Tensor,
    input_tensor_2: torch.Tensor,
    symm_mem_hdl_0: _SymmetricMemory,
    symm_mem_hdl_1: _SymmetricMemory,
    symm_mem_hdl_2: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    output_byte_offset_0: int = 0,
    output_byte_offset_1: int = 0,
    output_byte_offset_2: int = 0,
    **kwargs,
) -> tuple:
    """Unicast drop-in for multimem_all_gatherv_3tensor (same arguments and result).

    Gathers three [local_tokens, hidden_i] tensors from every rank into their [global_tokens,
    hidden_i] symmetric buffers at row rank_token_offset + t, storing each row to every rank
    through the peer pointers instead of NVLink multicast. Rows must be 8-byte aligned (128-bit
    packs when 16-byte aligned, else 64-bit). The barrier uses symm_mem_hdl_0's signal pads.
    """
    assert HAVE_TRITON, "Triton is required for naive_all_gatherv_3tensor."
    inputs = (input_tensor_0, input_tensor_1, input_tensor_2)
    outputs = (output_tensor_0, output_tensor_1, output_tensor_2)
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        assert inp.ndim == 2 and out.ndim == 2, f"tensor {i}: tensors must be 2-D"
        assert inp.shape[1] == out.shape[1], f"tensor {i}: row width mismatch"
        assert inp.dtype == out.dtype, f"tensor {i}: dtype mismatch"
        assert inp.is_contiguous(), f"tensor {i}: input must be contiguous"
    local_tokens = input_tensor_0.shape[0]
    assert all(t.shape[0] == local_tokens for t in inputs), "inputs must share local_tokens"
    assert is_device_nvls_capable(input_tensor_0.device), "needs a Hopper+ GPU with NVLink"
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."
    hdls = (symm_mem_hdl_0, symm_mem_hdl_1, symm_mem_hdl_2)
    assert len({h.rank for h in hdls}) == 1 and len({h.world_size for h in hdls}) == 1

    (bits_0, packs_0), (bits_1, packs_1), (bits_2, packs_2) = (_row_params(t) for t in inputs)

    max_num_blocks = kwargs.get("max_num_blocks", device_num_sms(input_tensor_0.device))
    block_size = min(
        max(triton.next_power_of_2(p) for p in (packs_0, packs_1, packs_2)), _MAX_BLOCK_SIZE
    )
    num_warps = max(1, block_size // _WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _naive_all_gatherv_3tensor_kernel[(num_blocks, 1, 1)](
        input_tensor_0.data_ptr(),
        symm_mem_hdl_0.buffer_ptrs_dev,
        output_byte_offset_0,
        input_tensor_1.data_ptr(),
        symm_mem_hdl_1.buffer_ptrs_dev,
        output_byte_offset_1,
        input_tensor_2.data_ptr(),
        symm_mem_hdl_2.buffer_ptrs_dev,
        output_byte_offset_2,
        symm_mem_hdl_0.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        PACKS_0=packs_0,
        BITS_0=bits_0,
        PACKS_1=packs_1,
        BITS_1=bits_1,
        PACKS_2=packs_2,
        BITS_2=bits_2,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl_0.rank,
        WORLD_SIZE=symm_mem_hdl_0.world_size,
        num_warps=num_warps,
    )
    return output_tensor_0, output_tensor_1, output_tensor_2
