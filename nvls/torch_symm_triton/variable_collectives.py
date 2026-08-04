# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Variable-count NVLS collectives (AllGatherV / ReduceScatterV).

Unlike the uniform collectives in collectives.py, each rank may contribute
a different number of tokens. The caller provides:
  - rank_token_offset: prefix sum of token counts for all lower-ranked ranks.
  - local_tokens: this rank's token count.

One CTA processes one token; the outer loop is persistent over local_tokens.
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
from .multimem_asm import (
    ld_64,
    ld_128,
    ld_128_nc,
    ld_128_p2p,
    st_32_p2p,
    st_64,
    st_128,
    st_128_p2p,
)
from .utils import is_device_nvls_capable, sync_threads


@triton.jit
def _multimem_all_gather_v_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    output_byte_offset,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    BITS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Variable-count multicast all-gather kernel. One CTA processes one token.

    Each rank contributes local_tokens tokens starting at rank_token_offset in
    the global output. Ranks may have different local_tokens values.

    Args:
        local_ptr: pointer to this rank's local input, shape [local_tokens, hidden_size].
        multicast_ptr: multicast pointer to the output symmetric memory buffer.
        signal_pad_ptrs: signal pads for barrier synchronization.
        local_tokens: number of tokens this rank contributes.
        rank_token_offset_ptr: pointer to a scalar int32 CUDA tensor holding the index
            of the first token this rank writes in the global output (prefix sum of
            local_tokens for all lower-ranked ranks). Fixed address; value set each step.
        ep_max_tokens_ptr: pointer to a scalar int32 CUDA tensor holding the
            maximum local_tokens across all EP ranks for this iteration. Fixed address;
            value set each step. CTAs with pid >= this value exit immediately. Safe
            because the value is identical on all ranks, so paired CTAs on every rank
            exit together — the barrier for those CTAs is never entered on any rank.
        output_byte_offset: byte offset of this tensor within the symmetric memory buffer.
        HIDDEN_SIZE: hidden dimension, i.e. number of elements per token row (constexpr).
        BLOCK_SIZE: threads per block (constexpr, >= numel_per_token).
        NUMEL_PER_THREAD: elements per thread per load/store, i.e. BITS / element_bits (constexpr).
        BITS: width of each load/store in bits — 128 for activations (bf16) and expert
            indices (int64, always 16-byte aligned for any topk); 64 for routing probs
            (fp32 with topk=6 or topk=22 yields 24/88-byte rows, not 16-byte aligned
            but 8-byte aligned) (constexpr).
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
    """
    pid = tl.program_id(axis=0)

    # Exit before the barrier if this CTA's pid exceeds the iteration maximum.
    # ep_max_tokens is the max over all EP ranks, so all ranks agree on
    # which CTAs exit — the barrier slots for those CTAs are never touched on any rank.
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    local_numel = local_tokens * numel_per_token
    # BLOCK_SIZE is the next power of 2 >= numel_per_token, so it may be larger.
    # channel_mask deactivates the extra padding threads (tid >= numel_per_token).
    channel_mask = tid < numel_per_token

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        for channel_offset in range(0, numel_per_token, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token + channel_offset + tid
            # Two independent masks in orthogonal dimensions:
            #   channel_mask — deactivates power-of-2 padding threads (tid >= numel_per_token).
            #   token_mask   — deactivates overflow threads in the last inner-loop chunk
            #                  when numel_per_token > BLOCK_SIZE and the window
            #                  [channel_offset, channel_offset+BLOCK_SIZE) extends past
            #                  the final token row.
            token_mask = local_offsets < local_numel
            mask = token_mask & channel_mask

            # This rank's tokens start at rank_token_offset in the global output.
            global_offsets = rank_token_offset * numel_per_token + local_offsets

            if BITS == 128:
                # Each 128-bit pack occupies 2 uint64 units; output_byte_offset // 8 converts
                # the tensor's byte offset within the symm-mem buffer to uint64 units.
                # The global offset is multiplied by 2 to convert from 128-bit
                # units to uint64 units.
                multicast_ptrs = (
                    multicast_ptr.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset // 8
                    + global_offsets * 2
                )
                local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
                (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
                st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
            else:
                # Each 64-bit pack is exactly 1 uint64, so offsets index directly (no * 2 stride).
                multicast_ptrs = (
                    multicast_ptr.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset // 8
                    + global_offsets
                )
                local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets
                (x, y) = ld_64(local_ptrs, mask=mask)
                st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

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
def _multimem_reduce_scatter_v_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    input_byte_offset,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    REDUCE_F32: tl.constexpr = False,
):
    """Variable-count multicast reduce-scatter kernel. One CTA processes one token.

    Reads this rank's token shard from the symmetric buffer via multimem.ld_reduce
    (which atomically sums contributions from all EP ranks) and writes the result
    to local memory.

    The barrier runs first — it waits for all ranks to have written their expert
    GEMM outputs into the symmetric buffer before any rank starts reading.

    Args:
        local_ptr: output pointer to this rank's local buffer, shape [local_tokens, hidden_size].
        multicast_ptr: multicast pointer to the symmetric memory buffer holding all expert outputs.
        signal_pad_ptrs: signal pads for barrier synchronization.
        local_tokens: number of tokens this rank owns.
        rank_token_offset_ptr: pointer to a scalar int32 CUDA tensor holding the index of the
            first token this rank owns in the global token sequence. Fixed address; set each step.
        ep_max_tokens_ptr: pointer to a scalar int32 CUDA tensor holding the maximum local_tokens
            across all EP ranks. Fixed address; set each step. CTAs with pid >= this value exit
            immediately — safe because the value is identical on all ranks.
        input_byte_offset: byte offset of the input tensor within the symmetric memory buffer.
        HIDDEN_SIZE: number of elements per token row (constexpr).
        BLOCK_SIZE: threads per block (constexpr, >= numel_per_token).
        NUMEL_PER_THREAD: elements per thread per load/store, i.e. 128 / element_bits (constexpr).
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
    """
    pid = tl.program_id(axis=0)

    # Exit before the barrier if this CTA's pid exceeds the iteration maximum.
    # ep_max_tokens is the max over all EP ranks, so all ranks agree on which
    # CTAs exit — the barrier slots for those CTAs are never touched on any rank.
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Required Triton-3.6 fix (NOT diagnostic): widen raw pointer int args to i64
    # (tt.int_to_ptr requires i64; low VAs get specialized as i32). Value-preserving.
    local_ptr = local_ptr.to(tl.int64)
    multicast_ptr = multicast_ptr.to(tl.int64)

    # Wait for all ranks to have written their expert GEMM outputs to symm_mem
    # before any rank starts the reduce-load.
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

    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    local_numel = local_tokens * numel_per_token
    # channel_mask: deactivates power-of-2 padding threads (tid >= numel_per_token).
    channel_mask = tid < numel_per_token

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        program_offset = token_offset * numel_per_token

        for channel_offset in range(0, numel_per_token, BLOCK_SIZE):
            local_offsets = program_offset + channel_offset + tid
            # Two independent masks in orthogonal dimensions:
            #   channel_mask — deactivates power-of-2 padding threads (tid >= numel_per_token).
            #   token_mask   — deactivates overflow threads in the last inner-loop chunk
            #                  when numel_per_token > BLOCK_SIZE and the window
            #                  [channel_offset, channel_offset+BLOCK_SIZE) extends past
            #                  the final token row.
            token_mask = local_offsets < local_numel
            mask = token_mask & channel_mask

            # This rank's tokens start at rank_token_offset in the global input.
            global_offsets = rank_token_offset * numel_per_token + local_offsets

            # Each 128-bit pack occupies 2 uint64 units; input_byte_offset // 8 converts
            # the tensor's byte offset within the symm-mem buffer to uint64 units.
            multicast_ptrs = (
                multicast_ptr.to(tl.pointer_type(tl.uint64))
                + input_byte_offset // 8
                + global_offsets * 2
            )
            local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2

            (x, y, z, w) = ld_128(
                multicast_ptrs, mask=mask, multicast_op=True, reduce_f32=REDUCE_F32
            )
            st_128(local_ptrs, x, y, z, w, mask=mask, multicast_op=False)


def multimem_reduce_scatter_v(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    input_byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Variable-count multicast reduce-scatter for a single 2-D tensor.

    Reduces expert GEMM outputs across all EP ranks. Each rank reads its owned
    token shard [rank_token_offset : rank_token_offset + local_tokens] from the
    symmetric buffer using multimem.ld_reduce (which atomically sums all ranks'
    contributions), and writes the result to output_tensor.

    Both tensors must be 2-D and 16-byte row-aligned (128-bit path only).
    hidden_size is inferred from output_tensor.shape[1].

    Args:
        output_tensor: local output, shape [local_tokens, hidden_size].
        input_tensor: symmetric memory buffer holding all expert outputs,
            shape [global_tokens, hidden_size].
        symm_mem_hdl: symmetric memory handle for input_tensor.
        rank_token_offset: pre-allocated scalar int32 CUDA tensor. The dispatcher
            writes this rank's token offset into it each step before kernel launch.
        ep_max_tokens: pre-allocated scalar int32 CUDA tensor. The dispatcher writes
            the maximum local_tokens across all EP ranks each step. CTAs with
            pid >= ep_max_tokens exit immediately without entering the barrier.
        per_rank_max_tokens: static int set at model init. Determines the CTA grid size
            as min(per_rank_max_tokens, MAX_NUM_BLOCKS).
        input_byte_offset: byte offset of input_tensor within the symmetric memory
            buffer (for packing multiple tensors into one buffer; 0 otherwise).

    Returns:
        output_tensor populated with this rank's reduced token outputs.
    """
    assert HAVE_TRITON, "Triton is required for multimem reduce-scatter-v."
    assert (
        output_tensor.ndim == 2 and input_tensor.ndim == 2
    ), "output_tensor and input_tensor must be 2-D [tokens, hidden_size]."
    assert is_device_nvls_capable(
        output_tensor.device
    ), "multimem_reduce_scatter_v requires a Hopper+ GPU with NVLink (SM >= 9)."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."
    assert output_tensor.dtype in (
        torch.bfloat16,
        torch.float32,
    ), f"Only bfloat16 and float32 are supported, got {output_tensor.dtype}"
    assert (
        output_tensor.dtype == input_tensor.dtype
    ), f"output and input dtype mismatch: {output_tensor.dtype} vs {input_tensor.dtype}"

    hidden_size = output_tensor.shape[1]
    assert (
        input_tensor.shape[1] == hidden_size
    ), f"input and output hidden_size mismatch: {input_tensor.shape[1]} vs {hidden_size}"
    row_bytes = hidden_size * output_tensor.element_size()
    assert row_bytes % 16 == 0, (
        f"Row size ({hidden_size} elements × {output_tensor.element_size()} bytes) = "
        f"{row_bytes} bytes is not 16-byte aligned; RSV requires 128-bit alignment."
    )

    # Hardcoded to 148 (B200 SM count; raised from upstream Megatron's 128). One CTA
    # processes one token, so num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS) bounds
    # how many SMs the comm occupies. Callers may override via max_num_blocks; the bencher
    # fixes NVLS at 148 (see bench/README.md).
    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    MAX_BLOCK_SIZE = 1024
    WARP_SIZE = 32

    local_tokens = output_tensor.shape[0]
    numel_per_thread = 128 // (output_tensor.element_size() * 8)
    numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread

    block_size = min(triton.next_power_of_2(numel_per_token), MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS)

    reduce_f32 = output_tensor.dtype == torch.float32
    _multimem_reduce_scatter_v_kernel[(num_blocks, 1, 1)](
        output_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        input_byte_offset=input_byte_offset,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        REDUCE_F32=reduce_f32,
        num_warps=num_warps,
    )

    return output_tensor


@triton.jit
def _multimem_all_gatherv_3tensor_kernel(
    local_ptr_0,
    multicast_ptr_0,
    output_byte_offset_0,
    local_ptr_1,
    multicast_ptr_1,
    output_byte_offset_1,
    local_ptr_2,
    multicast_ptr_2,
    output_byte_offset_2,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    HIDDEN_SIZE_0: tl.constexpr,
    HIDDEN_SIZE_1: tl.constexpr,
    HIDDEN_SIZE_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD_0: tl.constexpr,
    NUMEL_PER_THREAD_1: tl.constexpr,
    NUMEL_PER_THREAD_2: tl.constexpr,
    BITS_0: tl.constexpr,
    BITS_1: tl.constexpr,
    BITS_2: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Variable-count multicast all-gather for three tensors in a single kernel.

    Identical semantics to _multimem_all_gather_v_kernel but processes three
    tensors per CTA iteration, sharing a single barrier. This avoids launching
    three separate kernels (and three separate barriers) for the common case
    of gathering hidden states, routing probabilities, and expert indices together.

    The outer token loop is shared across all three tensors; each tensor has its
    own inner channel loop with independent masking. BLOCK_SIZE is the maximum
    of the three per-tensor block sizes — smaller tensors mask out the extra threads
    via channel_mask.

    signal_pad_ptrs from the first output buffer's symmetric memory handle are used
    for the single end-of-kernel barrier. Since all three writes complete before the
    barrier, a single sync suffices for all three tensors.

    Args:
        local_ptr_0/1/2: pointers to each rank's local input for tensors 0/1/2.
        multicast_ptr_0/1/2: multicast pointers to the output symmetric memory buffers.
        output_byte_offset_0/1/2: byte offsets of each tensor within its symmetric
            memory buffer (0 when the buffer holds only that tensor).
        signal_pad_ptrs: signal pads from symm_mem_hdl_0, used for the single barrier.
        local_tokens: number of tokens this rank contributes (shared across tensors).
        rank_token_offset_ptr: pointer to a scalar int32 CUDA tensor holding this rank's
            write offset in the global output (prefix sum over lower-ranked EP ranks).
        ep_max_tokens_ptr: pointer to a scalar int32 CUDA tensor holding the maximum
            local_tokens across all EP ranks. CTAs with pid >= this value exit immediately.
        HIDDEN_SIZE_0/1/2: hidden dimension (elements per token row) for each tensor (constexpr).
        BLOCK_SIZE: threads per block — max of the three per-tensor block sizes (constexpr).
        NUMEL_PER_THREAD_0/1/2: elements per thread per load/store for each tensor (constexpr).
        BITS_0/1/2: load/store width in bits (128 or 64) for each tensor (constexpr).
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
    """
    pid = tl.program_id(axis=0)

    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Required Triton-3.6 fix (NOT diagnostic): raw pointer int args are specialized
    # as i32 when a GPU VA fits in 32 bits, but tt.int_to_ptr requires i64 -> compile
    # error. Widen to i64 (value-preserving). Without this, this kernel does not compile.
    local_ptr_0 = local_ptr_0.to(tl.int64)
    multicast_ptr_0 = multicast_ptr_0.to(tl.int64)
    local_ptr_1 = local_ptr_1.to(tl.int64)
    multicast_ptr_1 = multicast_ptr_1.to(tl.int64)
    local_ptr_2 = local_ptr_2.to(tl.int64)
    multicast_ptr_2 = multicast_ptr_2.to(tl.int64)

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    numel_per_token_0 = tl.cdiv(HIDDEN_SIZE_0, NUMEL_PER_THREAD_0)
    numel_per_token_1 = tl.cdiv(HIDDEN_SIZE_1, NUMEL_PER_THREAD_1)
    numel_per_token_2 = tl.cdiv(HIDDEN_SIZE_2, NUMEL_PER_THREAD_2)

    local_numel_0 = local_tokens * numel_per_token_0
    local_numel_1 = local_tokens * numel_per_token_1
    local_numel_2 = local_tokens * numel_per_token_2

    # channel_mask: deactivates threads beyond each tensor's numel_per_token (power-of-2 padding).
    channel_mask_0 = tid < numel_per_token_0
    channel_mask_1 = tid < numel_per_token_1
    channel_mask_2 = tid < numel_per_token_2

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        # --- Tensor 0 ---
        for channel_offset in range(0, numel_per_token_0, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token_0 + channel_offset + tid
            token_mask = local_offsets < local_numel_0
            mask = token_mask & channel_mask_0
            global_offsets = rank_token_offset * numel_per_token_0 + local_offsets
            if BITS_0 == 128:
                multicast_ptrs = (
                    multicast_ptr_0.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_0 // 8
                    + global_offsets * 2
                )
                local_ptrs = local_ptr_0.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
                (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
                st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
            else:
                multicast_ptrs = (
                    multicast_ptr_0.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_0 // 8
                    + global_offsets
                )
                local_ptrs = local_ptr_0.to(tl.pointer_type(tl.uint64)) + local_offsets
                (x, y) = ld_64(local_ptrs, mask=mask)
                st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

        # --- Tensor 1 ---
        for channel_offset in range(0, numel_per_token_1, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token_1 + channel_offset + tid
            token_mask = local_offsets < local_numel_1
            mask = token_mask & channel_mask_1
            global_offsets = rank_token_offset * numel_per_token_1 + local_offsets
            if BITS_1 == 128:
                multicast_ptrs = (
                    multicast_ptr_1.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_1 // 8
                    + global_offsets * 2
                )
                local_ptrs = local_ptr_1.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
                (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
                st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
            else:
                multicast_ptrs = (
                    multicast_ptr_1.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_1 // 8
                    + global_offsets
                )
                local_ptrs = local_ptr_1.to(tl.pointer_type(tl.uint64)) + local_offsets
                (x, y) = ld_64(local_ptrs, mask=mask)
                st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

        # --- Tensor 2 ---
        for channel_offset in range(0, numel_per_token_2, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token_2 + channel_offset + tid
            token_mask = local_offsets < local_numel_2
            mask = token_mask & channel_mask_2
            global_offsets = rank_token_offset * numel_per_token_2 + local_offsets
            if BITS_2 == 128:
                multicast_ptrs = (
                    multicast_ptr_2.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_2 // 8
                    + global_offsets * 2
                )
                local_ptrs = local_ptr_2.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
                (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
                st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
            else:
                multicast_ptrs = (
                    multicast_ptr_2.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_2 // 8
                    + global_offsets
                )
                local_ptrs = local_ptr_2.to(tl.pointer_type(tl.uint64)) + local_offsets
                (x, y) = ld_64(local_ptrs, mask=mask)
                st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multimem_all_gather_v(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    output_byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Variable-count multicast all-gather for a single 2-D tensor.

    Gathers [local_tokens, hidden_size] from each EP rank into a shared
    output_tensor of shape [global_tokens, hidden_size], where global_tokens is
    the sum of all ranks' local_tokens. Each rank writes its slice starting at
    rank_token_offset in the output.

    Both tensors must be 2-D; hidden_size is inferred from input_tensor.shape[1].
    The 128-bit or 64-bit NVLS path is selected automatically based on row alignment.

    Args:
        output_tensor: symmetric memory buffer, shape [global_tokens, hidden_size].
        input_tensor: this rank's local input, shape [local_tokens, hidden_size].
        symm_mem_hdl: symmetric memory handle for output_tensor.
        rank_token_offset: pre-allocated scalar int32 CUDA tensor. The dispatcher
            writes this rank's token offset (prefix sum over lower-ranked EP ranks)
            into it each step before kernel launch.
        ep_max_tokens: pre-allocated scalar int32 CUDA tensor. The dispatcher writes
            the maximum local_tokens across all EP ranks into it each step. CTAs with
            pid >= ep_max_tokens exit immediately — safe because all ranks agree on
            this value, so the corresponding CTAs exit on every rank simultaneously.
        per_rank_max_tokens: static int set at model init. Determines the CTA grid size
            as min(per_rank_max_tokens, MAX_NUM_BLOCKS). Typically > MAX_NUM_BLOCKS so
            we always launch MAX_NUM_BLOCKS CTAs.
        output_byte_offset: byte offset of this tensor within the symmetric memory buffer
            (for packing multiple tensors into one buffer; 0 if the buffer holds only
            this tensor).

    Returns:
        output_tensor with all ranks' data written.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather-v."
    assert input_tensor.ndim == 2 and output_tensor.ndim == 2, (
        f"input_tensor and output_tensor must be 2-D [tokens, hidden_size], "
        f"got input_tensor.shape={input_tensor.shape}, output_tensor.shape={output_tensor.shape}."
    )
    assert is_device_nvls_capable(
        input_tensor.device
    ), "multimem_all_gather_v requires a Hopper+ GPU with NVLink (SM >= 9)."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."

    hidden_size = input_tensor.shape[1]
    assert (
        input_tensor.shape[1] == output_tensor.shape[1]
    ), f"input and output hidden_size mismatch: {input_tensor.shape[1]} vs {output_tensor.shape[1]}"

    row_bytes = hidden_size * input_tensor.element_size()
    assert row_bytes % 8 == 0, (
        f"Row size ({hidden_size} elements × {input_tensor.element_size()} bytes) = "
        f"{row_bytes} bytes is not 8-byte aligned; cannot use NVLS."
    )
    bits = 128 if row_bytes % 16 == 0 else 64

    # Hardcoded to 148 (B200 SM count; raised from upstream Megatron's 128). One CTA
    # processes one token, so num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS) bounds
    # how many SMs the comm occupies. Callers may override via max_num_blocks; the bencher
    # fixes NVLS at 148 (see bench/README.md).
    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    MAX_BLOCK_SIZE = 1024
    WARP_SIZE = 32

    local_tokens = input_tensor.shape[0]
    numel_per_thread = bits // (input_tensor.element_size() * 8)
    numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread

    # BLOCK_SIZE must be a constexpr and >= numel_per_token; round up to next power of 2.
    block_size = min(triton.next_power_of_2(numel_per_token), MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // WARP_SIZE)

    # All ranks launch the same fixed number of CTAs. CTAs with
    # pid >= ep_max_tokens exit immediately at kernel entry.
    num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS)

    _multimem_all_gather_v_kernel[(num_blocks, 1, 1)](
        input_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        output_byte_offset=output_byte_offset,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        BITS=bits,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )

    return output_tensor


def multimem_all_gatherv_3tensor(
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
    """Variable-count multicast all-gather for three tensors in a single kernel launch.

    Gathers three independent [local_tokens, hidden_size_i] tensors from every EP rank
    into their respective output symmetric memory buffers in one fused kernel, sharing a
    single end-of-kernel barrier. This is more efficient than calling multimem_all_gather_v
    three times because the barrier cost (one per kernel) is paid only once.

    All three input tensors must share the same local_tokens dimension (i.e. the same
    number of token rows per rank). Each tensor may have a different hidden_size and dtype.
    The 128-bit or 64-bit NVLS path is selected independently per tensor based on row
    alignment.

    The barrier at the end of the kernel uses signal_pad_ptrs from symm_mem_hdl_0. Since
    all three multicast stores complete before the barrier, a single sync covers all three
    tensors. All three handles must belong to the same EP group (identical rank/world_size).

    Args:
        output_tensor_0/1/2: symmetric memory buffers for each tensor,
            shape [global_tokens, hidden_size_i].
        input_tensor_0/1/2: this rank's local inputs, shape [local_tokens, hidden_size_i].
        symm_mem_hdl_0/1/2: symmetric memory handles for each output buffer.
            signal_pad_ptrs from hdl_0 are used for the single end-of-kernel barrier.
        rank_token_offset: pre-allocated scalar int32 CUDA tensor. The dispatcher writes
            this rank's token offset (prefix sum over lower-ranked EP ranks) each step.
        ep_max_tokens: pre-allocated scalar int32 CUDA tensor. The dispatcher writes the
            maximum local_tokens across all EP ranks each step. CTAs with
            pid >= ep_max_tokens exit immediately — safe because all ranks agree.
        per_rank_max_tokens: static int set at model init. Determines the CTA grid size as
            min(per_rank_max_tokens, MAX_NUM_BLOCKS).
        output_byte_offset_0/1/2: byte offset of each tensor within its symmetric memory
            buffer (for packing multiple tensors into one buffer; 0 otherwise).

    Returns:
        Tuple of (output_tensor_0, output_tensor_1, output_tensor_2) with all ranks'
        data written.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather-v3."
    for i, (inp, out) in enumerate(
        zip(
            (input_tensor_0, input_tensor_1, input_tensor_2),
            (output_tensor_0, output_tensor_1, output_tensor_2),
        )
    ):
        assert inp.ndim == 2 and out.ndim == 2, (
            f"input_tensor_{i} and output_tensor_{i} must be 2-D [tokens, hidden_size], "
            f"got input_tensor_{i}.shape={inp.shape}, output_tensor_{i}.shape={out.shape}."
        )
        assert inp.shape[1] == out.shape[1], (
            f"input_tensor_{i} and output_tensor_{i} hidden_size mismatch: "
            f"{inp.shape[1]} vs {out.shape[1]}."
        )
    assert (
        input_tensor_0.shape[0] == input_tensor_1.shape[0] == input_tensor_2.shape[0]
    ), "All three input tensors must have the same local_tokens (first dimension)."
    assert is_device_nvls_capable(
        input_tensor_0.device
    ), "multimem_all_gatherv_3tensor requires a Hopper+ GPU with NVLink (SM >= 9)."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."
    assert (
        symm_mem_hdl_0.rank == symm_mem_hdl_1.rank == symm_mem_hdl_2.rank
    ), "All three symmetric memory handles must belong to the same EP group (rank mismatch)."
    assert (
        symm_mem_hdl_0.world_size == symm_mem_hdl_1.world_size == symm_mem_hdl_2.world_size
    ), "All three symmetric memory handles must belong to the same EP group (world_size mismatch)."

    # Hardcoded to 148 (B200 SM count; raised from upstream Megatron's 128). One CTA
    # processes one token, so num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS) bounds
    # how many SMs the comm occupies. Callers may override via max_num_blocks; the bencher
    # fixes NVLS at 148 (see bench/README.md).
    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    MAX_BLOCK_SIZE = 1024
    WARP_SIZE = 32

    local_tokens = input_tensor_0.shape[0]

    def _tensor_params(inp):
        hidden_size = inp.shape[1]
        row_bytes = hidden_size * inp.element_size()
        assert row_bytes % 8 == 0, (
            f"Row size ({hidden_size} elements × {inp.element_size()} bytes) = "
            f"{row_bytes} bytes is not 8-byte aligned; cannot use NVLS."
        )
        bits = 128 if row_bytes % 16 == 0 else 64
        numel_per_thread = bits // (inp.element_size() * 8)
        numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread
        block_size = min(triton.next_power_of_2(numel_per_token), MAX_BLOCK_SIZE)
        return hidden_size, bits, numel_per_thread, block_size

    hidden_size_0, bits_0, numel_per_thread_0, block_size_0 = _tensor_params(input_tensor_0)
    hidden_size_1, bits_1, numel_per_thread_1, block_size_1 = _tensor_params(input_tensor_1)
    hidden_size_2, bits_2, numel_per_thread_2, block_size_2 = _tensor_params(input_tensor_2)

    # Use the largest block size so all threads are occupied for at least one tensor;
    # smaller tensors mask out excess threads via channel_mask inside the kernel.
    block_size = max(block_size_0, block_size_1, block_size_2)
    num_warps = max(1, block_size // WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS)

    _multimem_all_gatherv_3tensor_kernel[(num_blocks, 1, 1)](
        input_tensor_0.data_ptr(),
        symm_mem_hdl_0.multicast_ptr,
        output_byte_offset_0,
        input_tensor_1.data_ptr(),
        symm_mem_hdl_1.multicast_ptr,
        output_byte_offset_1,
        input_tensor_2.data_ptr(),
        symm_mem_hdl_2.multicast_ptr,
        output_byte_offset_2,
        symm_mem_hdl_0.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        HIDDEN_SIZE_0=hidden_size_0,
        HIDDEN_SIZE_1=hidden_size_1,
        HIDDEN_SIZE_2=hidden_size_2,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD_0=numel_per_thread_0,
        NUMEL_PER_THREAD_1=numel_per_thread_1,
        NUMEL_PER_THREAD_2=numel_per_thread_2,
        BITS_0=bits_0,
        BITS_1=bits_1,
        BITS_2=bits_2,
        RANK=symm_mem_hdl_0.rank,
        WORLD_SIZE=symm_mem_hdl_0.world_size,
        num_warps=num_warps,
    )

    return output_tensor_0, output_tensor_1, output_tensor_2


# ══════════════════════════════════════════════════════════════════════════════
# All-to-all-v collectives (dense layout, routing-driven unicast)
# ══════════════════════════════════════════════════════════════════════════════
#
# These mirror the all-gather-v / reduce-scatter-v kernels above but move the HIDDEN
# activations by *unicast* to only a token's destination ranks, instead of multicasting
# to every rank. Everything else is kept identical to the NVLS path so the surrounding
# harness (metadata, dense layout, vLLM compute) is untouched:
#
#   * DENSE layout: a token from this rank at local index t lands at the SAME global
#     offset `rank_token_offset + t` on every destination rank (source-based, never
#     compacted), exactly like AGV. Only the store TARGET changes: the multicast pointer
#     (fans out to all ranks) becomes `buffer_ptrs_dev[d]` for each destination rank d
#     (fans out to none — we place each copy ourselves).
#   * ROUTING + PROBS stay full all-gather-v (multicast): every rank sees every token's
#     routing so the compute writes 0-or-sum everywhere and combine works unchanged.
#   * DESTINATION ranks are derived on-device from a token's top-k experts
#     (expert // experts_per_rank), deduplicated implicitly: `routes_to_d` is a
#     block-reduction over the token's experts, so each rank is considered once. A
#     non-destination rank's store/load is predicated off (all-lane-false mask) and
#     therefore generates NO NVLink traffic (the p2p asm skips the memory op per lane).


@triton.jit
def _pack_bf16x2(hi, lo):
    """Pack two fp32 blocks into one uint32 block of bf16x2 (round-to-nearest).

    hi -> bits [31:16], lo -> bits [15:0]. Mirrors the pack in fused_collectives'
    apply_norm; used to store the fp32-accumulated pull-combine result as bf16.
    """
    hi_u = (hi.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)) << 16
    lo_u = lo.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    return hi_u | lo_u


@triton.jit
def _unpack_bf16x2(x, mask):
    """Unpack a uint32 block of bf16x2 into (hi_fp32, lo_fp32); masked lanes -> 0.

    Local copy of fused_collectives.unpack_bf16x2 (kept here to avoid a cross-module
    import). `x * mask` forces masked-off / non-destination lanes to 0 so they contribute
    nothing to the accumulator.
    """
    x = x * mask
    x_hi = (x >> 16).cast(tl.uint16).cast(tl.bfloat16, bitcast=True).cast(tl.float32)
    x_lo = x.cast(tl.uint16).cast(tl.bfloat16, bitcast=True).cast(tl.float32)
    return x_hi, x_lo


@triton.jit
def _or_combine(a, b):
    """Bitwise-OR reduction operator for tl.reduce (folds a token's per-expert
    destination-rank bits into a single WORLD_SIZE-wide mask)."""
    return a | b


@triton.jit
def _multimem_a2av_dispatch_3tensor_kernel(
    local_ptr_h,
    buffer_ptrs_h,
    output_byte_offset_h,
    local_ptr_r,
    multicast_ptr_r,
    output_byte_offset_r,
    local_ptr_p,
    multicast_ptr_p,
    output_byte_offset_p,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    HIDDEN_SIZE_H: tl.constexpr,
    HIDDEN_SIZE_R: tl.constexpr,
    HIDDEN_SIZE_P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD_H: tl.constexpr,
    NUMEL_PER_THREAD_R: tl.constexpr,
    NUMEL_PER_THREAD_P: tl.constexpr,
    BITS_R: tl.constexpr,
    BITS_P: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """All-to-all-v dispatch: unicast HIDDEN to a token's destination ranks; multicast
    (all-gather-v) ROUTING and PROBS to every rank. One CTA per token, persistent grid.

    HIDDEN is always the 128-bit path (row is 16-byte aligned). ROUTING/PROBS pick 128 or
    64 bits per their row alignment (BITS_R / BITS_P), matching the AGV-3tensor kernel. A
    single end-of-kernel barrier (release/acquire) publishes all writes.
    """
    pid = tl.program_id(axis=0)
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Required Triton-3.6 fix: widen raw pointer int args to i64 (tt.int_to_ptr needs i64).
    local_ptr_h = local_ptr_h.to(tl.int64)
    buffer_ptrs_h = buffer_ptrs_h.to(tl.int64)
    local_ptr_r = local_ptr_r.to(tl.int64)
    multicast_ptr_r = multicast_ptr_r.to(tl.int64)
    local_ptr_p = local_ptr_p.to(tl.int64)
    multicast_ptr_p = multicast_ptr_p.to(tl.int64)

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)

    numel_per_token_h = tl.cdiv(HIDDEN_SIZE_H, NUMEL_PER_THREAD_H)
    numel_per_token_r = tl.cdiv(HIDDEN_SIZE_R, NUMEL_PER_THREAD_R)
    numel_per_token_p = tl.cdiv(HIDDEN_SIZE_P, NUMEL_PER_THREAD_P)
    local_numel_h = local_tokens * numel_per_token_h
    local_numel_r = local_tokens * numel_per_token_r
    local_numel_p = local_tokens * numel_per_token_p
    channel_mask_h = tid < numel_per_token_h
    channel_mask_r = tid < numel_per_token_r
    channel_mask_p = tid < numel_per_token_p

    # Per-rank base pointers of the HIDDEN symmetric buffer (int64 array), and this rank's
    # local routing rows (int64 expert ids) used to derive destination ranks.
    buffer_ptrs_h_i64 = buffer_ptrs_h.to(tl.pointer_type(tl.int64))
    routing_row_ptr = local_ptr_r.to(tl.pointer_type(tl.int64))

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        # --- Destination ranks for this token (dedup implicit via routes_to_d) ---
        experts = tl.load(routing_row_ptr + token_offset * TOPK + tid, mask=tid < TOPK, other=-1)
        dest = tl.where(experts >= 0, experts // EXPERTS_PER_RANK, -1)

        # --- HIDDEN: all-to-all-v unicast (load each 128-bit chunk once, send to each dest) ---
        for channel_offset in range(0, numel_per_token_h, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token_h + channel_offset + tid
            token_mask = local_offsets < local_numel_h
            mask = token_mask & channel_mask_h
            global_offsets = rank_token_offset * numel_per_token_h + local_offsets
            local_ptrs = local_ptr_h.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
            (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
            for d in range(WORLD_SIZE):
                routes_to_d = tl.max(tl.where(dest == d, 1, 0)) == 1
                send_mask = mask & routes_to_d  # all-false (no NVLink traffic) if not a dest
                peer_base = tl.load(buffer_ptrs_h_i64 + d)
                peer_ptrs = (
                    peer_base.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_h // 8
                    + global_offsets * 2
                )
                st_128_p2p(peer_ptrs, x, y, z, w, mask=send_mask)

        # --- ROUTING: all-gather-v multicast ---
        for channel_offset in range(0, numel_per_token_r, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token_r + channel_offset + tid
            token_mask = local_offsets < local_numel_r
            mask = token_mask & channel_mask_r
            global_offsets = rank_token_offset * numel_per_token_r + local_offsets
            if BITS_R == 128:
                multicast_ptrs = (
                    multicast_ptr_r.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_r // 8
                    + global_offsets * 2
                )
                local_ptrs = local_ptr_r.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
                (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
                st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
            else:
                multicast_ptrs = (
                    multicast_ptr_r.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_r // 8
                    + global_offsets
                )
                local_ptrs = local_ptr_r.to(tl.pointer_type(tl.uint64)) + local_offsets
                (x, y) = ld_64(local_ptrs, mask=mask)
                st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

        # --- PROBS: all-gather-v multicast ---
        for channel_offset in range(0, numel_per_token_p, BLOCK_SIZE):
            local_offsets = token_offset * numel_per_token_p + channel_offset + tid
            token_mask = local_offsets < local_numel_p
            mask = token_mask & channel_mask_p
            global_offsets = rank_token_offset * numel_per_token_p + local_offsets
            if BITS_P == 128:
                multicast_ptrs = (
                    multicast_ptr_p.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_p // 8
                    + global_offsets * 2
                )
                local_ptrs = local_ptr_p.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
                (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
                st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
            else:
                multicast_ptrs = (
                    multicast_ptr_p.to(tl.pointer_type(tl.uint64))
                    + output_byte_offset_p // 8
                    + global_offsets
                )
                local_ptrs = local_ptr_p.to(tl.pointer_type(tl.uint64)) + local_offsets
                (x, y) = ld_64(local_ptrs, mask=mask)
                st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multimem_a2av_dispatch_3tensor(
    output_tensor_h: torch.Tensor,
    output_tensor_r: torch.Tensor,
    output_tensor_p: torch.Tensor,
    input_tensor_h: torch.Tensor,
    input_tensor_r: torch.Tensor,
    input_tensor_p: torch.Tensor,
    symm_mem_hdl_h: _SymmetricMemory,
    symm_mem_hdl_r: _SymmetricMemory,
    symm_mem_hdl_p: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    num_experts: int,
    output_byte_offset_h: int = 0,
    output_byte_offset_r: int = 0,
    output_byte_offset_p: int = 0,
    **kwargs,
) -> tuple:
    """All-to-all-v dispatch of HIDDEN + all-gather-v of ROUTING/PROBS in one kernel/barrier.

    HIDDEN (input_tensor_h, bf16) is unicast to each token's destination ranks using
    symm_mem_hdl_h.buffer_ptrs_dev (per-rank base pointers of the hidden symmetric buffer).
    ROUTING (input_tensor_r, int64 expert ids) and PROBS (input_tensor_p, fp32) are multicast
    to every rank exactly as multimem_all_gatherv_3tensor does. Destination ranks are derived
    on-device from routing: expert // (num_experts // world_size).

    Layout is DENSE and identical to AGV: this rank's token t -> global offset
    rank_token_offset + t on every destination rank.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-to-all-v dispatch."
    assert (
        input_tensor_h.ndim == 2 and input_tensor_r.ndim == 2 and input_tensor_p.ndim == 2
    ), "inputs must be 2-D [tokens, hidden]."
    assert is_device_nvls_capable(
        input_tensor_h.device
    ), "multimem_a2av_dispatch_3tensor requires a Hopper+ GPU with NVLink (SM >= 9)."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."
    assert hasattr(symm_mem_hdl_h, "buffer_ptrs_dev"), (
        "symmetric-memory handle has no buffer_ptrs_dev; the installed torch build does not "
        "expose per-rank symmetric pointers required for all-to-all-v unicast."
    )

    world_size = symm_mem_hdl_h.world_size
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size."
    experts_per_rank = num_experts // world_size
    topk = input_tensor_r.shape[1]

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    MAX_BLOCK_SIZE = 1024
    WARP_SIZE = 32

    local_tokens = input_tensor_h.shape[0]

    # HIDDEN: 128-bit path only (the p2p unicast primitive is 128-bit).
    hidden_h = input_tensor_h.shape[1]
    row_bytes_h = hidden_h * input_tensor_h.element_size()
    assert row_bytes_h % 16 == 0, (
        f"Hidden row ({hidden_h} x {input_tensor_h.element_size()}B = {row_bytes_h}B) must be "
        f"16-byte aligned for the all-to-all-v 128-bit path."
    )
    numel_per_thread_h = 128 // (input_tensor_h.element_size() * 8)
    numel_per_token_h = (hidden_h + numel_per_thread_h - 1) // numel_per_thread_h
    block_size_h = min(triton.next_power_of_2(numel_per_token_h), MAX_BLOCK_SIZE)

    def _agv_params(inp):
        hidden = inp.shape[1]
        row_bytes = hidden * inp.element_size()
        assert row_bytes % 8 == 0, "AGV tensor row must be 8-byte aligned."
        bits = 128 if row_bytes % 16 == 0 else 64
        npt = bits // (inp.element_size() * 8)
        numel_per_token = (hidden + npt - 1) // npt
        block_size = min(triton.next_power_of_2(numel_per_token), MAX_BLOCK_SIZE)
        return hidden, bits, npt, block_size

    hidden_r, bits_r, npt_r, block_size_r = _agv_params(input_tensor_r)
    hidden_p, bits_p, npt_p, block_size_p = _agv_params(input_tensor_p)

    # Block must cover the widest tensor AND the top-k lanes (experts are read into lanes < TOPK).
    block_size = max(block_size_h, block_size_r, block_size_p, triton.next_power_of_2(topk))
    num_warps = max(1, block_size // WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS)

    _multimem_a2av_dispatch_3tensor_kernel[(num_blocks, 1, 1)](
        input_tensor_h.data_ptr(),
        symm_mem_hdl_h.buffer_ptrs_dev,
        output_byte_offset_h,
        input_tensor_r.data_ptr(),
        symm_mem_hdl_r.multicast_ptr,
        output_byte_offset_r,
        input_tensor_p.data_ptr(),
        symm_mem_hdl_p.multicast_ptr,
        output_byte_offset_p,
        symm_mem_hdl_h.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        HIDDEN_SIZE_H=hidden_h,
        HIDDEN_SIZE_R=hidden_r,
        HIDDEN_SIZE_P=hidden_p,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD_H=numel_per_thread_h,
        NUMEL_PER_THREAD_R=npt_r,
        NUMEL_PER_THREAD_P=npt_p,
        BITS_R=bits_r,
        BITS_P=bits_p,
        TOPK=topk,
        EXPERTS_PER_RANK=experts_per_rank,
        RANK=symm_mem_hdl_h.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output_tensor_h, output_tensor_r, output_tensor_p


@triton.jit
def _multimem_a2av_pull_combine_kernel(
    output_ptr,
    buffer_ptrs_out,
    routing_local_ptr,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    input_byte_offset,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """All-to-all-v pull combine: for each of THIS rank's tokens, read its expert output
    from only its destination ranks' symmetric output buffers (at the same dense global
    offset), sum in fp32, and write the bf16 result to local memory.

    Mirror image of the RSV kernel: instead of a switch-reduce over ALL peers, it pulls
    from `<= topk` deduplicated destination ranks via per-peer unicast loads. The barrier
    runs FIRST (wait for all ranks' expert outputs); there is no end barrier (the next
    dispatch's end barrier interlocks). fp32 accumulation matches NVLS RSV's acc::f32.
    """
    pid = tl.program_id(axis=0)
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    # Required Triton-3.6 fix: widen raw pointer int args to i64.
    output_ptr = output_ptr.to(tl.int64)
    buffer_ptrs_out = buffer_ptrs_out.to(tl.int64)
    routing_local_ptr = routing_local_ptr.to(tl.int64)

    # Wait for all ranks to have written their expert outputs before pulling.
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
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    local_numel = local_tokens * numel_per_token
    channel_mask = tid < numel_per_token

    buffer_ptrs_out_i64 = buffer_ptrs_out.to(tl.pointer_type(tl.int64))
    routing_row_ptr = routing_local_ptr.to(tl.pointer_type(tl.int64))

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        experts = tl.load(routing_row_ptr + token_offset * TOPK + tid, mask=tid < TOPK, other=-1)
        dest = tl.where(experts >= 0, experts // EXPERTS_PER_RANK, -1)
        # Destination-rank bitmask for this token, computed ONCE (WORLD_SIZE <= 64 -> uint64).
        # Replaces the per-rank block reduction `tl.max(tl.where(dest==d,...))` that used to run
        # INSIDE the d-loop below -- that was WORLD_SIZE CTA-wide barriers per token, which
        # serialized the remote pulls so each of the <= topk real loads paid full NVLink latency
        # in series. With the mask precomputed, the d-loop's destination test is a barrier-free
        # bit lookup, so the real pulls can pipeline (overlap latency). Rank dedup is automatic
        # (OR): a rank hit by several of the token's experts still contributes one set bit.
        safe_dest = tl.where(dest >= 0, dest, 0).to(tl.uint64)
        dest_bit = tl.where(dest >= 0,
                            tl.full([BLOCK_SIZE], 1, tl.uint64) << safe_dest,
                            tl.zeros([BLOCK_SIZE], tl.uint64))
        dest_mask = tl.reduce(dest_bit, 0, _or_combine)  # scalar uint64; ONE reduction per token
        program_offset = token_offset * numel_per_token
        for channel_offset in range(0, numel_per_token, BLOCK_SIZE):
            local_offsets = program_offset + channel_offset + tid
            token_mask = local_offsets < local_numel
            mask = token_mask & channel_mask
            global_offsets = rank_token_offset * numel_per_token + local_offsets

            # fp32 accumulators: one (hi, lo) pair per 32-bit word of the 128-bit chunk.
            acc_x_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_x_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_y_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_y_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_z_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_z_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_w_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            acc_w_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

            for d in range(WORLD_SIZE):
                routes_to_d = ((dest_mask >> d) & 1) == 1  # barrier-free bit test (no reduction)
                pull_mask = mask & routes_to_d  # all-false (no NVLink traffic) if not a dest
                peer_base = tl.load(buffer_ptrs_out_i64 + d)
                peer_ptrs = (
                    peer_base.to(tl.pointer_type(tl.uint64))
                    + input_byte_offset // 8
                    + global_offsets * 2
                )
                (x, y, z, w) = ld_128_p2p(peer_ptrs, mask=pull_mask)
                x_hi, x_lo = _unpack_bf16x2(x, pull_mask)
                y_hi, y_lo = _unpack_bf16x2(y, pull_mask)
                z_hi, z_lo = _unpack_bf16x2(z, pull_mask)
                w_hi, w_lo = _unpack_bf16x2(w, pull_mask)
                acc_x_hi += x_hi
                acc_x_lo += x_lo
                acc_y_hi += y_hi
                acc_y_lo += y_lo
                acc_z_hi += z_hi
                acc_z_lo += z_lo
                acc_w_hi += w_hi
                acc_w_lo += w_lo

            out_x = _pack_bf16x2(acc_x_hi, acc_x_lo)
            out_y = _pack_bf16x2(acc_y_hi, acc_y_lo)
            out_z = _pack_bf16x2(acc_z_hi, acc_z_lo)
            out_w = _pack_bf16x2(acc_w_hi, acc_w_lo)
            local_ptrs = output_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
            st_128(local_ptrs, out_x, out_y, out_z, out_w, mask=mask, multicast_op=False)


# ── Push combine (Idea 2: compact token lists built during dispatch) ──────────
#
# The first push design made every destination rank re-derive validity from the
# all-gathered routing: for each of the `valid_tokens` global slots it loaded TOPK
# expert ids and ran two block reductions (an OR-reduce for the destination bitmask
# and a range search for the owning source rank) *between* consecutive p2p stores.
# Two things came out of that, and both are fixed here.
#
# 1. Idea 2 -- move the routing work off the data path.  A cheap builder kernel runs
#    once per dispatch and publishes, into every destination rank d, a COMPACT list
#    of the local token indices that rank R is sending to d:
#
#      recv_list [R, j * SEG + i] = local token index (relative to R) of the i-th
#                                   token of R's segment j that routes to d
#      recv_count[R, j]           = how many entries segment j has (0 .. SEG)
#
#    A "segment" is a fixed SEG-token slice of R's local tokens, so the compaction
#    is purely intra-segment: no atomics, no global prefix sum, no zero-fill between
#    steps, and the slot a token occupies is a pure function of (R, j).  The push
#    kernel reads one scalar count per SEG tokens and then streams listed tokens
#    with zero per-token metadata work.  The builder also writes a per-token
#    destination BITMASK (`dest_mask`, one int64 per local token) that the receive
#    kernel consumes in place of its own TOPK reduction.
#
# 2. Feed NVLink enough outstanding traffic.  The old kernel ran 148 CTAs x 128
#    threads and had each thread issue one 128-bit load, wait, then one store --
#    about 2 KB in flight per CTA.  DeepEP's intranode kernels run one ~1024-thread
#    CTA per SM and keep 4-5 independent 16-byte loads per thread, issuing every
#    load before any store (`UNROLLED_WARP_COPY`), for ~60 KB in flight per CTA.
#    Below, a CTA's work tile is A2AV_UNROLL * 1024 lanes wide while the launch uses
#    1024 threads, so Triton hands each thread A2AV_UNROLL independent chunks and
#    emits them as a load batch followed by a store batch -- the same shape, ~64 KB
#    of loads in flight per CTA.  The lane space is split as
#    (TPT tokens) x (NPT_P2 channels) with a power-of-two channel stride, which keeps
#    each warp inside one token row so the 128-bit accesses stay fully coalesced.
#
# Layout of combine_recv:  [WORLD_SIZE, per_rank_max_tokens, H]  bf16, symmetric.
#   Slot [d, local_t, :] = expert GEMM output from dest rank d for source token local_t.
#
# Visibility: the builder's p2p writes need no barrier of their own -- it runs
# immediately before the dispatch kernel on the same stream, and dispatch's
# end-of-kernel release/acquire barrier publishes them along with the dispatched
# activations.


# 128-bit chunks a single thread keeps in flight, and the launch width. DeepEP's
# NVLink copy path runs UNROLLED_WARP_COPY with an unroll of 8 on a 1024-thread CTA
# (128 B/thread, ~64 KB/CTA); its reducing path uses 4, because the accumulators
# compete for the same registers. Same split here.
A2AV_UNROLL_COPY = 8
A2AV_UNROLL_REDUCE = 4
# 512 threads, not 1024: unroll 8 needs ~100 registers/thread for the in-flight chunks
# and their address vectors, and a 1024-thread block caps out at 65536/1024 = 64
# registers/thread, which would spill. 512 x 8 chunks still puts the same ~64 KB of
# loads in flight per CTA (DeepEP's figure) with 128 registers/thread of headroom.
A2AV_THREADS = 512
# Tokens per compaction segment (power of two). Also the builder's per-CTA work unit,
# so keeping it small keeps the builder parallel at modest batch sizes; the push
# kernel subdivides a segment into TILES_PER_SEG tiles, so SEG does not bound its
# parallelism. Only the per-segment count load (one scalar per SEG tokens) scales with it.
A2AV_SEGMENT_TOKENS = 256


def _tile_shape(numel_per_token: int, unroll: int = A2AV_UNROLL_COPY):
    """Lane/thread geometry for the tiled A2AV data-movement kernels.

    Returns (npt_p2, tokens_per_tile, block_size, num_warps) where `block_size` is the
    Triton tile width in 128-bit chunks and the launch uses A2AV_THREADS threads, so
    each thread receives block_size / A2AV_THREADS independent chunks -- issued as one
    load batch followed by one store batch, which is what puts bytes in flight.
    """
    npt_p2 = triton.next_power_of_2(numel_per_token)
    tokens_per_tile = max(1, (A2AV_THREADS * unroll) // npt_p2)
    block_size = npt_p2 * tokens_per_tile
    num_warps = min(32, max(1, min(A2AV_THREADS, block_size) // 32))
    return npt_p2, tokens_per_tile, block_size, num_warps


def a2av_index_buffer_shapes(per_rank_max_tokens: int, world_size: int):
    """Shapes of the two symmetric index buffers the push combine needs.

    recv_list  : [world_size, nseg_max * SEG] int32 -- compact local token indices,
                 written by each source rank into every destination rank.
    recv_count : [world_size, nseg_max]       int32 -- entries per (source rank, segment).
    """
    seg = A2AV_SEGMENT_TOKENS
    nseg_max = max(1, (per_rank_max_tokens + seg - 1) // seg)
    return (world_size, nseg_max * seg), (world_size, nseg_max)


@triton.jit
def _a2av_build_index_kernel(
    routing_ptr,        # int64: LOCAL [local_tokens, TOPK] int64 expert ids
    dest_mask_ptr,      # int64: LOCAL [local_tokens] int64 destination bitmask (out)
    list_ptrs,          # int64: buffer_ptrs_dev of the recv_list symmetric buffer
    count_ptrs,         # int64: buffer_ptrs_dev of the recv_count symmetric buffer
    local_tokens,       # int: this rank's token count this step
    list_stride,        # int: int32 elements per source-rank row of recv_list
    count_stride,       # int: int32 elements per source-rank row of recv_count
    SEG: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Build the per-destination compact token lists and the local destination bitmask.

    One CTA handles whole SEG-token segments in a grid-stride loop.  The destination
    bitmask is folded one top-k column at a time (a [SEG] vector per column) rather
    than materialising the [SEG, TOPK] block, which keeps the register footprint at a
    few values per thread; the TOPK columns of a token are contiguous, so the column
    loads hit in cache.  For each destination rank the hitting tokens are compacted
    with an intra-segment `tl.cumsum` and the resulting index run (plus its length) is
    p2p-stored into that rank's recv_list / recv_count.

    Because the compaction is intra-segment the write offsets are deterministic --
    segment j always owns list slots [j*SEG, (j+1)*SEG) -- so no atomics are needed
    and stale entries from a previous step are never read (the reader bounds its
    inner loop by recv_count, which is rewritten every step).
    """
    pid = tl.program_id(0)

    routing_ptr = routing_ptr.to(tl.int64)
    dest_mask_ptr = dest_mask_ptr.to(tl.int64)
    list_ptrs = list_ptrs.to(tl.int64)
    count_ptrs = count_ptrs.to(tl.int64)

    rptr = routing_ptr.to(tl.pointer_type(tl.int64))
    mptr = dest_mask_ptr.to(tl.pointer_type(tl.int64))
    lptrs = list_ptrs.to(tl.pointer_type(tl.int64))
    cptrs = count_ptrs.to(tl.pointer_type(tl.int64))

    ti = tl.arange(0, SEG)

    for j in range(pid, tl.cdiv(local_tokens, SEG), tl.num_programs(0)):
        t = j * SEG + ti
        in_rng = t < local_tokens
        row = t.to(tl.int64) * TOPK

        dest_mask = tl.zeros([SEG], tl.uint64)
        for k in tl.static_range(TOPK):
            expert = tl.load(rptr + row + k, mask=in_rng, other=-1)
            dest = tl.where(expert >= 0, expert // EXPERTS_PER_RANK, 0).to(tl.uint64)
            dest_mask = dest_mask | tl.where(
                expert >= 0,
                tl.full([SEG], 1, tl.uint64) << dest,
                tl.zeros([SEG], tl.uint64),
            )
        tl.store(mptr + t, dest_mask.to(tl.int64), mask=in_rng)

        for d in tl.static_range(WORLD_SIZE):
            hit = in_rng & (((dest_mask >> d) & 1) == 1)
            flag = tl.where(hit, 1, 0)
            pos = tl.cumsum(flag, 0) - flag  # exclusive prefix sum within the segment
            count = tl.sum(flag, 0)

            list_base = (
                tl.load(lptrs + d).to(tl.pointer_type(tl.int32)) + RANK * list_stride + j * SEG
            )
            st_32_p2p(list_base + pos, t.to(tl.uint32), mask=hit)

            # One lane publishes the segment's length.
            count_base = (
                tl.load(cptrs + d).to(tl.pointer_type(tl.int32)) + RANK * count_stride + j
            )
            st_32_p2p(
                count_base + tl.zeros([SEG], tl.int32),
                (count + tl.zeros([SEG], tl.int32)).to(tl.uint32),
                mask=ti == 0,
            )


@triton.jit
def _multimem_a2av_push_combine_kernel(
    out_buf_ptr,           # int64: LOCAL [global_cap, H] bf16 expert output
    combine_recv_ptrs,     # int64: buffer_ptrs_dev of the combine_recv symmetric buffer
    signal_pad_ptrs,       # signal_pad_ptrs_dev of the combine_recv symmetric buffer
    recv_list_ptr,         # int64: LOCAL recv_list  [WORLD_SIZE, list_stride]  int32
    recv_count_ptr,        # int64: LOCAL recv_count [WORLD_SIZE, count_stride] int32
    tokens_per_rank_ptr,   # int64: LOCAL [WORLD_SIZE] int32 per-rank token counts
    per_rank_max_tokens,   # int: token stride of combine_recv
    list_stride,           # int: int32 elements per source-rank row of recv_list
    count_stride,          # int: int32 elements per source-rank row of recv_count
    SEG: tl.constexpr,
    TILES_PER_SEG: tl.constexpr,  # ceil(SEG / TPT): tile slots a segment is split into
    NPT: tl.constexpr,     # 128-bit chunks per token row
    NPT_P2: tl.constexpr,  # NPT rounded up to a power of two (lane stride per token)
    TPT: tl.constexpr,     # token rows a CTA moves per iteration
    BLOCK_SIZE: tl.constexpr,  # == NPT_P2 * TPT
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Push this rank's expert outputs into each source rank's combine_recv buffer.

    Walks the compact lists published by the builder, so every active lane maps to a
    real token -- no hole scanning and no routing reduction on the data path.  The
    tile is A2AV_UNROLL chunks per thread wide, so each iteration issues its whole
    load batch before its store batch and keeps ~64 KB in flight per CTA.

    Ends with a release barrier so every pushed row is visible before any rank runs
    the receive kernel.
    """
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    out_buf_ptr = out_buf_ptr.to(tl.int64)
    combine_recv_ptrs = combine_recv_ptrs.to(tl.int64)
    recv_list_ptr = recv_list_ptr.to(tl.int64)
    recv_count_ptr = recv_count_ptr.to(tl.int64)
    tokens_per_rank_ptr = tokens_per_rank_ptr.to(tl.int64)

    obuf = out_buf_ptr.to(tl.pointer_type(tl.uint64))
    lst = recv_list_ptr.to(tl.pointer_type(tl.int32))
    cnt = recv_count_ptr.to(tl.pointer_type(tl.int32))
    tok = tokens_per_rank_ptr.to(tl.pointer_type(tl.int32))
    rptrs = combine_recv_ptrs.to(tl.pointer_type(tl.int64))

    tid = tl.arange(0, BLOCK_SIZE)
    ch = tid % NPT_P2
    slot = tid // NPT_P2
    ch_ok = ch < NPT

    for step in tl.static_range(WORLD_SIZE):
        # Rotate the peer walk by RANK so all ranks do not target the same peer at the
        # same instant (DeepEP rotates its warp->peer map the same way).
        src = (step + RANK) % WORLD_SIZE
        # Source rank `src`'s token count and its base offset in the dense global layout.
        n_src = tl.load(tok + src)
        rank_off = 0
        for q in tl.static_range(WORLD_SIZE):
            if q < src:
                rank_off += tl.load(tok + q)
        recv_base = tl.load(rptrs + src).to(tl.pointer_type(tl.uint64))

        # Grid-stride over (segment, tile) pairs, NOT whole segments. Striding over
        # segments alone leaves the grid idle whenever a rank holds fewer than
        # num_pid * SEG tokens -- at one segment per rank a single CTA would move that
        # rank's entire payload while the other 147 wait on the barrier. Tiles beyond a
        # segment's `count` mask off, so the fixed TILES_PER_SEG stride costs nothing.
        for unit in range(pid, tl.cdiv(n_src, SEG) * TILES_PER_SEG, num_pid):
            j = unit // TILES_PER_SEG
            count = tl.load(cnt + src * count_stride + j)
            i = (unit % TILES_PER_SEG) * TPT + slot
            ok = (i < count) & ch_ok
            local_t = tl.load(lst + src * list_stride + j * SEG + i, mask=ok, other=0)
            # Source side of the copy: the dense global slot this token occupies.
            src_off = (rank_off + local_t).to(tl.int64) * NPT + ch
            (x, y, z, w) = ld_128_nc(obuf + src_off * 2, mask=ok)
            # Destination: slot [RANK, local_t, :] of the source rank's buffer.
            dst_off = (RANK * per_rank_max_tokens + local_t).to(tl.int64) * NPT + ch
            st_128_p2p(recv_base + dst_off * 2, x, y, z, w, mask=ok)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=True, hasSubsequentMemAccess=True,
    )


@triton.jit
def _multimem_a2av_recv_combine_kernel(
    output_ptr,           # int64: LOCAL [local_tokens, H] bf16 output
    combine_recv_ptr,     # int64: LOCAL combine_recv [WORLD_SIZE, per_rank_max_tokens, H] bf16
    dest_mask_ptr,        # int64: LOCAL [local_tokens] int64 destination bitmask
    local_tokens,         # int: this rank's token count this step
    per_rank_max_tokens,  # int: token stride of combine_recv
    NPT: tl.constexpr,
    NPT_P2: tl.constexpr,
    TPT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Sum the pushed contributions out of the LOCAL combine_recv buffer.

    Purely local traffic: the push kernel's release barrier already made every remote
    write visible, so this pass just reads the WORLD_SIZE candidate slots for each of
    this rank's tokens, accumulates the ones its destination bitmask marks, and writes
    the bf16 result.  Same tiling as the push kernel, and the destination bitmask comes
    from the builder, so there is no TOPK reduction here either.
    """
    pid = tl.program_id(0)

    output_ptr = output_ptr.to(tl.int64)
    combine_recv_ptr = combine_recv_ptr.to(tl.int64)
    dest_mask_ptr = dest_mask_ptr.to(tl.int64)

    obuf = output_ptr.to(tl.pointer_type(tl.uint64))
    rbuf = combine_recv_ptr.to(tl.pointer_type(tl.uint64))
    mptr = dest_mask_ptr.to(tl.pointer_type(tl.int64))

    tid = tl.arange(0, BLOCK_SIZE)
    ch = tid % NPT_P2
    slot = tid // NPT_P2
    ch_ok = ch < NPT

    for tile in range(pid, tl.cdiv(local_tokens, TPT), tl.num_programs(0)):
        t = tile * TPT + slot
        ok = (t < local_tokens) & ch_ok
        dest_mask = tl.load(mptr + t, mask=ok, other=0).to(tl.uint64)

        acc_x_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_x_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_y_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_y_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_z_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_z_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_w_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_w_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for d in tl.static_range(WORLD_SIZE):
            rd = ok & (((dest_mask >> d) & 1) == 1)
            off = (d * per_rank_max_tokens + t).to(tl.int64) * NPT + ch
            (x, y, z, w) = ld_128(rbuf + off * 2, mask=rd, multicast_op=False)
            x_hi, x_lo = _unpack_bf16x2(x, rd)
            y_hi, y_lo = _unpack_bf16x2(y, rd)
            z_hi, z_lo = _unpack_bf16x2(z, rd)
            w_hi, w_lo = _unpack_bf16x2(w, rd)
            acc_x_hi += x_hi; acc_x_lo += x_lo
            acc_y_hi += y_hi; acc_y_lo += y_lo
            acc_z_hi += z_hi; acc_z_lo += z_lo
            acc_w_hi += w_hi; acc_w_lo += w_lo

        out_off = t.to(tl.int64) * NPT + ch
        st_128(
            obuf + out_off * 2,
            _pack_bf16x2(acc_x_hi, acc_x_lo),
            _pack_bf16x2(acc_y_hi, acc_y_lo),
            _pack_bf16x2(acc_z_hi, acc_z_lo),
            _pack_bf16x2(acc_w_hi, acc_w_lo),
            mask=ok,
            multicast_op=False,
        )


def multimem_a2av_build_index(
    routing: torch.Tensor,
    dest_mask: torch.Tensor,
    recv_list: torch.Tensor,
    recv_count: torch.Tensor,
    recv_list_hdl: _SymmetricMemory,
    recv_count_hdl: _SymmetricMemory,
    num_experts: int,
    world_size: int,
    **kwargs,
) -> None:
    """Publish the compact per-destination token lists and the local destination bitmask.

    Runs immediately before the dispatch kernel on the same stream; dispatch's
    end-of-kernel release barrier is what makes the p2p writes visible to peers, so
    this kernel issues no barrier of its own.

    Args:
        routing: this rank's [local_tokens, TOPK] int64 expert ids.
        dest_mask: LOCAL [>= local_tokens] int64 output, one destination bitmask per token.
        recv_list / recv_count: LOCAL views of the symmetric index buffers; only their
            shapes are read here (the kernel writes into the PEERS' copies).
        recv_list_hdl / recv_count_hdl: symmetric handles for those buffers.
        num_experts: total expert count, used to derive experts_per_rank.
        world_size: number of EP ranks.
    """
    assert HAVE_TRITON, "Triton is required for multimem_a2av_build_index."
    assert routing.ndim == 2, "routing must be 2-D [local_tokens, TOPK]."
    assert routing.dtype == torch.int64, f"routing must be int64, got {routing.dtype}."
    assert dest_mask.dtype == torch.int64, f"dest_mask must be int64, got {dest_mask.dtype}."
    assert recv_list.dtype == torch.int32 and recv_count.dtype == torch.int32, (
        "recv_list / recv_count must be int32."
    )
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size."
    assert world_size <= 64, "destination ranks are encoded in a uint64 bitmask."
    assert hasattr(recv_list_hdl, "buffer_ptrs_dev"), (
        "recv_list_hdl has no buffer_ptrs_dev; this torch build does not expose per-rank "
        "symmetric pointers."
    )

    local_tokens, topk = routing.shape
    if local_tokens == 0:
        return
    assert dest_mask.numel() >= local_tokens, "dest_mask is too small for local_tokens."

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    seg = A2AV_SEGMENT_TOKENS
    num_blocks = min(max(1, (local_tokens + seg - 1) // seg), MAX_NUM_BLOCKS)

    _a2av_build_index_kernel[(num_blocks, 1, 1)](
        routing.data_ptr(),
        dest_mask.data_ptr(),
        recv_list_hdl.buffer_ptrs_dev,
        recv_count_hdl.buffer_ptrs_dev,
        local_tokens,
        recv_list.shape[1],
        recv_count.shape[1],
        SEG=seg,
        TOPK=topk,
        EXPERTS_PER_RANK=num_experts // world_size,
        RANK=recv_list_hdl.rank,
        WORLD_SIZE=world_size,
        num_warps=8,
    )


def multimem_a2av_push_combine(
    out_buf: torch.Tensor,
    combine_recv_hdl: _SymmetricMemory,
    recv_list: torch.Tensor,
    recv_count: torch.Tensor,
    tokens_per_rank: torch.Tensor,
    per_rank_max_tokens: int,
    **kwargs,
) -> None:
    """Push expert GEMM outputs into every source rank's combine_recv buffer.

    Consumes the compact lists written by `multimem_a2av_build_index`, so the kernel
    touches only real tokens and performs no routing work on the data path.  Ends with
    a release barrier; run `multimem_a2av_recv_combine` afterwards to reduce.

    Args:
        out_buf: this rank's expert output, [global_cap, H] bf16 (dense global layout).
        combine_recv_hdl: symmetric handle for the [world*cap, H] bf16 receive buffer.
        recv_list / recv_count: this rank's LOCAL views of the index buffers that the
            peers' builders wrote into.
        tokens_per_rank: [world_size] int32 per-rank token counts (metadata buffer).
        per_rank_max_tokens: static per-rank capacity (combine_recv token stride).
    """
    assert HAVE_TRITON, "Triton is required for multimem_a2av_push_combine."
    assert out_buf.ndim == 2, "out_buf must be 2-D [global_cap, H]."
    assert out_buf.dtype == torch.bfloat16, f"out_buf must be bf16, got {out_buf.dtype}."
    assert is_device_nvls_capable(out_buf.device), (
        "multimem_a2av_push_combine requires Hopper+ GPU with NVLink (SM >= 9)."
    )
    assert hasattr(combine_recv_hdl, "buffer_ptrs_dev"), (
        "combine_recv_hdl has no buffer_ptrs_dev; this torch build does not expose "
        "per-rank symmetric pointers."
    )

    world_size = combine_recv_hdl.world_size
    hidden_size = out_buf.shape[1]
    row_bytes = hidden_size * out_buf.element_size()
    assert row_bytes % 16 == 0, (
        f"Hidden row ({hidden_size} x {out_buf.element_size()}B = {row_bytes}B) must be "
        f"16-byte aligned for the 128-bit push path."
    )

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    numel_per_thread = 128 // (out_buf.element_size() * 8)
    npt = (hidden_size + numel_per_thread - 1) // numel_per_thread
    npt_p2, tpt, block_size, num_warps = _tile_shape(npt)

    _multimem_a2av_push_combine_kernel[(MAX_NUM_BLOCKS, 1, 1)](
        out_buf.data_ptr(),
        combine_recv_hdl.buffer_ptrs_dev,
        combine_recv_hdl.signal_pad_ptrs_dev,
        recv_list.data_ptr(),
        recv_count.data_ptr(),
        tokens_per_rank.data_ptr(),
        per_rank_max_tokens,
        recv_list.shape[1],
        recv_count.shape[1],
        SEG=A2AV_SEGMENT_TOKENS,
        TILES_PER_SEG=max(1, A2AV_SEGMENT_TOKENS // tpt),
        NPT=npt,
        NPT_P2=npt_p2,
        TPT=tpt,
        BLOCK_SIZE=block_size,
        RANK=combine_recv_hdl.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )


def multimem_a2av_recv_combine(
    output_tensor: torch.Tensor,
    combine_recv: torch.Tensor,
    dest_mask: torch.Tensor,
    per_rank_max_tokens: int,
    rank: int,
    world_size: int,
    **kwargs,
) -> torch.Tensor:
    """Reduce the pushed contributions from the LOCAL combine_recv into `output_tensor`.

    No barrier: the push kernel's end-of-kernel release barrier already ordered every
    remote write ahead of this kernel.  Reads only local memory.

    Args:
        output_tensor: LOCAL [local_tokens, H] bf16 output.
        combine_recv: LOCAL [world_size * per_rank_max_tokens, H] bf16 receive buffer.
        dest_mask: LOCAL [>= local_tokens] int64 destination bitmask from the builder.
        per_rank_max_tokens: combine_recv token stride.
    """
    assert HAVE_TRITON, "Triton is required for multimem_a2av_recv_combine."
    assert output_tensor.ndim == 2 and combine_recv.ndim == 2, "tensors must be 2-D."
    assert output_tensor.dtype == torch.bfloat16, (
        f"output_tensor must be bf16, got {output_tensor.dtype}."
    )
    assert combine_recv.dtype == torch.bfloat16, (
        f"combine_recv must be bf16, got {combine_recv.dtype}."
    )
    assert is_device_nvls_capable(output_tensor.device), (
        "multimem_a2av_recv_combine requires Hopper+ GPU with NVLink (SM >= 9)."
    )

    local_tokens, hidden_size = output_tensor.shape
    if local_tokens == 0:
        return output_tensor
    assert combine_recv.shape[1] == hidden_size, "hidden mismatch."

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    numel_per_thread = 128 // (output_tensor.element_size() * 8)
    npt = (hidden_size + numel_per_thread - 1) // numel_per_thread
    npt_p2, tpt, block_size, num_warps = _tile_shape(npt, A2AV_UNROLL_REDUCE)
    num_blocks = min(max(1, (local_tokens + tpt - 1) // tpt), MAX_NUM_BLOCKS)

    _multimem_a2av_recv_combine_kernel[(num_blocks, 1, 1)](
        output_tensor.data_ptr(),
        combine_recv.data_ptr(),
        dest_mask.data_ptr(),
        local_tokens,
        per_rank_max_tokens,
        NPT=npt,
        NPT_P2=npt_p2,
        TPT=tpt,
        BLOCK_SIZE=block_size,
        RANK=rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output_tensor


def multimem_a2av_combine(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    routing: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    num_experts: int,
    input_byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """All-to-all-v pull combine for a single 2-D bf16 tensor.

    For each of THIS rank's local tokens, read the token's expert output from only its
    destination ranks' copies of the symmetric output buffer (same dense global offset),
    sum in fp32, and write bf16 to output_tensor. Destination ranks are derived from
    `routing` (this rank's [local_tokens, topk] expert ids), deduplicated.

    output_tensor: local output [local_tokens, hidden] bf16 (regular tensor).
    input_tensor : the symmetric output buffer [global_tokens, hidden] bf16 (used for
                   shape/dtype checks; the actual reads use symm_mem_hdl.buffer_ptrs_dev).
    routing      : local [local_tokens, topk] int64 expert ids.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-to-all-v combine."
    assert output_tensor.ndim == 2 and input_tensor.ndim == 2, "tensors must be 2-D."
    assert is_device_nvls_capable(
        output_tensor.device
    ), "multimem_a2av_combine requires a Hopper+ GPU with NVLink (SM >= 9)."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."
    assert (
        output_tensor.dtype == torch.bfloat16 and input_tensor.dtype == torch.bfloat16
    ), f"a2av combine is bf16-only, got {output_tensor.dtype}/{input_tensor.dtype}."
    assert hasattr(symm_mem_hdl, "buffer_ptrs_dev"), (
        "symmetric-memory handle has no buffer_ptrs_dev; the installed torch build does not "
        "expose per-rank symmetric pointers required for all-to-all-v pull combine."
    )

    hidden_size = output_tensor.shape[1]
    assert input_tensor.shape[1] == hidden_size, "hidden mismatch."
    row_bytes = hidden_size * output_tensor.element_size()
    assert row_bytes % 16 == 0, (
        f"Hidden row ({hidden_size} x {output_tensor.element_size()}B = {row_bytes}B) must be "
        f"16-byte aligned for the all-to-all-v 128-bit path."
    )
    world_size = symm_mem_hdl.world_size
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size."
    assert world_size <= 64, (
        "pull combine encodes destination ranks in a uint64 bitmask; WORLD_SIZE must be <= 64."
    )
    experts_per_rank = num_experts // world_size
    topk = routing.shape[1]

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 148)
    MAX_BLOCK_SIZE = 1024
    WARP_SIZE = 32

    local_tokens = output_tensor.shape[0]
    numel_per_thread = 128 // (output_tensor.element_size() * 8)
    numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread
    block_size = min(triton.next_power_of_2(numel_per_token), MAX_BLOCK_SIZE)
    block_size = max(block_size, triton.next_power_of_2(topk))
    num_warps = max(1, block_size // WARP_SIZE)
    num_blocks = min(per_rank_max_tokens, MAX_NUM_BLOCKS)

    _multimem_a2av_pull_combine_kernel[(num_blocks, 1, 1)](
        output_tensor.data_ptr(),
        symm_mem_hdl.buffer_ptrs_dev,
        routing.data_ptr(),
        symm_mem_hdl.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        input_byte_offset=input_byte_offset,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        TOPK=topk,
        EXPERTS_PER_RANK=experts_per_rank,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output_tensor
