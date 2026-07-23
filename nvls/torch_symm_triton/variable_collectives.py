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
from .multimem_asm import ld_64, ld_128, ld_128_p2p, st_64, st_128, st_128_p2p
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
