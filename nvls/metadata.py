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
    5. Optionally (hybrid dispatcher, torch_symm_triton/hybrid_v.py) also writes this rank's
       two mode flags, derived from all ranks' counts and the mode parameters:
       step_metadata[3] = multicast dispatch, step_metadata[4] = multicast-reduce combine.
"""

from typing import Optional
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

# Hybrid dispatcher mode parameters: int32 [MODE_PARAMS_LEN] =
#   [dispatch_threshold, dispatch_ep_max_weight_q8, combine_threshold, combine_ep_max_weight_q8,
#    local_rule, dest_prob_q16]
# They parametrize the rule that sets this rank's two mode flags each step. The dispatch
# (combine) flag is 1 (multicast, multicast-reduce) iff
#     valid_tokens - ep_max_weight * ep_max_tokens <= threshold                      (global)
#  or local_rule and u * n_s > q * valid_tokens + (1 - q) * S_s                      (local)
# All in integer arithmetic (ep_max_weight in Q8, q in Q16), so ranks agree wherever they must.
#  * Global term, the same on every rank. threshold is in global tokens. ep_max_weight = 0 makes
#    it a plain threshold on the global token count (the latency-regime crossover);
#    ep_max_weight = u also multicasts every rank once the heaviest rank's unicast fan-out
#    u * ep_max_tokens dominates (heavy imbalance).
#  * Local term (bandwidth model, docs/hybrid_dispatcher_design.md): q = dest_prob = P(a given
#    rank is one of a token's destinations), u = (EP - 1) * q = a token's expected remote
#    destinations, n_s = this rank's count, S_s = n_s + the counts of all heavier ranks (ties:
#    lower rank first). Rank s multicasts while its unicast egress u * n_s exceeds the ingress
#    every GPU would see with s and all heavier ranks multicasting, q * N + (1 - q) * S_s: the
#    heaviest senders multicast, the rest unicast. Flags then differ across ranks, which is
#    correct because routing is always multicast and the expert always writes zero rows.
MODE_PARAMS_LEN = 6
EP_MAX_WEIGHT_SCALE = 256  # Q8 fixed point
DEST_PROB_SCALE = 65536  # Q16 fixed point
ALWAYS_UNICAST = -1  # threshold with ep_max_weight = 0: never multicast (valid_tokens >= 0)
ALWAYS_MULTICAST = 2**31 - 1  # threshold: always multicast
HYBRID_STEP_METADATA_LEN = 5  # + [dispatch_multicast, combine_multicast_reduce]


def hybrid_mode_params(dispatch_threshold, combine_threshold, dispatch_ep_max_weight=0.0,
                       combine_ep_max_weight=0.0, local_rule=False, dest_prob=0.0):
    """Host-side mode parameters (Python ints, layout above MODE_PARAMS_LEN): per-collective
    thresholds (global tokens) and ep_max weights (floats), and, with local_rule, the local (per-rank)
    bandwidth rule with q = dest_prob in [0, 1]."""
    assert 0.0 <= dest_prob <= 1.0
    vals = [int(dispatch_threshold), round(dispatch_ep_max_weight * EP_MAX_WEIGHT_SCALE),
            int(combine_threshold), round(combine_ep_max_weight * EP_MAX_WEIGHT_SCALE),
            int(bool(local_rule)), round(dest_prob * DEST_PROB_SCALE)]
    for v in vals:
        assert -(2**31) <= v < 2**31, f"mode parameter {v} does not fit int32"
    return vals


def hybrid_mode_flags(mode_params, counts, rank):
    """Reference (host) evaluation of rank `rank`'s mode flags -> (dispatch_multicast,
    combine_reduce), given every rank's token count."""
    (d_threshold, d_weight, c_threshold, c_weight, local_rule, dest_prob) = mode_params
    total, ep_max, my_tokens = sum(counts), max(counts), counts[rank]
    ws, ps = EP_MAX_WEIGHT_SCALE, DEST_PROB_SCALE
    local_multicast = False
    if local_rule:
        heavier = sum(c for r, c in enumerate(counts)
                      if c > my_tokens or (c == my_tokens and r < rank))
        local_multicast = ((len(counts) - 1) * dest_prob * my_tokens
                              > dest_prob * total + (ps - dest_prob) * (my_tokens + heavier))
    return (int(total * ws - d_weight * ep_max <= d_threshold * ws or local_multicast),
            int(total * ws - c_weight * ep_max <= c_threshold * ws or local_multicast))


@triton.jit
def _fused_metadata_kernel(
    local_tokens,
    local_buf_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    step_metadata_ptr,
    mode_params_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    COMPUTE_MODES: tl.constexpr = False,
    EP_MAX_WEIGHT_SCALE: tl.constexpr = EP_MAX_WEIGHT_SCALE,
    DEST_PROB_SCALE: tl.constexpr = DEST_PROB_SCALE,
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
        step_metadata_ptr: pointer to the 3-element int32 output tensor (5 with COMPUTE_MODES).
        mode_params_ptr: int32 [MODE_PARAMS_LEN] hybrid mode parameters (see the comment above
            MODE_PARAMS_LEN); read only with COMPUTE_MODES.
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
        COMPUTE_MODES: also write the hybrid mode flags to step_metadata[3:5] (constexpr).
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

    # 4. Hybrid mode flags (see the comment above MODE_PARAMS_LEN).
    if COMPUTE_MODES:
        total64 = total.to(tl.int64)
        ep_max = max_val.to(tl.int64)
        # Local (per-rank) bandwidth rule: u * n_s > q * N + (1 - q) * S_s, scaled by
        # DEST_PROB_SCALE.
        local_rule = tl.load(mode_params_ptr + 4) != 0
        dest_prob = tl.load(mode_params_ptr + 5).to(tl.int64)
        my_tokens = tl.sum(tl.where(offsets == RANK, vals, tl.zeros_like(vals))).to(tl.int64)
        vals64 = vals.to(tl.int64)
        is_heavier = (vals64 > my_tokens) | ((vals64 == my_tokens) & (offsets < RANK))
        my_and_heavier_tokens = my_tokens + tl.sum(tl.where(is_heavier, vals64,
                                                            tl.zeros_like(vals64)))
        local_multicast = local_rule & (
            (WORLD_SIZE - 1) * dest_prob * my_tokens
            > dest_prob * total64 + (DEST_PROB_SCALE - dest_prob) * my_and_heavier_tokens
        )
        for i in tl.static_range(2):  # 0: dispatch, 1: combine
            threshold = tl.load(mode_params_ptr + 2 * i).to(tl.int64)
            ep_max_weight_q8 = tl.load(mode_params_ptr + 2 * i + 1).to(tl.int64)
            global_multicast = (total64 * EP_MAX_WEIGHT_SCALE - ep_max_weight_q8 * ep_max
                                <= threshold * EP_MAX_WEIGHT_SCALE)
            multicast = global_multicast | local_multicast
            tl.store(step_metadata_ptr + 3 + i, multicast.to(tl.int32))


def fused_metadata_update(
    local_tokens: int,
    local_buf: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    step_metadata: torch.Tensor,
    mode_params: Optional[torch.Tensor] = None,
) -> None:
    """Fused NVLS allgather + reduce for MoE step metadata.

    Args:
        local_tokens: number of tokens on this rank this step.
        local_buf: the local symmetric memory buffer tensor ([WORLD_SIZE] int32).
            Used for reads after the barrier.
        symm_mem_hdl: symmetric memory handle for the metadata buffer.
            Provides the multicast pointer for writes and signal pads for barrier.
        step_metadata: [3] int32 CUDA tensor to write
            [valid_tokens, rank_token_offset, ep_max_tokens] into; [5] with `mode_params`,
            which adds this rank's [dispatch_multicast, combine_multicast_reduce] (0/1).
        mode_params: optional int32 [MODE_PARAMS_LEN] CUDA tensor (hybrid_mode_params),
            identical on every rank. Read on device each call, so it can change between
            CUDA-graph replays.
    """
    assert HAVE_TRITON, "Triton is required for fused_metadata_update."
    compute_modes = mode_params is not None
    if compute_modes:
        assert (mode_params.dtype == torch.int32 and mode_params.numel() == MODE_PARAMS_LEN
                and mode_params.is_cuda)
        assert step_metadata.numel() >= HYBRID_STEP_METADATA_LEN, (
            f"step_metadata needs {HYBRID_STEP_METADATA_LEN} elements with mode_params")

    _fused_metadata_kernel[(1, 1, 1)](
        local_tokens,
        local_buf,
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        step_metadata,
        mode_params if compute_modes else step_metadata,  # unread without COMPUTE_MODES
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        COMPUTE_MODES=compute_modes,
        num_warps=min(max(1, (symm_mem_hdl.world_size + 31) // 32), 8),
    )
