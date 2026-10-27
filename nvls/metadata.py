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
       two mode flags, derived from all ranks' counts with a per-step policy:
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

# Hybrid-dispatcher policy: int32 [6] = [dispatch_n0, dispatch_kappa_q8, combine_n0,
# combine_kappa_q8, per_rank, dest_prob_q16]. Rank s's dispatch (combine) flag is 1 (multicast,
# multicast-reduce) iff
#     valid_tokens - kappa * ep_max_tokens <= n0                                    (global)
#  or per_rank and u * n_s > q * valid_tokens + (1 - q) * S_s                       (per rank)
# All in integer arithmetic (kappa in Q8, q in Q16), so ranks agree wherever they must.
#  * Global term: the same on every rank. kappa = 0 is a plain threshold on the global token
#    count (latency-regime crossover); kappa = u also multicasts everyone under heavy imbalance.
#  * Per-rank term (bandwidth model, docs/hybrid_dispatcher_design.md): q = P(a given rank is
#    one of a token's destinations), u = (EP - 1) * q = a token's expected remote destinations,
#    n_s = this rank's count, S_s = n_s + the counts of all heavier ranks (ties: lower rank
#    first). Rank s multicasts while its unicast egress u * n_s exceeds the ingress every GPU
#    would see with s and all heavier ranks multicasting, q * N + (1 - q) * S_s: the heaviest
#    senders multicast, the rest unicast. Flags then differ across ranks, which is correct
#    because routing is always multicast and the expert always writes zero rows.
POLICY_LEN = 6
POLICY_KAPPA_SCALE = 256
POLICY_Q_SCALE = 65536
POLICY_FORCE_UNICAST = -1  # n0 with kappa = 0: never multicast (valid_tokens >= 0 > n0)
POLICY_FORCE_MULTICAST = 2**31 - 1  # n0: always multicast
HYBRID_STEP_METADATA_LEN = 5  # + [dispatch_multicast, combine_multicast_reduce]


def hybrid_policy_values(dispatch_n0, combine_n0, dispatch_kappa=0.0, combine_kappa=0.0,
                         per_rank=False, dest_prob=0.0):
    """Host-side policy (Python ints, see POLICY_LEN): global thresholds n0, kappa (float) and,
    with per_rank, the per-rank bandwidth rule with q = dest_prob in [0, 1]."""
    assert 0.0 <= dest_prob <= 1.0
    vals = [int(dispatch_n0), round(dispatch_kappa * POLICY_KAPPA_SCALE),
            int(combine_n0), round(combine_kappa * POLICY_KAPPA_SCALE),
            int(bool(per_rank)), round(dest_prob * POLICY_Q_SCALE)]
    for v in vals:
        assert -(2**31) <= v < 2**31, f"policy value {v} does not fit int32"
    return vals


def hybrid_policy_flags(policy_values, counts, rank):
    """Reference (host) evaluation of rank `rank`'s mode flags -> (dispatch_mc, combine_rs),
    given every rank's token count."""
    d_n0, d_k, c_n0, c_k, per_rank, q = policy_values
    total, n_max, n = sum(counts), max(counts), counts[rank]
    s, qs = POLICY_KAPPA_SCALE, POLICY_Q_SCALE
    bandwidth = False
    if per_rank:
        heavier = sum(c for r, c in enumerate(counts) if c > n or (c == n and r < rank))
        bandwidth = (len(counts) - 1) * q * n > q * total + (qs - q) * (n + heavier)
    return (int(total * s - d_k * n_max <= d_n0 * s or bandwidth),
            int(total * s - c_k * n_max <= c_n0 * s or bandwidth))


@triton.jit
def _fused_metadata_kernel(
    local_tokens,
    local_buf_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    step_metadata_ptr,
    policy_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HAS_POLICY: tl.constexpr = False,
    KAPPA_SCALE: tl.constexpr = POLICY_KAPPA_SCALE,
    Q_SCALE: tl.constexpr = POLICY_Q_SCALE,
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
        step_metadata_ptr: pointer to the 3-element int32 output tensor (5 with HAS_POLICY).
        policy_ptr: int32 [POLICY_LEN] hybrid policy (see the comment above POLICY_LEN); read
            only with HAS_POLICY.
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
        HAS_POLICY: also write the hybrid mode flags to step_metadata[3:5] (constexpr).
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

    # 4. Hybrid mode flags (see the policy comment at the top of this file).
    if HAS_POLICY:
        total64 = total.to(tl.int64)
        max_val = max_val.to(tl.int64)
        # Per-rank bandwidth rule: u * n_s > q * N + (1 - q) * S_s, scaled by Q_SCALE.
        q = tl.load(policy_ptr + 5).to(tl.int64)
        mine = tl.sum(tl.where(offsets == RANK, vals, tl.zeros_like(vals))).to(tl.int64)
        vals64 = vals.to(tl.int64)
        heavier = (vals64 > mine) | ((vals64 == mine) & (offsets < RANK))
        s_rank = mine + tl.sum(tl.where(heavier, vals64, tl.zeros_like(vals64)))
        bandwidth = (tl.load(policy_ptr + 4) != 0) & (
            (WORLD_SIZE - 1) * q * mine > q * total64 + (Q_SCALE - q) * s_rank
        )
        for i in tl.static_range(2):
            n0 = tl.load(policy_ptr + 2 * i).to(tl.int64)
            kappa_q8 = tl.load(policy_ptr + 2 * i + 1).to(tl.int64)
            global_rule = (total64 * KAPPA_SCALE - kappa_q8 * max_val) <= n0 * KAPPA_SCALE
            multicast = global_rule | bandwidth
            tl.store(step_metadata_ptr + 3 + i, multicast.to(tl.int32))


def fused_metadata_update(
    local_tokens: int,
    local_buf: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    step_metadata: torch.Tensor,
    policy: Optional[torch.Tensor] = None,
) -> None:
    """Fused NVLS allgather + reduce for MoE step metadata.

    Args:
        local_tokens: number of tokens on this rank this step.
        local_buf: the local symmetric memory buffer tensor ([WORLD_SIZE] int32).
            Used for reads after the barrier.
        symm_mem_hdl: symmetric memory handle for the metadata buffer.
            Provides the multicast pointer for writes and signal pads for barrier.
        step_metadata: [3] int32 CUDA tensor to write
            [valid_tokens, rank_token_offset, ep_max_tokens] into; [5] with `policy`, which
            adds this rank's [dispatch_multicast, combine_multicast_reduce] (0/1).
        policy: optional int32 [POLICY_LEN] CUDA tensor (hybrid_policy_values), identical on
            every rank. Read on device each call, so it can change between CUDA-graph replays.
    """
    assert HAVE_TRITON, "Triton is required for fused_metadata_update."
    has_policy = policy is not None
    if has_policy:
        assert policy.dtype == torch.int32 and policy.numel() == POLICY_LEN and policy.is_cuda
        assert step_metadata.numel() >= HYBRID_STEP_METADATA_LEN, (
            f"step_metadata needs {HYBRID_STEP_METADATA_LEN} elements with a policy")

    _fused_metadata_kernel[(1, 1, 1)](
        local_tokens,
        local_buf,
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        step_metadata,
        policy if has_policy else step_metadata,  # unread without HAS_POLICY
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        HAS_POLICY=has_policy,
        num_warps=min(max(1, (symm_mem_hdl.world_size + 31) // 32), 8),
    )
