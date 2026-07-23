# Copyright (c) 2026. MoE dispatch/combine decode benchmark — shared utilities.
"""
Shared configuration, input generation, and timing utilities for the
DeepEP-v2 (A2A) vs Megatron-NVLS (AGV/RSV) MoE dispatch/combine decode benchmark.

Model / parallelism (Nemotron-3 Super, single GB200 node, 4x B200):
  num_experts=512, top-k=22, hidden=1024, EP=4 (one rank per GPU, TP=1).

Batch-size axis is GLOBAL: B is the total number of decode tokens across all
EP ranks. Tokens are distributed as evenly as possible; for B<EP some ranks
hold 0 tokens. See docs/moe_dispatcher_deep_dive*.md and bench/README.md.
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

# ---- Model / parallelism constants -------------------------------------------
NUM_EXPERTS = 512
TOPK = 22
HIDDEN = 1024
# EP size == world size (pure EP=4, TP=1). Resolved at runtime from the PG.


@dataclass
class Config:
    num_experts: int = NUM_EXPERTS
    topk: int = TOPK
    hidden: int = HIDDEN
    # Filled at init from the process group / args:
    ep_size: int = 0
    rank: int = 0
    local_rank: int = 0
    per_rank_cap: int = 0          # max tokens any single rank can hold (ceil(max_B/ep))
    seed: int = 1234

    @property
    def num_local_experts(self) -> int:
        assert self.num_experts % self.ep_size == 0
        return self.num_experts // self.ep_size

    @property
    def global_cap(self) -> int:
        return self.per_rank_cap * self.ep_size


# ---- Distributed init --------------------------------------------------------
def init_distributed() -> "tuple[dist.ProcessGroup, int, int, int]":
    """Initialize the default NCCL process group from torchrun env vars.

    Returns (group, rank, world_size, local_rank).
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Pin the PG to this rank's local GPU. Without device_id, NCCL guesses the device
        # from the GLOBAL rank -- wrong on multi-node (e.g. rank 4 -> "device 4" on a 4-GPU
        # node) and it warns this "can cause a hang". The explicit device_id avoids that.
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                                device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    return group, rank, world_size, local_rank


# ---- Global -> per-rank token distribution -----------------------------------
def local_token_count(global_B: int, rank: int, world: int) -> int:
    """Balanced distribution of B global tokens across `world` ranks.

    rank r gets B//world + (1 if r < B%world else 0). For B<world the first B
    ranks get 1 token and the rest get 0.
    """
    return global_B // world + (1 if rank < (global_B % world) else 0)


def all_rank_counts(global_B: int, world: int) -> "list[int]":
    return [local_token_count(global_B, r, world) for r in range(world)]


# ---- Input generation (identical for both dispatchers) -----------------------
def make_inputs(cfg: Config, global_B: int, device: torch.device):
    """Generate this rank's local decode-token inputs for a global batch B.

    Uniform routing: each token selects `topk` DISTINCT experts uniformly from
    `num_experts` (via top-k over random scores). Weights are softmax over the
    selected scores. Deterministic in (global_B, rank, seed) so the DeepEP and
    NVLS runs see byte-identical routing (=> identical data-movement volume).

    Returns (hidden[n,H] bf16, topk_idx[n,K] int64, topk_weights[n,K] fp32),
    where n = this rank's local token count (may be 0).
    """
    n = local_token_count(global_B, cfg.rank, cfg.ep_size)
    K, E, H = cfg.topk, cfg.num_experts, cfg.hidden
    # Per-(B, rank) seed so both impls reproduce the same routing.
    seed = cfg.seed + global_B * 131 + cfg.rank
    gen = torch.Generator(device=device).manual_seed(seed)

    hidden = torch.randn((n, H), device=device, dtype=torch.bfloat16, generator=gen)
    if n == 0:
        topk_idx = torch.empty((0, K), device=device, dtype=torch.int64)
        topk_weights = torch.empty((0, K), device=device, dtype=torch.float32)
        return hidden, topk_idx, topk_weights

    scores = torch.rand((n, E), device=device, generator=gen)
    sel_w, topk_idx = torch.topk(scores, K, dim=-1)          # distinct uniform experts
    topk_weights = torch.softmax(sel_w, dim=-1).to(torch.float32)
    topk_idx = topk_idx.to(torch.int64)
    return hidden, topk_idx, topk_weights


# ---- Byte accounting ---------------------------------------------------------
def size_mb(shape, dtype) -> int:
    nbytes = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
    return max(1, (nbytes + (1 << 20) - 1) >> 20)


# ---- Timing ------------------------------------------------------------------
def _make_events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def time_region(fn, group, warmup: int, iters: int, use_graph: bool):
    """Time a callable `fn` (one dispatch or one combine) with CUDA events.

    Two modes:
      use_graph=True : capture `fn` into a CUDA graph and time N replays. This
        removes per-iteration host launch overhead — the realistic device-
        resident decode path (no CPU sync).
      use_graph=False: eager — time N direct calls (includes host launch).

    A cross-rank barrier precedes the timed window. Returns (local_us, max_us),
    where max_us is the max per-iter latency across ranks (the critical path).
    """
    # Warmup (also triggers JIT/autotune for DeepEP kernels and Triton kernels).
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)

    start, end = _make_events()

    if use_graph:
        # A capture-warmup on a side stream is recommended before graph capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group)

        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
    else:
        start.record()
        for _ in range(iters):
            fn()
        end.record()

    torch.cuda.synchronize()
    local_us = (start.elapsed_time(end) * 1e3) / iters  # ms -> us, per iter

    t = torch.tensor([local_us], device=torch.cuda.current_device(), dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return local_us, float(t.item())
