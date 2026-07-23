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
    routing_block: int = 0         # 0=uniform routing; L>0 = each token routes ONLY to experts in
                                   # its source rank's aligned L-rank block (locality knob for vmcast)
    routing_unaligned: bool = False  # place the locality window UNALIGNED (source-anchored, random offset)
                                     # instead of snapping to an aligned power-of-2 block. Models the real
                                     # decoupling of #dest-ranks from vmcast's aligned multicast-group size.
    routing_mix: str = ""          # "" = off; else comma weights over nested sizes [2,4,..,ep] (ascending):
                                   # each token independently draws a locality size s ~ these weights and
                                   # routes within its source rank's aligned s-block. A realistic MIXED
                                   # distribution (most tokens local + a tail spanning wide) vs the hard
                                   # single-size routing_block. Overrides routing_block when set.

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
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
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


def nested_sizes(P: int) -> "list[int]":
    """Aligned power-of-2 group sizes [2, 4, ..., P] (matches bench_vmcast's nested groups)."""
    s, out = 2, []
    while s <= P:
        out.append(s)
        s *= 2
    return out


def parse_routing_mix(mix_str: str, P: int) -> "tuple[list[int], list[float]]":
    """Map comma weights (ascending by size) onto nested sizes [2,4,..,P] -> normalized probs.
    Fewer weights than sizes => trailing sizes get 0; extra weights are ignored. Zero total => uniform."""
    sizes = nested_sizes(P)
    w = [float(x) for x in mix_str.split(",") if x.strip() != ""]
    w = (w + [0.0] * len(sizes))[:len(sizes)]
    tot = sum(w)
    probs = [x / tot for x in w] if tot > 0 else [1.0 / len(sizes)] * len(sizes)
    return sizes, probs


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
    epr, P = E // cfg.ep_size, cfg.ep_size
    use_block = 0 < cfg.routing_block < P
    if cfg.routing_mix or use_block:
        # Per-token locality window of s_t ranks (drawn from routing_mix weights, or a fixed
        # routing_block). Placement:
        #   ALIGNED (default): snap to the source's aligned s-block (r//s*s). Best case for vmcast --
        #     the aligned multicast group == the window, so group size == locality window.
        #   UNALIGNED (routing_unaligned): a window of s ranks COVERING the source at a random offset,
        #     NOT snapped. Realistic: the smallest ALIGNED nested group spanning an unaligned window of
        #     s ranks is typically 2*s, so vmcast's multicast group size DECOUPLES from (and exceeds) the
        #     #distinct dest ranks -- e.g. a token to 2 ranks 8 apart needs a size-16 group. With topk<s
        #     this is the "few ranks, wide aligned span" regime that actually stresses vmcast.
        if cfg.routing_mix:
            sizes, probs = parse_routing_mix(cfg.routing_mix, P)
            cum = torch.tensor(probs, device=device).cumsum(0)
            u = torch.rand((n,), device=device, generator=gen)
            sidx = torch.bucketize(u, cum).clamp(max=len(sizes) - 1)
            s_t = torch.tensor(sizes, device=device)[sidx]                     # [n] per-token window size
        else:
            s_t = torch.full((n,), cfg.routing_block, device=device, dtype=torch.long)
        if cfg.routing_unaligned:
            off = (torch.rand((n,), device=device, generator=gen) * s_t).to(torch.long)   # 0..s_t-1
            w0 = torch.minimum(torch.clamp(cfg.rank - off, min=0), P - s_t)                # window start
        else:
            w0 = (cfg.rank // s_t) * s_t                                        # aligned block start
        e = torch.arange(E, device=device).view(1, E)
        blockmask = (e >= (w0 * epr).view(n, 1)) & (e < ((w0 + s_t) * epr).view(n, 1))     # [n, E]
        scores = scores.masked_fill(~blockmask, float("-inf"))
    sel_w, topk_idx = torch.topk(scores, K, dim=-1)          # distinct experts (within window if local)
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
