# Copyright (c) 2026. MoE dispatch/combine decode benchmark — shared utilities.
"""
Shared configuration, input generation, timing and expert-compute utilities for the MoE
dispatch/combine decode benchmark (DeepEP EPBuffer, Megatron NVLS / NCCL, A2A-V).

Model shape (experts, top-k, dispatched row width, expert FFN width) comes from the --model
preset in models.py (Nemotron-3 Super or Ultra). Parallelism: pure EP (one rank per GPU, TP=1),
EP = world size.

Batch-size axis is GLOBAL: B is the total number of decode tokens across all
EP ranks. Tokens are distributed as evenly as possible; for B<EP some ranks
hold 0 tokens. See README.md.
"""

import inspect
import math
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from models import DEFAULT_MODEL, ModelSpec, get_model


@dataclass
class Config:
    model: str = DEFAULT_MODEL     # models.MODELS key; sets the three fields below
    # From the model preset (not constructor arguments):
    num_experts: int = field(init=False)  # n_routed_experts
    topk: int = field(init=False)         # num_experts_per_tok
    hidden: int = field(init=False)       # dispatched row width = moe_latent_size
    # Filled at init from the process group / args (EP size == world size):
    ep_size: int = 0
    rank: int = 0
    local_rank: int = 0
    per_rank_cap: int = 0          # max tokens any single rank can hold (ceil(max_B/ep))
    seed: int = 1234

    def __post_init__(self):
        spec = self.spec
        self.num_experts = spec.num_experts
        self.topk = spec.topk
        self.hidden = spec.dispatch_hidden

    @property
    def spec(self) -> ModelSpec:
        return get_model(self.model)

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


# ---- Real expert compute (optional) ------------------------------------------
def expert_weights(cfg: Config, rank: int, device: torch.device):
    """Random BF16 weights (fc1 [E, ffn, H], fc2 [E, H, ffn]) of EP rank `rank`'s local experts,
    seeded per rank, so any rank can regenerate any other rank's experts (the reference check).
    Scaled so activations stay O(1)."""
    ffn, H, E = cfg.spec.moe_intermediate_size, cfg.hidden, cfg.num_local_experts
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 7 * rank)
    fc1 = (torch.randn(E, ffn, H, generator=gen, device=device) * H ** -0.5).to(torch.bfloat16)
    fc2 = (torch.randn(E, H, ffn, generator=gen, device=device) * ffn ** -0.5).to(torch.bfloat16)
    return fc1, fc2


class VllmExperts:
    """Megatron's inference expert GEMMs (vllm_fused_moe, BF16, squared ReLU), run between
    dispatch and combine with the same arguments Megatron's MoE layer passes for each
    dispatcher. Weights are random: only the work matters, not the values.

    Needs the Megatron-LM repo on sys.path (run.py --megatron-path).
    """

    def __init__(self, cfg: Config, max_rows: int, device: torch.device):
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import (
            VllmFusedMoeBuffers, vllm_fused_moe,
        )
        self.cfg = cfg
        self.fc1, self.fc2 = expert_weights(cfg, cfg.rank, device)
        self.num_local_experts = cfg.num_local_experts
        self._fn = vllm_fused_moe
        # Megatron before write_unrouted_rows always wrote every row.
        self._has_skip = "write_unrouted_rows" in inspect.signature(vllm_fused_moe).parameters
        self._act = ActivationType.SQUARED_RELU
        # Worst-case-sized intermediates, allocated once as Megatron's context does (max rows
        # entering the MoE = per-rank cap * EP), so nothing allocates inside graph capture.
        VllmFusedMoeBuffers.allocate_buffers(
            max_tokens=max_rows, topk=cfg.topk, fc1_output_size=cfg.spec.moe_intermediate_size,
            hidden_size=cfg.hidden, num_local_experts=self.num_local_experts, device=device,
        )

    def __call__(self, hidden, probs, routing, valid_tokens, local_expert_start, out,
                 num_tokens_hint, write_unrouted_rows=True):
        return self._fn(
            hidden, probs, self.fc1, self.fc2,
            activation_type=self._act,
            num_local_experts=self.num_local_experts,
            local_expert_start=local_expert_start,
            valid_tokens=valid_tokens,
            routing_map=routing,
            out=out,
            num_tokens_hint=num_tokens_hint,
            **({"write_unrouted_rows": write_unrouted_rows} if self._has_skip else {}),
        )

    def reference(self, hidden, topk_idx, topk_weights) -> torch.Tensor:
        """fp32 MoE output of this rank's tokens over all their experts, computed with plain
        torch from every owner rank's regenerated weights: sum_k p[t,k] * fc2 relu(fc1 x[t])^2.
        What dispatch -> experts -> combine must return, up to bf16 rounding (validation)."""
        cfg, E = self.cfg, self.num_local_experts
        x = hidden.to(torch.float32)
        out = torch.zeros_like(x)
        for r in range(cfg.ep_size):
            on_r = (topk_idx >= 0) & (topk_idx // E == r)
            if not bool(on_r.any()):
                continue
            fc1, fc2 = ((self.fc1, self.fc2) if r == cfg.rank
                        else expert_weights(cfg, r, hidden.device))
            tok, k = on_r.nonzero(as_tuple=True)
            local = topk_idx[tok, k] - r * E
            for e in local.unique().tolist():
                sel = local == e
                t = tok[sel]
                h = torch.relu(x[t] @ fc1[e].to(torch.float32).T).square()
                y = (h @ fc2[e].to(torch.float32).T) * topk_weights[t, k[sel]].unsqueeze(1)
                out.index_add_(0, t, y)
        return out
