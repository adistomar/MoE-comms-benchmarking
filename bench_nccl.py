# Copyright (c) 2026. Megatron NCCL AllGather (CUDA-graph path) dispatch/combine bencher.
"""
Isolated bencher for Megatron's NCCLAllGatherDispatcher, CUDA-graphable (CG) path.

The CG path (Megatron token_dispatcher_inference.py: _use_allgather_v=False) requires
every EP rank to contribute the SAME token count, then does a plain NCCL AllGather to
gather all tokens to all ranks and a plain ReduceScatter to sum expert outputs back.
Decode CUDA graphs guarantee equal counts in production; here the GLOBAL batch B is split
across ranks and may be uneven (counts differ by <=1), so we PAD each rank up to the
per-step max count to satisfy the equal-count requirement -- exactly the dispatcher's
"does padding if this is not the case" behavior. The pad target is discovered with a
one-time all-reduce(MAX) in setup_batch, OUTSIDE any CUDA graph, so the timed decode step
has no host sync.

Same dense algorithm as NVLSAllGatherVDispatcher; the ONLY difference is the transport
(NCCL ring/tree collectives vs NVLS NVLink multicast). To keep the comparison to transport
alone, dispatch gathers the same three tensors NVLS does (hidden bf16, routing int64,
probs fp32) and combine ReduceScatters in fp32 (matching NVLS's fp32 RSV precision).

We time the PAIRED dispatch->combine as one decode step (num_layers x), all captured in a
single CUDA graph. There is NO once-per-step metadata collective on the CG path (counts
are equal by construction), unlike NVLS's fused_metadata_update.
"""

import torch
import torch.distributed as dist

from common import Config


class NCCLBencher:
    name = "nccl"

    def __init__(self, cfg: Config, group):
        self.cfg = cfg
        self.group = group
        # NCCL collectives choose their own grid internally; there is no SM knob to sweep.
        self.num_sms = -1  # reported as n/a
        self.device = torch.device("cuda", cfg.local_rank)
        self._built = False

    def set_num_sms(self, num_sms: int):
        """No-op: NCCL AllGather/ReduceScatter pick their own grid; no num_sms knob."""

    # -- one-time allocation ---------------------------------------------------
    def build(self):
        cfg = self.cfg
        cap, gcap = cfg.per_rank_cap, cfg.global_cap
        H, K, dev = cfg.hidden, cfg.topk, self.device
        # Padded local inputs (source of the AllGather) and gathered outputs, sized to the
        # worst case; each step uses the [:pad_to] / [:ep*pad_to] contiguous prefixes.
        self.pad_h = torch.zeros(cap, H, dtype=torch.bfloat16, device=dev)
        self.pad_r = torch.zeros(cap, K, dtype=torch.int64, device=dev)
        self.pad_p = torch.zeros(cap, K, dtype=torch.float32, device=dev)
        self.g_h = torch.empty(gcap, H, dtype=torch.bfloat16, device=dev)
        self.g_r = torch.empty(gcap, K, dtype=torch.int64, device=dev)
        self.g_p = torch.empty(gcap, K, dtype=torch.float32, device=dev)
        # Combine input = expert output; ReduceScatter runs in fp32 (matches NVLS RSV, so
        # NCCL-vs-NVLS isolates transport). Pre-filled so combine timing has valid data.
        self.rs_in = torch.empty(gcap, H, dtype=torch.float32, device=dev)
        self.rs_in.normal_()
        self.out = torch.empty(cap, H, dtype=torch.float32, device=dev)
        self.n = 0
        self.pad_to = 0
        self._built = True

    def _max_count(self, n: int) -> int:
        """Max local token count across ranks = the equal-count pad target. One-time
        all-reduce(MAX) done in setup (outside CUDA graph), so no host sync on the hot path."""
        t = torch.tensor([n], device=self.device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.group)
        return int(t.item())

    # -- per-batch setup -------------------------------------------------------
    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        n = hidden.shape[0]
        self.n = n
        self.pad_to = self._max_count(n)  # equalize counts across ranks (CG requirement)
        # Fill the padded local buffers; rows [n:pad_to] stay zero (the padding).
        self.pad_h[:n] = hidden
        self.pad_h[n:self.pad_to] = 0
        self.pad_r[:n] = topk_idx.to(torch.int64)
        self.pad_r[n:self.pad_to] = 0
        self.pad_p[:n] = topk_weights
        self.pad_p[n:self.pad_to] = 0

    # -- timed ops -------------------------------------------------------------
    def dispatch(self):
        """AllGather: gather every rank's padded (hidden, routing, probs) to all ranks.
        Gathered layout is [rank0's pad_to rows, rank1's pad_to rows, ...]."""
        p = self.pad_to
        g = p * self.cfg.ep_size
        dist.all_gather_into_tensor(self.g_h[:g], self.pad_h[:p], group=self.group)
        dist.all_gather_into_tensor(self.g_r[:g], self.pad_r[:p], group=self.group)
        dist.all_gather_into_tensor(self.g_p[:g], self.pad_p[:p], group=self.group)

    def combine(self):
        """ReduceScatter (fp32): sum expert outputs across ranks, scatter to each owner.
        out[:pad_to] on rank r = sum over ranks of rs_in[r*pad_to:(r+1)*pad_to]; the first
        n rows are this rank's real tokens (rows [n:pad_to] are padding, discarded)."""
        p = self.pad_to
        g = p * self.cfg.ep_size
        dist.reduce_scatter_tensor(self.out[:p], self.rs_in[:g], group=self.group)

    def step(self):
        """One MoE layer: AllGather dispatch -> (identity expert) -> ReduceScatter combine."""
        self.dispatch()
        self.combine()

    def decode_step(self, num_layers: int):
        """One full decode step: num_layers x (dispatch -> combine). No once-per-step
        metadata collective on the CG path (equal counts guaranteed by padding)."""
        for _ in range(num_layers):
            self.step()

    def destroy(self):
        pass

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Round-trip identity check with RANDOM tensors -> list of (name, ok, detail).

        Padded AllGather -> masked identity expert (fp32) -> ReduceScatter must return
        (#distinct destination ranks for t) * x[t] for this rank's owned tokens,
        element-wise -- the SAME m*x that NVLS's masked-identity round-trip and DeepEP's
        rank-layout combine yield. Random per-element x catches column/stride displacement.
        Tested with all ranks populated (B=2*ep) and 0-token ranks (B=1)."""
        world, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts
        epr = E // world
        out = []
        for B in (world * 2, 1):
            counts = [B // world + (1 if r < B % world else 0) for r in range(world)]
            n = counts[rank]
            gen = torch.Generator(device=dev).manual_seed(990701 + B + rank * 149)
            x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                 if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
            idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
                   if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
            got = self.functional_roundtrip(x, idx).to(torch.float32)
            torch.cuda.synchronize()
            if n > 0:
                m = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                 device=dev, dtype=torch.float32).view(n, 1)
                ref = m * x.to(torch.float32)
                ok = bool(torch.allclose(got, ref, rtol=0.03, atol=0.05))
                detail = (f"n={n} max|got-m*x|={float((got - ref).abs().max()):.4f} "
                          f"m(#dest_ranks)={m.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
            else:
                ok, detail = True, "n=0 (idle rank still participates in the AllGather)"
            out.append((f"NCCL dispatch->combine   B={B:<4d}", ok, detail))
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Full functional padded-AllGather -> masked identity expert (fp32) -> ReduceScatter,
        returning [n,H] == (#distinct dest ranks)*x for this rank's owned tokens. Matches the
        NVLS masked-identity round-trip (same m*x), enabling the cross-impl equality check."""
        world, rank = self.cfg.ep_size, self.cfg.rank
        epr = self.cfg.num_experts // world
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx,
                         torch.zeros(n, self.cfg.topk, device=self.device, dtype=torch.float32))
        p, g = self.pad_to, self.pad_to * world
        dist.all_gather_into_tensor(self.g_h[:g], self.pad_h[:p], group=self.group)
        dist.all_gather_into_tensor(self.g_r[:g], self.pad_r[:p], group=self.group)
        # Masked identity: rank r contributes a gathered token iff that token routed >=1
        # expert local to rank r. Padded rows have hidden==0, so they contribute 0 (and are
        # truncated anyway), regardless of their (zero) routing.
        gr = self.g_r[:g]
        local = ((gr >= rank * epr) & (gr < (rank + 1) * epr)).any(dim=1)  # [g] bool
        self.rs_in[:g] = torch.where(local.view(g, 1), self.g_h[:g].to(torch.float32),
                                     torch.zeros((), device=self.device))
        dist.reduce_scatter_tensor(self.out[:p], self.rs_in[:g], group=self.group)
        return self.out[:n]
