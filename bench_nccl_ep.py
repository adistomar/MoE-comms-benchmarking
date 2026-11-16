# Copyright (c) 2026. NCCL EP (NVIDIA/nccl-extensions) dispatch/combine bencher.
"""
Isolated bencher for NCCL EP's MoE dispatch/combine (python bindings `nccl.ep`), in either of
its two algorithms:

  ncclep_ll : LOW_LATENCY, RANK_MAJOR layout -- the decode algorithm. Dispatch writes each token
              once into every rank that owns one of its experts, into a [num_ranks,
              max_dispatch_tokens_per_rank, hidden] receive buffer (row = src_rank, token) with
              the token's routing weights and LOCAL top-k ids (-1 = not on this rank);
              src_rank_counters gets the token count from each source rank. Combine sums each
              rank's row back on the token's source rank.
  ncclep_ht : HIGH_THROUGHPUT, FLAT layout -- the prefill / large-batch algorithm. Dispatch
              packs the received tokens into [max_recv_tokens_per_rank, hidden] rows (the first
              recv_total_counter rows valid) with weights and LOCAL top-k ids. Its per-rank
              dispatch capacity is a library build constant (MAX_SUPPORTED_TOKENS_PER_RANK =
              8192), so batches whose per-rank token count exceeds it are skipped.

Both layouts deliver one row per (token, destination rank) plus the token's top-k routing, the
same contract as DeepEP's do_expand=False, so vllm_fused_moe consumes them directly. Setup and
per-step calls follow FlashInfer's NCCL EP backend (flashinfer/moe_ep/backends/split/comm/
nccl_ep): a fresh NCCL communicator mirroring the torch group (nccl.ep.interop.torch), one
Group, one handle initialized OUTSIDE graph capture (ncclEpInitHandle), then per MoE layer:

  handle.update(topk_idx)   per-step routing (HT: also the metadata exchange); graph-safe
  handle.dispatch(...)      + handle.complete(), as FlashInfer issues it
  expert (optional)
  handle.combine(...)       (+ handle.complete() for HT, as FlashInfer issues it)

With --expert vllm, each layer runs vllm_fused_moe on the received rows (local expert ids,
start 0) and writes the output into a combine-input buffer, which combine reads. LL rank-major
rows past each source rank's count are padding whose ids NCCL EP does not write, so the expert
step first sets their ids to -1 (one masked fill) -- what an integration must do each step.
With the identity expert, combine sends the received rows back.
"""

import torch
import torch.distributed as dist

from common import Config

# Per-rank dispatch capacity of HT: nccl_ep/include/common.hpp MAX_SUPPORTED_TOKENS_PER_RANK.
HT_MAX_TOKENS_PER_RANK = 8192


class NcclEpBencher:
    def __init__(self, cfg: Config, group, algorithm: str):
        assert algorithm in ("ll", "ht"), algorithm
        self.cfg = cfg
        self.group = group
        self.algorithm = algorithm
        self.name = f"ncclep_{algorithm}"
        self.device = torch.device("cuda", cfg.local_rank)
        self.num_sms = -1  # NCCL EP's own SM budget (N/A in the CSV)
        # Largest per-rank token count this bencher can dispatch; run.py skips larger batches.
        self.max_tokens_per_rank = (min(cfg.per_rank_cap, HT_MAX_TOKENS_PER_RANK)
                                    if algorithm == "ht" else cfg.per_rank_cap)
        self._built = False
        self.expert = None  # None = identity expert; else a common.VllmExperts

    def build(self):
        import nccl.ep as ep
        from nccl.ep.interop.torch import get_nccl_comm_from_group
        self.ep = ep
        cfg = self.cfg
        cap, world, K, H = self.max_tokens_per_rank, cfg.ep_size, cfg.topk, cfg.hidden
        self.comm = get_nccl_comm_from_group(self.group)
        group_config = dict(
            algorithm=(ep.Algorithm.LOW_LATENCY if self.algorithm == "ll"
                       else ep.Algorithm.HIGH_THROUGHPUT),
            num_experts=cfg.num_experts,
            max_dispatch_tokens_per_rank=cap,
            max_token_bytes=H * 2,  # bf16 rows
        )
        if self.algorithm == "ht":
            # Static receive capacity (worst case: every rank sends every token here), which
            # CUDA graph capture of dispatch requires; FlashInfer sizes it the same way.
            group_config["max_recv_tokens_per_rank"] = cap * world
        self.ep_group = ep.Group.create(self.comm, ep.GroupConfig(**group_config))
        if self.algorithm == "ll":
            self.layout = ep.Layout.RANK_MAJOR
            self.handle = self.ep_group.init_handle(self.layout, num_topk=K)
            recv_rows = world * cap
        else:
            self.layout = ep.Layout.FLAT
            # This build rejects num_topk=-1 for HT ("HT mode requires num_topk > 0").
            self.handle = self.ep_group.init_handle(self.layout, num_topk=K)
            recv_rows = cap * world
        dev = self.device
        # Receive buffers (fixed addresses: the graph replays against them).
        self.recv_tokens = torch.empty(recv_rows, H, dtype=torch.bfloat16, device=dev)
        self.recv_weights = torch.empty(recv_rows, K, dtype=torch.float32, device=dev)
        self.recv_topk_idx = torch.full((recv_rows, K), -1,
                                        dtype=torch.int32 if self.algorithm == "ll"
                                        else torch.int64, device=dev)
        # LL: tokens received from each source rank; HT: total received rows.
        self.recv_counts = torch.zeros(world if self.algorithm == "ll" else 1,
                                       dtype=torch.int32, device=dev)
        # The expert's output, read by combine (Megatron's DeepEP combine_input analogue).
        self.combine_input = torch.empty(recv_rows, H, dtype=torch.bfloat16, device=dev)
        if self.algorithm == "ll":
            self._recv_shape = (world, cap)
            # Token slot index of every (source rank, slot) row, for the padding mask.
            self._slot_index = torch.arange(cap, device=dev).view(1, cap).expand(world, cap)
            # The expert sees every rank-major row (padding rows carry -1 ids).
            self._all_recv_rows = torch.full((1,), world * cap, dtype=torch.int32, device=dev)
        self.dispatch_config = ep.DispatchConfig(send_only=0, round_scales=0)
        self.combine_config = ep.CombineConfig(send_only=0)
        self._built = True

    def _ep_tensor(self, t, shape=None):
        """nccl.ep descriptor over torch tensor `t`, viewed as `shape`."""
        return self.ep.Tensor(t if shape is None else t.view(*shape))

    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        ep, cfg = self.ep, self.cfg
        n = hidden.shape[0]
        assert n <= self.max_tokens_per_rank, (n, self.max_tokens_per_rank)
        self.n = n
        self.hidden = hidden.contiguous()
        self.topk_idx = topk_idx.to(torch.int64).contiguous()
        self.topk_weights = topk_weights.to(torch.float32).contiguous()
        self.out = torch.empty(n, cfg.hidden, dtype=torch.bfloat16, device=self.device)
        world, K, H = cfg.ep_size, cfg.topk, cfg.hidden
        # Descriptors built once per batch; the per-layer path is only the C calls.
        self.topk_idx_t = ep.Tensor(self.topk_idx)
        if self.algorithm == "ll":
            cap = self.max_tokens_per_rank
            recv = dict(tokens=self._ep_tensor(self.recv_tokens, (world, cap, H)),
                        topk_weights=self._ep_tensor(self.recv_weights, (world, cap, K)),
                        topk_idx=self._ep_tensor(self.recv_topk_idx, (world, cap, K)))
            self.dispatch_layout_info = ep.LayoutInfo(
                src_rank_counters=ep.Tensor(self.recv_counts),
                recv_topk_idx_kind=ep.ExpertIdKind.LOCAL)
            self.identity_combine_in = ep.CombineInputs(
                tokens=self._ep_tensor(self.recv_tokens, (world, cap, H)))
            self.expert_combine_in = ep.CombineInputs(
                tokens=self._ep_tensor(self.combine_input, (world, cap, H)))
        else:
            recv = dict(tokens=ep.Tensor(self.recv_tokens),
                        topk_weights=ep.Tensor(self.recv_weights),
                        topk_idx=ep.Tensor(self.recv_topk_idx))
            # Flat layout count mode: dispatch publishes the received-row total itself.
            self.dispatch_layout_info = ep.LayoutInfo(
                recv_total_counter=ep.Tensor(self.recv_counts),
                recv_topk_idx_kind=ep.ExpertIdKind.LOCAL)
            self.identity_combine_in = ep.CombineInputs(tokens=ep.Tensor(self.recv_tokens))
            self.expert_combine_in = ep.CombineInputs(tokens=ep.Tensor(self.combine_input))
        self.dispatch_in = ep.DispatchInputs(tokens=ep.Tensor(self.hidden),
                                             topk_weights=ep.Tensor(self.topk_weights))
        self.dispatch_out = ep.DispatchOutputs(**recv)
        self.combine_out = ep.CombineOutputs(tokens=ep.Tensor(self.out))
        # The first update of a handle must run outside graph capture (ordering after
        # ncclEpInitHandle); also rebinds the routing to this batch's token count.
        self.update()
        torch.cuda.synchronize()

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def update(self):
        self.handle.update(self.topk_idx_t, layout_info=None, stream=self._stream())

    def dispatch(self):
        stream = self._stream()
        self.handle.dispatch(self.dispatch_in, self.dispatch_out,
                             layout_info=self.dispatch_layout_info,
                             config=self.dispatch_config, stream=stream)
        self.handle.complete(stream=stream)

    def combine(self, combine_in):
        stream = self._stream()
        self.handle.combine(combine_in, self.combine_out, config=self.combine_config,
                            stream=stream)
        if self.algorithm == "ht":
            self.handle.complete(stream=stream)

    def run_expert(self):
        cfg = self.cfg
        if self.algorithm == "ll":
            # Rank-major padding rows keep stale ids: mask them to -1 before the expert.
            world, cap = self._recv_shape
            idx = self.recv_topk_idx.view(world, cap, cfg.topk)
            padding = self._slot_index >= self.recv_counts.view(world, 1)
            idx.masked_fill_(padding.unsqueeze(-1), -1)
            valid = self._all_recv_rows
        else:
            valid = self.recv_counts
        self.expert(self.recv_tokens, self.recv_weights, self.recv_topk_idx, valid, 0,
                    self.combine_input, self.n * cfg.ep_size, write_unrouted_rows=False)

    def step(self) -> torch.Tensor:
        """One MoE layer: update -> dispatch -> expert -> combine. Returns the combined
        [local_tokens, hidden] bf16 output."""
        self.update()
        self.dispatch()
        if self.expert is not None:
            self.run_expert()
            self.combine(self.expert_combine_in)
        else:
            self.combine(self.identity_combine_in)
        return self.out

    def decode_step(self, num_layers: int) -> torch.Tensor:
        for _ in range(num_layers):
            out = self.step()
        return out

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Identity-expert round trip on random per-element tokens: combine sums one copy of
        token t per destination rank -> (#distinct dest ranks) * x[t] element-wise. Tested with
        all ranks populated (B=2*ep) and 0-token ranks (B=1). Also checks the received-row
        counts against the routing."""
        assert self.expert is None, "validate() checks the identity-expert layer"
        world, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts
        epr = E // world
        out = []
        for B in (world * 2, 1):
            n = B // world + (1 if rank < B % world else 0)
            gen = torch.Generator(device=dev).manual_seed(990907 + B + rank * 131)
            x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                 if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
            idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
                   if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
            combined = self.functional_roundtrip(x, idx)
            torch.cuda.synchronize()
            if n > 0:
                m = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                 device=dev, dtype=torch.float32).view(n, 1)
                got, ref = combined.to(torch.float32), m * x.to(torch.float32)
                ok = bool(torch.allclose(got, ref, rtol=0.03, atol=0.05))
                detail = (f"n={n} max|got-m*x|={float((got - ref).abs().max()):.4f} "
                          f"m(#dest_ranks)={m.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
            else:
                ok, detail = True, "n=0 (idle rank still participates in the A2A)"
            out.append((f"NCCL EP {self.algorithm.upper()} dispatch->combine B={B:<4d}", ok,
                        detail))
            out.append(self._check_recv_counts(x, idx, B))
        return out

    def _check_recv_counts(self, x, idx, B):
        """Received-row counters vs the routing: LL src_rank_counters[s] = #tokens of rank s
        with an expert here; HT recv_total_counter = their sum."""
        world, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        epr = self.cfg.num_local_experts
        n = x.shape[0]
        # sent[s, d] = #tokens of rank s with >= 1 expert on rank d.
        sent = torch.zeros(world, world, dtype=torch.int32, device=dev)
        if n > 0:
            dest = torch.zeros(n, world, dtype=torch.bool, device=dev)
            dest.scatter_(1, idx // epr, True)
            sent[rank] = dest.sum(dim=0).to(torch.int32)
        dist.all_reduce(sent, group=self.group)
        expected = sent[:, rank]
        got = self.recv_counts
        if self.algorithm == "ht":
            expected = expected.sum().view(1)
        ok = bool(torch.equal(got.to(torch.int64), expected.to(torch.int64)))
        return (f"NCCL EP {self.algorithm.upper()} received-row counts B={B:<4d}", ok,
                f"got={got.tolist()} expected={expected.tolist()}")

    def functional_roundtrip(self, hidden, topk_idx):
        """The timed identity-expert layer on the given tokens -> [n,H] == (#distinct dest ranks
        for t) * x[t] (run.py's cross-impl check)."""
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx,
                         torch.ones(n, self.cfg.topk, device=self.device, dtype=torch.float32))
        return self.step()[:n]

    def destroy(self):
        if getattr(self, "handle", None) is not None:
            self.handle.destroy()
            self.handle = None
        if getattr(self, "ep_group", None) is not None:
            self.ep_group.destroy()
            self.ep_group = None
        if getattr(self, "comm", None) is not None:
            self.comm.destroy()
            self.comm = None
