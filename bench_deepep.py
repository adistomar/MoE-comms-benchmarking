# Copyright (c) 2026. DeepEP (EPBuffer, DeepEP >= 2.5) dispatch/combine bencher.
"""
Isolated DeepEP bencher. Calls EPBuffer.dispatch / .combine directly, configured and called
like Megatron's DeepEPDispatcher (token_dispatcher_inference.py):

  buffer   : EPBuffer(num_max_tokens_per_rank=per_rank_cap, hidden, num_topk,
             allow_hybrid_mode=False) with EP_DISABLE_GIN=1 -- NVLink only -- and EPBuffer's
             defaults otherwise: bf16, allow_multiple_reduction=True (rank-layout combine: one
             reduced copy per destination rank, comparable to NVLS RSV),
             prefer_overlap_with_compute=True, deterministic=False.
  dispatch : do_expand=False (one copy of each token per destination rank, no expert-grouping
             permute -- vllm_fused_moe groups the rows itself), do_cpu_sync=False (received
             count stays on the GPU, graph-safe), num_sms shared by combine.
  combine  : combine(expert_output, handle) -- unweighted; vllm_fused_moe already applied the
             routing probs.

With --expert vllm, each layer runs vllm_fused_moe on the received rows (local expert ids,
valid rows = the received count) and writes the output straight into the combine input,
as Megatron does. With the identity expert, combine sends the received rows back.

We time the PAIRED dispatch->combine as one layer: combine consumes the handle produced by
its immediately-preceding dispatch, so the two belong together.
"""

import os

import torch
import torch.distributed as dist

import deep_ep

from common import Config


class DeepEPBencher:
    name = "deepep"

    def __init__(self, cfg: Config, group, num_sms: int):
        self.cfg = cfg
        self.group = group
        self.num_sms = num_sms  # 0 = DeepEP's theoretical count, resolved in build()
        self.device = torch.device("cuda", cfg.local_rank)
        self._built = False
        self.expert = None  # None = identity expert; else a common.VllmExperts

    def build(self):
        cfg = self.cfg
        if not hasattr(deep_ep, "EPBuffer"):
            raise RuntimeError(f"deep_ep {getattr(deep_ep, '__version__', '?')} has no EPBuffer; "
                               f"the DeepEP bencher needs DeepEP >= 2.5 (see README).")
        # NVLink only: skip DeepEP's RDMA (NCCL GIN) setup unless the user chose otherwise.
        os.environ.setdefault("EP_DISABLE_GIN", "1")
        self.buf = deep_ep.EPBuffer(
            self.group,
            num_max_tokens_per_rank=cfg.per_rank_cap,
            hidden=cfg.hidden,
            num_topk=cfg.topk,
            allow_hybrid_mode=False,
            explicitly_destroy=True,
        )
        if self.buf.num_rdma_ranks != 1:
            raise RuntimeError(f"the EP group spans {self.buf.num_rdma_ranks} NVLink domains; "
                               f"this benchmark needs one")
        self.topk_idx_t = deep_ep.topk_idx_t
        # Megatron's combine input: vllm_fused_moe writes the expert output here.
        self.combine_input = torch.empty(cfg.ep_size * cfg.per_rank_cap, cfg.hidden,
                                         dtype=torch.bfloat16, device=self.device)
        # Received-token count of the current layer (Megatron's _valid_tokens_tensor).
        self.recv_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._built = True
        self.set_num_sms(self.num_sms)

    def theoretical_num_sms(self) -> int:
        """DeepEP's bandwidth-model SM count, agreed across ranks with MAX (what Megatron
        uses when inference_deepep_num_sms is unset)."""
        n = self.buf.get_theoretical_num_sms(self.cfg.num_experts, self.cfg.topk)
        t = torch.tensor([n], dtype=torch.int32, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.group)
        return int(t.item())

    def set_num_sms(self, num_sms: int):
        """Change the SM count used by dispatch/combine (buffer is SM-count-agnostic,
        so no rebuild needed; each value JIT-compiles its own kernels on first use).
        0 means DeepEP's theoretical count (Megatron's default)."""
        if num_sms == 0 and self._built:
            num_sms = self.theoretical_num_sms()
        self.num_sms = num_sms

    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        self.hidden = hidden.contiguous()
        self.topk_idx = topk_idx.to(self.topk_idx_t).contiguous()
        self.topk_weights = topk_weights.contiguous()

    def step(self) -> torch.Tensor:
        """One MoE layer: dispatch -> expert -> combine, PAIRED (combine consumes the fresh
        dispatch's handle). Returns the combined [local_tokens, hidden] bf16 output."""
        # Same arguments as Megatron's DeepEPDispatcher.token_dispatch.
        recv_x, recv_idx, recv_w, handle, _ = self.buf.dispatch(
            self.hidden,
            topk_idx=self.topk_idx,
            topk_weights=self.topk_weights,
            num_experts=self.cfg.num_experts,
            num_max_tokens_per_rank=self.cfg.per_rank_cap,
            num_sms=self.num_sms,
            do_cpu_sync=False,
            do_expand=False,
        )
        x = recv_x
        if self.expert is not None:
            # Megatron's DeepEP path: received count -> valid tokens, local expert ids
            # (start 0), expert output written straight into the combine input.
            self.recv_count.copy_(handle.psum_num_recv_tokens_per_scaleup_rank[-1:])
            self.expert(recv_x, recv_w, recv_idx, self.recv_count, 0, self.combine_input,
                        self.hidden.shape[0] * self.cfg.ep_size)
            x = self.combine_input
        # Same call as Megatron's DeepEPDispatcher.token_combine.
        combined, _, _ = self.buf.combine(x, handle=handle)
        return combined

    def decode_step(self, num_layers: int) -> torch.Tensor:
        """One full decode step across `num_layers` MoE layers: each layer does a
        fresh dispatch->combine. DeepEP has NO once-per-step metadata -- its
        routing/count-exchange is per-layer, inside each dispatch. Returns the last
        layer's combined output."""
        for _ in range(num_layers):
            out = self.step()
        return out

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Round-trip correctness of the identity-expert layer (the timed step() with no
        expert) on RANDOM per-element tensors -> list of (name, ok, detail).

        With do_expand=False, dispatch delivers one copy of each token per destination rank,
        and the rank-layout combine (allow_multiple_reduction=True) sums one copy per receiving
        rank, so the combined result for source token t must equal
        (#distinct destination ranks for t) * x[t], element-wise. x is random per element
        (distinct per token AND per hidden column), so the element-wise check also catches
        column/stride displacement and any cross-token contamination -- not just a wrong
        reduction count. Tested with all ranks populated (B=2*ep) and 0-token ranks (B=1)."""
        assert self.expert is None, "validate() checks the identity-expert layer"
        world, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts
        epr = E // world  # contiguous experts per rank -> dest rank = expert // epr
        out = []
        for B in (world * 2, 1):
            counts = [B // world + (1 if r < B % world else 0) for r in range(world)]
            n = counts[rank]
            gen = torch.Generator(device=dev).manual_seed(770701 + B + rank * 131)
            x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                 if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
            idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
                   if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
            combined = self.functional_roundtrip(x, idx)
            torch.cuda.synchronize()
            if n > 0:
                m = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                 device=dev, dtype=torch.float32).view(n, 1)
                got = combined.to(torch.float32)
                ref = m * x.to(torch.float32)
                ok = bool(torch.allclose(got, ref, rtol=0.03, atol=0.05))
                detail = (f"n={n} max|got-m*x|={float((got - ref).abs().max()):.4f} "
                          f"m(#dest_ranks)={m.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
            else:
                ok, detail = True, "n=0 (idle rank still participates in the A2A)"
            out.append((f"DeepEP dispatch->combine B={B:<4d}", ok, detail))
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """The timed identity-expert layer on the given tokens, returning the combined output
        [n,H] == (#distinct dest ranks for t) * x[t] -- the same quantity NVLS's masked-identity
        round-trip yields, for a cross-impl equality check in run.py."""
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx,
                         torch.ones(n, self.cfg.topk, device=self.device, dtype=torch.float32))
        return self.step()[:n]

    def destroy(self):
        if getattr(self, "buf", None) is not None:
            self.buf.destroy()
            self.buf = None
