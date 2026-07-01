# Copyright (c) 2026. DeepEP-v2 (elastic, A2A) dispatch/combine bencher.
"""
Isolated DeepEP-v2 collective bencher. Calls ElasticBuffer.dispatch / .combine
directly (already free of any expert-compute / shared-expert machinery).

Locked config (see bench/README.md):
  bf16 dispatch (use_fp8_dispatch=False), do_expand=False (pure collective,
  no expert-grouping permute — matches NVLS deferring permute to the GEMM),
  do_cpu_sync=False (device-resident, graph-safe), allow_hybrid_mode=False
  (single NVLink domain -> flat/direct path), allow_multiple_reduction=True
  (since ep_size(4) <= topk(22) -> rank-layout combine: one reduced copy per
  rank, comparable to NVLS RSV). num_sms is the swept knob; combine reuses it.

We time the PAIRED dispatch->combine as one decode step: DeepEP combine consumes
the handle + buffer state produced by its immediately-preceding dispatch, so the
two belong together (timing them separately is unsupported).
"""

import torch

import deep_ep

from common import Config


class DeepEPBencher:
    name = "deepep"

    def __init__(self, cfg: Config, group, num_sms: int):
        self.cfg = cfg
        self.group = group
        self.num_sms = num_sms
        self.device = torch.device("cuda", cfg.local_rank)
        self._built = False

    def build(self):
        cfg = self.cfg
        self.buf = deep_ep.ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=cfg.per_rank_cap,
            hidden=cfg.hidden,
            num_topk=cfg.topk,
            use_fp8_dispatch=False,            # bf16 dispatch
            deterministic=False,
            allow_hybrid_mode=False,           # single NVLink domain: flat/direct path
            allow_multiple_reduction=True,     # rank-layout combine (ep<=topk)
            prefer_overlap_with_compute=True,
            explicitly_destroy=True,
            num_gpu_timeout_secs=100,
            num_cpu_timeout_secs=100,
        )
        # topk index dtype the build expects (int64 by default; int32 if built with
        # EP_NUM_TOPK_IDX_BITS=32). NVLS routing is int64 — note any mismatch.
        self.topk_idx_t = getattr(deep_ep, "topk_idx_t", torch.int64)
        self._built = True

    def set_num_sms(self, num_sms: int):
        """Change the SM count used by dispatch/combine (buffer is SM-count-agnostic,
        so no rebuild needed; each value JIT-compiles its own kernels on first use)."""
        self.num_sms = num_sms

    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        self.hidden = hidden.contiguous()
        self.topk_idx = topk_idx.to(self.topk_idx_t).contiguous()
        self.topk_weights = topk_weights.contiguous()

    def _dispatch_call(self):
        return self.buf.dispatch(
            self.hidden,
            topk_idx=self.topk_idx,
            topk_weights=self.topk_weights,
            num_experts=self.cfg.num_experts,
            num_max_tokens_per_rank=self.cfg.per_rank_cap,
            expert_alignment=1,
            num_sms=self.num_sms,
            num_qps=0,
            do_cpu_sync=False,                 # device-resident, graph-safe
            do_expand=False,                   # pure collective, no permute
            do_handle_copy=False,
            async_with_compute_stream=False,   # single-stream (graph-capturable)
            allocate_on_comm_stream=False,
        )

    def step(self):
        """One MoE layer: dispatch -> (identity expert) -> combine, PAIRED
        (combine consumes the fresh dispatch's handle)."""
        recv_x, _, _, handle, _ = self._dispatch_call()
        rx = recv_x if isinstance(recv_x, torch.Tensor) else recv_x[0]
        self.buf.combine(
            rx, handle=handle, topk_weights=None, bias=None,
            num_sms=self.num_sms, num_qps=0,
            async_with_compute_stream=False, allocate_on_comm_stream=False,
        )

    def decode_step(self, num_layers: int):
        """One full decode step across `num_layers` MoE layers: each layer does a
        fresh dispatch->combine. DeepEP has NO once-per-step metadata — its
        routing/count-exchange is per-layer, inside each dispatch."""
        for _ in range(num_layers):
            self.step()

    def destroy(self):
        if getattr(self, "buf", None) is not None:
            self.buf.destroy()
            self.buf = None
