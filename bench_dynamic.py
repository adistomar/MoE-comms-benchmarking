# Copyright (c) 2026. Regime-adaptive (dynamic) MoE dispatch/combine bencher.
"""
Dynamic dispatch/combine bencher: ONE captured kernel sequence that picks, on the
device and per step, between the NVLS all-gather-v / reduce-scatter-v collectives and
the all-to-all-v unicast dispatch + push combine.

Why this exists: the two families win in different regimes. AGv/RSv move every token to
every rank with a single `multimem` instruction per 128-bit chunk (the NVSwitch does the
fan-out and the reduction in hardware), which is unbeatable while a step is small enough
that launch and latency dominate. A2Av unicasts a token only to the ranks hosting one of
its top-k experts, so its wire bytes scale with avg_dest/ep_size instead of 1 — a saving
that only pays off once the link is the bottleneck. Decode steps sit in the first regime,
large prefill steps in the second, and a served model alternates between them
step-to-step.

The switch is INSIDE the kernels, driven by `step_metadata[0]` — the token count summed
over all EP ranks, published each step by `fused_metadata_update` and therefore identical
on every rank (so every rank takes the same branch and the barriers stay paired). Nothing
about the launch changes with the decision: same grid, same block size, same arguments;
only a value read from device memory differs. That is what makes it CUDA-graph safe — the
same graph replays as AGv/RSv on one step and as A2Av on the next, instead of needing a
separate graph per regime.

Dispatch and combine get independent thresholds because they cross over at different
token counts (unicast overtakes multicast before the software push overtakes the
in-switch reduce).

  dispatch (per layer) : multimem_dynamic_dispatch_3tensor
                         - HIDDEN: multimem.st to all ranks   OR   st.sys unicast to dests
                         - ROUTING/PROBS: always all-gather-v (compute path unchanged)
                         - A2Av branch also publishes the push combine's send index inline
  combine  (per layer) : multimem_dynamic_combine_push    (push, or an immediate return)
                         multimem_dynamic_combine_reduce  (local plane reduce, or RSv)
  metadata (per step)  : fused_metadata_update -> [valid, rank_token_offset, ep_max]

Buffers are bf16 on the combine wire, matching bench_nvls (`ep_rsv` is bf16) and
bench_a2av (`a2av_out` is bf16), so `nvls` / `a2av_push` / `dynamic` are directly
comparable. The kernels also support an fp32 expert-output buffer (what Megatron's
fused-MoE unpermute produces); that path is covered by the Megatron-side kernel test.

As in bench_nvls/bench_a2av, `fused_metadata_update` is routing-independent and runs ONCE
per decode step (first MoE layer); the per-layer timing covers only dispatch + combine.
The timed decode_step combines a pre-filled output buffer (collectives measured in
isolation, no expert GEMM); functional_roundtrip wires dispatch->identity->combine for
correctness and yields (#distinct destination ranks)*x, matching NVLS/A2AV/DeepEP.
"""

import torch
import torch.distributed as dist

from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import fused_metadata_update
from nvls.torch_symm_triton.variable_collectives import (
    A2AV_SEGMENT_TOKENS,
    a2av_index_buffer_shapes,
    multimem_dynamic_combine_push,
    multimem_dynamic_combine_reduce,
    multimem_dynamic_dispatch_3tensor,
)

from common import Config, size_mb

# Fixed CTA-block cap, identical to NVLS and A2AV (bounds how many SMs the comm takes).
DYNAMIC_MAX_BLOCKS = 148


class DynamicBencher:
    """Regime-adaptive dispatch/combine: AGv/RSv below the thresholds, A2Av at or above.

    `dispatch_threshold` / `combine_threshold` are GLOBAL token counts (summed over all
    EP ranks). Setting both above the largest batch pins the bencher to AGv/RSv; setting
    both to 0 pins it to A2Av — which is how validate() exercises each branch.
    """

    name = "dynamic"

    def __init__(
        self,
        cfg: Config,
        group,
        dispatch_threshold=None,
        combine_threshold=None,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        self.cfg = cfg
        self.group = group
        self.num_sms = DYNAMIC_MAX_BLOCKS
        self.device = torch.device("cuda", cfg.local_rank)
        self.dispatch_threshold = dispatch_threshold
        self.combine_threshold = combine_threshold
        # Expert-output buffer dtype. bf16 matches bench_nvls/bench_a2av, so the three
        # bench impls are directly comparable. fp32 matches what Megatron's fused-MoE
        # unpermute actually produces (it accumulates fp32 atomically), which makes the
        # RSv branch reduce 4-byte elements while the push branch still sends 2-byte ones
        # -- so the measured crossovers move, and only the fp32 numbers should be used to
        # pick the Megatron defaults.
        assert out_dtype in (torch.bfloat16, torch.float32)
        self.out_dtype = out_dtype
        self._built = False

    # -- one-time allocation (collective) --------------------------------------
    def _buf(self, key, shape, dtype):
        b = SymmetricMemoryManager.get_buffer(
            key, process_group=self.group, size_mb=size_mb(shape, dtype)
        ).maybe_get_tensor(list(shape), dtype=dtype)
        if b["handle"] is None:
            raise RuntimeError(
                f"Dynamic symmetric-memory init failed for '{key}'. Requires a GPU NVLink "
                f"domain with torch.distributed._symmetric_memory + multicast and triton."
            )
        return b

    def build(self):
        cfg = self.cfg
        gmax, K, H = cfg.global_cap, cfg.topk, cfg.hidden

        # Dispatch buffers: hidden (multicast OR unicast target, bf16), routing (int64)
        # and probs (fp32), the latter two always all-gathered.
        self.agv_h = self._buf("dyn_agv_h", [gmax, H], torch.bfloat16)
        self.agv_r = self._buf("dyn_agv_r", [gmax, K], torch.int64)
        self.agv_p = self._buf("dyn_agv_p", [gmax, K], torch.float32)
        # Expert-output buffer, SEPARATE from the dispatch buffers and symmetric: the RSv
        # branch reduce-loads it through the multicast pointer, the push branch reads it
        # locally. bf16 to match bench_nvls's `ep_rsv` and bench_a2av's `a2av_out`.
        self.out_buf = self._buf(
            f"dyn_out_{8 * self.out_dtype.itemsize}", [gmax, H], self.out_dtype
        )
        self.meta = self._buf("dyn_meta", [cfg.ep_size], torch.int32)
        # Push-combine receive planes: [ep, per_rank_cap, H], viewed flat.
        self.combine_recv = self._buf(
            "dyn_recv", [cfg.ep_size * cfg.per_rank_cap, H], torch.bfloat16
        )
        # Compact send-index buffers. Peers write into these; this rank reads its own.
        list_shape, count_shape = a2av_index_buffer_shapes(cfg.per_rank_cap, cfg.ep_size)
        self.recv_list = self._buf("dyn_list", list_shape, torch.int32)
        self.recv_count = self._buf("dyn_count", count_shape, torch.int32)
        # Per-token destination bitmask for this rank's own tokens (local, not symmetric).
        self.dest_mask = torch.zeros(cfg.per_rank_cap, dtype=torch.int64, device=self.device)

        for key, b in (
            ("dyn_agv_h", self.agv_h),
            ("dyn_recv", self.combine_recv),
            ("dyn_list", self.recv_list),
            ("dyn_count", self.recv_count),
        ):
            if not hasattr(b["handle"], "buffer_ptrs_dev"):
                raise RuntimeError(
                    f"Dynamic dispatch requires torch _SymmetricMemory.buffer_ptrs_dev "
                    f"(per-rank symmetric pointers) for '{key}'; this torch build does not "
                    f"expose it."
                )

        # [valid_tokens, rank_token_offset, ep_max_tokens]; written in-place each step.
        # step_metadata[0:1] is the switch signal read inside the kernels.
        self.step_metadata = torch.zeros(3, dtype=torch.int32, device=self.device)
        # Pre-fill the output buffer so combine timing operates on valid data.
        self.out_buf["tensor"].normal_()
        self._built = True

    # -- per-batch setup -------------------------------------------------------
    def setup_batch(self, hidden, topk_idx, topk_weights):
        """Store this step's local inputs. Metadata is published inside decode_step (once
        per step, at the first MoE layer)."""
        assert self._built
        self.local_tokens = hidden.shape[0]
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.contiguous()
        self.in_probs = topk_weights.contiguous()
        # Persistent combine output (stable address for graph replay), bf16 in both regimes.
        self.out = torch.empty(
            self.local_tokens, self.cfg.hidden, dtype=torch.bfloat16, device=self.device
        )

    # -- timed / setup ops -----------------------------------------------------
    def metadata(self):
        """Once-per-step token-count exchange (sum / prefix / max). Routing-independent.
        step_metadata[0] (the sum) is what the dynamic kernels branch on."""
        fused_metadata_update(
            local_tokens=self.local_tokens,
            local_buf=self.meta["tensor"],
            symm_mem_hdl=self.meta["handle"],
            step_metadata=self.step_metadata,
        )

    def dispatch(self):
        """AGv multicast or A2Av unicast for HIDDEN (decided on device); AGv for
        routing/probs. Publishes the push-combine send index when the combine will use it."""
        cfg = self.cfg
        multimem_dynamic_dispatch_3tensor(
            self.agv_h["tensor"],
            self.agv_r["tensor"],
            self.agv_p["tensor"],
            self.in_hidden,
            self.in_routing,
            self.in_probs,
            self.agv_h["handle"],
            self.agv_r["handle"],
            self.agv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            total_tokens=self.step_metadata[0:1],
            per_rank_max_tokens=cfg.per_rank_cap,
            num_experts=cfg.num_experts,
            dest_mask=self.dest_mask,
            recv_list=self.recv_list["tensor"],
            recv_count=self.recv_count["tensor"],
            recv_list_hdl=self.recv_list["handle"],
            recv_count_hdl=self.recv_count["handle"],
            dispatch_threshold=self.dispatch_threshold,
            combine_threshold=self.combine_threshold,
            max_num_blocks=self.num_sms,
        )

    def combine(self):
        """Two launches in both regimes: push-or-nothing, then local-reduce-or-RSv."""
        cfg = self.cfg
        multimem_dynamic_combine_push(
            out_buf=self.out_buf["tensor"],
            combine_recv_hdl=self.combine_recv["handle"],
            recv_list=self.recv_list["tensor"],
            recv_count=self.recv_count["tensor"],
            tokens_per_rank=self.meta["tensor"],
            total_tokens=self.step_metadata[0:1],
            per_rank_max_tokens=cfg.per_rank_cap,
            combine_threshold=self.combine_threshold,
            max_num_blocks=self.num_sms,
        )
        multimem_dynamic_combine_reduce(
            output_tensor=self.out,
            combine_recv=self.combine_recv["tensor"],
            dest_mask=self.dest_mask,
            rsv_tensor=self.out_buf["tensor"],
            rsv_hdl=self.out_buf["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            total_tokens=self.step_metadata[0:1],
            per_rank_max_tokens=cfg.per_rank_cap,
            combine_threshold=self.combine_threshold,
            max_num_blocks=self.num_sms,
        )

    def decode_step(self, num_layers: int):
        """One full decode step across `num_layers` MoE layers: token-count metadata runs
        ONCE per step (first MoE layer), then each layer does dispatch -> combine."""
        self.metadata()
        for _ in range(num_layers):
            self.dispatch()
            self.combine()

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Known-value checks for BOTH branches -> list of (name, ok, detail).

        Each batch size is checked twice with the thresholds forced, so a single run
        covers the AGv/RSv path, the A2Av path, and the fact that they agree:

          forced AGv  : hidden is all-gathered bit-exact on every rank; RSv output equals
                        ep_size * out_buf[global_id] (bf16 tolerance).
          forced A2Av : hidden is present (bit-exact) exactly at the global offsets whose
                        token routes to a LOCAL expert of this rank; the push+local-reduce
                        round-trip equals (#distinct dest ranks)*x.
          both        : the round-trip result is compared between the two regimes.

        Routing/probs must be gathered bit-exact in either regime. Tested with all ranks
        populated (B=2*ep) and with 0-token ranks (B=1).
        """
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts, self.cfg.global_cap
        epr = E // w
        saved = (self.dispatch_threshold, self.combine_threshold)
        out = []
        # Batch sizes: enough tokens per rank to span SEVERAL compaction segments, a
        # partially-filled multi-segment case, all ranks populated, and 0-token ranks.
        # The multi-segment cases matter: the builder's per-segment cumsum, the push
        # kernel's cdiv(n_src, SEG) walk and its per-(source, segment) recv_count
        # indexing are all dead code at local_tokens <= SEG, which is exactly where the
        # small cases sit -- so without these the timed large-step path is unvalidated.
        seg = A2AV_SEGMENT_TOKENS
        try:
            for B in (w * (2 * seg + 37), w * (seg + 1), w * 2, 1):
                counts = [B // w + (1 if r < B % w else 0) for r in range(w)]
                n, off, valid = counts[rank], sum(counts[:rank]), sum(counts)
                gen = torch.Generator(device=dev).manual_seed(20260901 + B)  # same on all ranks
                full_h = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
                full_r = (
                    torch.stack(
                        [torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(valid)]
                    )
                    if valid > 0
                    else torch.empty(0, K, device=dev, dtype=torch.int64)
                ).to(torch.int64)
                full_p = torch.rand(valid, K, generator=gen, device=dev)
                my_h = full_h[off : off + n].contiguous()
                my_r = full_r[off : off + n].contiguous()
                my_p = full_p[off : off + n].contiguous()
                if valid > 0:
                    local_mask = ((full_r >= rank * epr) & (full_r < (rank + 1) * epr)).any(dim=1)
                else:
                    local_mask = torch.zeros(0, dtype=torch.bool, device=dev)

                roundtrips = {}
                for regime, thr in (("AGv/RSv", 1 << 30), ("A2Av", 0)):
                    self.dispatch_threshold = thr
                    self.combine_threshold = thr

                    # -- dispatch placement check --
                    self.setup_batch(my_h, my_r, my_p)
                    # Poison the destination buffer so a branch that moves nothing cannot
                    # pass on data left behind by the other branch. Peers write into this
                    # rank's buffer, so the fills must all land before any dispatch starts.
                    self.agv_h["tensor"].fill_(-7.5)
                    self.agv_r["tensor"].fill_(-99)
                    self.agv_p["tensor"].fill_(-7.5)
                    torch.cuda.synchronize()
                    dist.barrier(self.group)
                    self.metadata()
                    self.dispatch()
                    torch.cuda.synchronize()
                    r_ok = torch.equal(self.agv_r["tensor"].view(gcap, K)[:valid], full_r)
                    p_ok = torch.equal(self.agv_p["tensor"].view(gcap, K)[:valid], full_p)
                    gh = self.agv_h["tensor"].view(gcap, H)[:valid]
                    if thr == 0:
                        # A2Av: only rows routing to a local expert are guaranteed present.
                        h_ok = bool(local_mask.sum() == 0) or torch.equal(
                            gh[local_mask], full_h[local_mask]
                        )
                        detail = "routing+probs all-gathered; hidden unicast lands at dest offsets"
                    else:
                        h_ok = valid == 0 or torch.equal(gh, full_h)
                        detail = "hidden+routing+probs all-gathered bit-exact"
                    out.append(
                        (f"dynamic dispatch[{regime:7s}] B={B:<4d}", bool(r_ok and p_ok and h_ok),
                         detail)
                    )

                    # -- round-trip check (dispatch -> identity expert -> combine) --
                    rt = self.functional_roundtrip(my_h, my_r)
                    torch.cuda.synchronize()
                    roundtrips[regime] = rt.clone()
                    if n > 0:
                        m = torch.tensor(
                            [torch.unique(full_r[off + t] // epr).numel() for t in range(n)],
                            device=dev,
                            dtype=torch.float32,
                        ).view(n, 1)
                        ref = m * full_h[off : off + n].to(torch.float32)
                        rt_ok = torch.allclose(rt.to(torch.float32), ref, rtol=2e-2, atol=2e-2)
                    else:
                        rt_ok = True
                    out.append(
                        (f"dynamic roundtrip[{regime:7s}] B={B:<4d}", bool(rt_ok),
                         f"combine == (#dest ranks)*x (bf16); local_tokens={n}")
                    )

                # -- the two regimes must agree --
                a, b = roundtrips["AGv/RSv"].to(torch.float32), roundtrips["A2Av"].to(torch.float32)
                if a.numel() == 0:
                    agree, dmax = True, 0.0
                else:
                    dmax = (a - b).abs().max().item()
                    scale = max(1e-6, a.abs().max().item())
                    agree = dmax / scale < 2e-2
                out.append(
                    (f"dynamic regimes agree      B={B:<4d}", bool(agree),
                     f"max|AGv-A2Av|={dmax:.6f}")
                )
        finally:
            self.dispatch_threshold, self.combine_threshold = saved
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Full functional dispatch -> (masked identity expert) -> combine, returning the
        combined output [n,H] for this rank's tokens. Identity expert: rank r adds a source
        token's dispatched value to its combine sum iff that token routes >=1 expert local
        to rank r, UNWEIGHTED. So the result is (#distinct destination ranks)*x[t] -- the
        SAME quantity NVLS/A2AV/DeepEP yield, which lets run.py cross-check them.

        NOTE: this wires dispatch->combine functionally (the output buffer is derived from
        the dispatch output), unlike the timed decode_step which combines a pre-filled
        buffer. Runs at whatever thresholds are currently set.
        """
        w, rank = self.cfg.ep_size, self.cfg.rank
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        epr = self.cfg.num_experts // w
        n = hidden.shape[0]
        self.setup_batch(
            hidden, topk_idx, torch.ones(n, K, device=self.device, dtype=torch.float32)
        )
        self.metadata()
        self.dispatch()
        valid = int(self.step_metadata[0].item())
        gh = self.agv_h["tensor"].view(gcap, H)[:valid]
        gr = self.agv_r["tensor"].view(gcap, K)[:valid]  # full gathered routing (expert ids)
        local = ((gr >= rank * epr) & (gr < (rank + 1) * epr)).any(dim=1)
        # Identity expert: out_buf[g] = x (unweighted) if token g routes to a local expert,
        # else 0. Where local, gh[g] is the dispatched token (valid in both regimes);
        # elsewhere it is don't-care under A2Av, so mask it to 0.
        self.out_buf["tensor"].view(gcap, H)[:valid] = torch.where(
            local.view(valid, 1), gh, torch.zeros((), dtype=torch.bfloat16, device=self.device)
        )
        self.combine()
        return self.out
