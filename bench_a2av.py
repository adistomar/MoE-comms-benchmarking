# Copyright (c) 2026. All-to-all-v (A2AV) dispatch/combine bencher.
"""
All-to-all-v MoE dispatch/combine bencher, built on the SAME vendored NVLS
`torch_symm_triton` collectives as bench_nvls.py, but moving the HIDDEN activations
by unicast to only a token's destination ranks (routing-driven) instead of
multicasting to every rank. Two variants share everything except the combine:

  Variant a2av     : A2AV dispatch  +  A2AV *pull* combine (owner-driven gather + fp32 sum).
  Variant a2av_rs  : A2AV dispatch  +  reduce-scatter-v combine (NVLink-SHARP switch reduce).

Both keep the NVLS design invariants so the surrounding harness is untouched:
  * DENSE layout: a token from this rank at local index t lands at the SAME global
    offset `rank_token_offset + t` on every destination rank (source-based). This is
    byte-identical to the AGV layout; only HIDDEN's store target changes (multicast ->
    per-destination unicast via `_SymmetricMemory.buffer_ptrs_dev`).
  * ROUTING + PROBS stay full all-gather-v (multicast), so a vLLM-style compute would be
    unchanged and both combines are correct (every rank sees every token's routing).
  * OUTPUT buffer is bf16 and SEPARATE from the dispatch buffers (never reused).

  dispatch (per MoE layer) : multimem_a2av_dispatch_3tensor(hidden[A2AV], routing/probs[AGV])
  combine  (per MoE layer) : multimem_a2av_combine (pull)  OR  multimem_reduce_scatter_v
  metadata (once per step) : fused_metadata_update -> [valid, rank_token_offset, ep_max]

The A2AV *pull* combine has each owner read its tokens' expert outputs from only their
destination ranks' output buffers (via buffer_ptrs_dev) and sum in fp32 -- minimal traffic
(reads ~avg_dest copies, not all EP), at the cost of coupling the reduction to remote-read
latency. The routing all-gather is what lets every rank derive a token's destination ranks.

As in bench_nvls.py, fused_metadata_update is routing-independent and runs ONCE per decode
step (first MoE layer); the per-layer timing covers only dispatch + combine. The timed
decode_step operates on a pre-filled output buffer (the collectives are measured in
isolation, no expert GEMM); functional_roundtrip wires dispatch->identity->combine for
correctness and yields (#distinct destination ranks)*x, matching NVLS/DeepEP.
"""

import torch

from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import fused_metadata_update
from nvls.torch_symm_triton.variable_collectives import (
    multimem_a2av_combine,
    multimem_a2av_dispatch_3tensor,
    multimem_reduce_scatter_v,
)

from common import Config, size_mb

# Fixed CTA-block cap, identical to NVLS (this bounds how many SMs the comm occupies).
# A2AV is NOT swept and is independent of DeepEP's --deepep-num-sms.
A2AV_MAX_BLOCKS = 148


class _A2AVBencher:
    """Shared A2AV dispatch + build/setup/metadata/validate. Subclasses set `name` and
    `combine_mode` ('scatter' or 'rsv') and (for scatter) override combine()."""

    name = "a2av_base"
    combine_mode = "rsv"

    def __init__(self, cfg: Config, group):
        self.cfg = cfg
        self.group = group
        # Fixed block cap passed to the A2AV / RSV kernels as max_num_blocks (and reported as
        # this impl's block count). Not a swept knob.
        self.num_sms = A2AV_MAX_BLOCKS
        self.device = torch.device("cuda", cfg.local_rank)
        self._built = False

    # -- one-time allocation (collective) --------------------------------------
    def build(self):
        cfg = self.cfg
        gmax = cfg.global_cap
        K, H = cfg.topk, cfg.hidden

        def buf(key, shape, dtype):
            b = SymmetricMemoryManager.get_buffer(
                key, process_group=self.group, size_mb=size_mb(shape, dtype)
            ).maybe_get_tensor(shape, dtype=dtype)
            if b["handle"] is None:
                raise RuntimeError(
                    f"A2AV symmetric-memory init failed for '{key}'. Requires a GPU NVLink "
                    f"domain with torch.distributed._symmetric_memory + multicast and triton."
                )
            return b

        # Dispatch buffers: hidden (A2AV target, bf16), routing (int64) + probs (fp32) (AGV).
        self.agv_h = buf("a2av_agv_h", [gmax, H], torch.bfloat16)
        self.agv_r = buf("a2av_agv_r", [gmax, K], torch.int64)
        self.agv_p = buf("a2av_agv_p", [gmax, K], torch.float32)
        # SEPARATE bf16 output buffer (never reused for dispatch). Both combines read it:
        # RSV via multicast-reduce; scatter via per-rank buffer_ptrs_dev. bf16 halves the
        # combine bytes; the reduction still accumulates in fp32 (RSV: acc::f32; scatter: fp32).
        self.out_buf = buf("a2av_out", [gmax, H], torch.bfloat16)
        self.meta = buf("a2av_meta", [cfg.ep_size], torch.int32)

        # A2AV unicast (dispatch) and pull combine both need per-rank symmetric pointers.
        if not hasattr(self.agv_h["handle"], "buffer_ptrs_dev"):
            raise RuntimeError(
                "A2AV requires torch _SymmetricMemory.buffer_ptrs_dev (per-rank symmetric "
                "pointers); this torch build does not expose it."
            )

        # [valid_tokens, rank_token_offset, ep_max_tokens]; written in-place each step.
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
        # Local (non-symmetric) inputs; kept stable for graph capture/replay. `in_routing` is
        # reused by the scatter reduce to derive destination ranks.
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.contiguous()
        self.in_probs = topk_weights.contiguous()
        # Persistent combine output (stable address for graph replay), bf16 for both variants.
        self.out = torch.empty(self.local_tokens, self.cfg.hidden,
                               dtype=torch.bfloat16, device=self.device)

    # -- timed / setup ops -----------------------------------------------------
    def metadata(self):
        """Once-per-step token-count exchange (sum / prefix / max). Routing-independent."""
        fused_metadata_update(
            local_tokens=self.local_tokens,
            local_buf=self.meta["tensor"],
            symm_mem_hdl=self.meta["handle"],
            step_metadata=self.step_metadata,
        )

    def dispatch(self):
        """A2AV-V: unicast hidden to each token's destination ranks; AGV routing + probs."""
        multimem_a2av_dispatch_3tensor(
            self.agv_h["tensor"], self.agv_r["tensor"], self.agv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.agv_h["handle"], self.agv_r["handle"], self.agv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_experts=self.cfg.num_experts,
            max_num_blocks=self.num_sms,
        )

    def combine(self):
        """Reduce-scatter-v combine (NVLS switch-reduce over all peers, bf16). The A2AV
        pull variant overrides this."""
        multimem_reduce_scatter_v(
            self.out,
            self.out_buf["tensor"],
            self.out_buf["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            max_num_blocks=self.num_sms,
        )

    def decode_step(self, num_layers: int):
        """One full decode step across `num_layers` MoE layers: token-count metadata runs
        ONCE per step (first MoE layer), then each layer does A2AV dispatch -> combine."""
        self.metadata()
        for _ in range(num_layers):
            self.dispatch()
            self.combine()

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Known-value correctness checks (random per-element tensors) -> list of
        (name, ok, detail). All ranks generate the SAME full [valid,H] tensors from a fixed
        seed, so each rank's owned slice is consistent and the expected gathered result is
        exactly that full tensor. Two checks per batch:

          dispatch : routing (int64) + probs (fp32) are all-gathered bit-exact on every rank;
                     hidden (bf16) is all-to-all-v -- present only at global offsets whose
                     token routes to a LOCAL expert of this rank -- checked exactly there.
          round-trip: dispatch -> masked identity expert -> combine == (#distinct dest ranks)*x
                     (bf16 tolerance; the reduction accumulates in fp32 then rounds to bf16).

        Tested with all ranks populated (B=2*ep) and with 0-token ranks (B=1).
        """
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts, self.cfg.global_cap
        epr = E // w
        out = []
        for B in (w * 2, 1):
            counts = [B // w + (1 if r < B % w else 0) for r in range(w)]
            n, off, valid = counts[rank], sum(counts[:rank]), sum(counts)
            gen = torch.Generator(device=dev).manual_seed(20260701 + B)  # identical on all ranks
            full_h = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
            full_r = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K]
                                   for _ in range(valid)])
                      if valid > 0 else torch.empty(0, K, device=dev, dtype=torch.int64)).to(torch.int64)
            full_p = torch.rand(valid, K, generator=gen, device=dev)

            # -- dispatch check --
            self.setup_batch(full_h[off:off + n].contiguous(),
                             full_r[off:off + n].contiguous(),
                             full_p[off:off + n].contiguous())
            self.metadata()
            self.dispatch()
            torch.cuda.synchronize()
            agv_r_ok = torch.equal(self.agv_r["tensor"].view(gcap, K)[:valid], full_r)
            agv_p_ok = torch.equal(self.agv_p["tensor"].view(gcap, K)[:valid], full_p)
            if valid > 0:
                local_mask = ((full_r >= rank * epr) & (full_r < (rank + 1) * epr)).any(dim=1)
            else:
                local_mask = torch.zeros(0, dtype=torch.bool, device=dev)
            agv_h_v = self.agv_h["tensor"].view(gcap, H)[:valid]
            agv_h_ok = bool(local_mask.sum() == 0) or torch.equal(
                agv_h_v[local_mask], full_h[local_mask])
            out.append((f"A2AV dispatch       B={B:<4d}",
                        bool(agv_r_ok and agv_p_ok and agv_h_ok),
                        "routing+probs all-gathered bit-exact; hidden unicast lands at dest offsets"))

            # -- round-trip check (dispatch -> identity -> combine == m*x) --
            rt = self.functional_roundtrip(full_h[off:off + n].contiguous(),
                                           full_r[off:off + n].contiguous())
            torch.cuda.synchronize()
            if n > 0:
                m = torch.tensor([torch.unique(full_r[off + t] // epr).numel() for t in range(n)],
                                 device=dev, dtype=torch.float32).view(n, 1)
                ref = m * full_h[off:off + n].to(torch.float32)
                rt_ok = torch.allclose(rt.to(torch.float32), ref, rtol=2e-2, atol=2e-2)
            else:
                rt_ok = True
            out.append((f"A2AV round-trip     B={B:<4d}", bool(rt_ok),
                        f"combine({self.combine_mode}) == (#dest ranks)*x (bf16); local_tokens={n}"))
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Full functional dispatch -> (masked identity expert) -> combine, returning the
        combined output [n,H] for this rank's tokens. Identity expert: rank r adds a source
        token's (A2AV-delivered) value to its combine sum iff that token routes >=1 expert
        local to rank r, UNWEIGHTED. So the result is (#distinct destination ranks)*x[t] --
        the SAME quantity NVLS/DeepEP yield, which lets run.py cross-check them.

        NOTE: this wires dispatch->combine functionally (the output buffer is derived from the
        A2AV dispatch output), unlike the timed decode_step which combines a pre-filled buffer.
        """
        w, rank = self.cfg.ep_size, self.cfg.rank
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        epr = self.cfg.num_experts // w
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx,
                         torch.ones(n, K, device=self.device, dtype=torch.float32))
        self.metadata()
        self.dispatch()  # A2AV: unicast hidden to dest ranks; AGV routing/probs to all
        valid = int(self.step_metadata[0].item())
        gh = self.agv_h["tensor"].view(gcap, H)[:valid]  # hidden present at dest offsets on this rank
        gr = self.agv_r["tensor"].view(gcap, K)[:valid]  # full gathered routing (expert ids)
        local = ((gr >= rank * epr) & (gr < (rank + 1) * epr)).any(dim=1)  # token hits a local expert
        # Identity expert: out_buf[g] = x (unweighted) if token g routes to a local expert,
        # else 0. Where local, gh[g] is the A2AV-delivered token (valid); elsewhere it is
        # don't-care (masked to 0 here).
        self.out_buf["tensor"].view(gcap, H)[:valid] = torch.where(
            local.view(valid, 1), gh,
            torch.zeros((), dtype=torch.bfloat16, device=self.device))
        self.combine()  # pull: gather+fp32-sum over dest ranks; rsv: switch-reduce over all ranks
        return self.out


class A2AVBencher(_A2AVBencher):
    """A2AV dispatch + A2AV *pull* combine: for each of this rank's tokens, read its expert
    outputs from only its destination ranks' output buffers (per-rank buffer_ptrs_dev) and sum
    in fp32 -> bf16. An owner-driven gather (minimal traffic, coupled to remote-read latency)."""

    name = "a2av"
    combine_mode = "pull"

    def combine(self):
        multimem_a2av_combine(
            self.out,
            self.out_buf["tensor"],
            self.in_routing,
            self.out_buf["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_experts=self.cfg.num_experts,
            max_num_blocks=self.num_sms,
        )


class A2AVRSBencher(_A2AVBencher):
    """A2AV dispatch + reduce-scatter-v combine (NVLink-SHARP switch reduce)."""

    name = "a2av_rs"
    combine_mode = "rsv"
