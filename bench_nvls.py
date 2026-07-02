# Copyright (c) 2026. NVLS (AGV-V / RSV-V) dispatch/combine bencher.
"""
Isolated NVLS collective bencher. Replicates exactly the collective sequence
that megatron NVLSAllGatherVDispatcher.token_dispatch / token_combine perform,
stripped of shared-expert / latent / expert-GEMM / preprocess machinery:

  dispatch (per MoE layer)  : multimem_all_gatherv_3tensor(hidden, routing, probs)
  combine  (per MoE layer)  : multimem_reduce_scatter_v(rsv)
  metadata (once per step)  : fused_metadata_update  -> [valid, offset, ep_max]

DECISION (see bench/README.md): fused_metadata_update runs ONCE per decode step
at the first MoE layer (verified: megatron inference/utils.py:157) and is
routing-independent, so it is EXCLUDED from the per-layer dispatch timing and
reported separately. We run it once in setup to populate _step_metadata, then
time only the AGV per iteration.
"""

import torch

from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import fused_metadata_update
from nvls.torch_symm_triton.variable_collectives import (
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)

from common import Config, size_mb

# Fixed CTA-block cap for the NVLS AGV/RSV comm kernels. They run one CTA per token, so
# this grid ceiling (the multimem `max_num_blocks`) bounds how many SMs the comm occupies.
# HARDCODED to 148 (the B200 SM count we standardize on; the device reports 152) and NOT
# swept, so NVLS always runs at ~all SMs. This is independent of DeepEP's --deepep-num-sms
# (no shared/parity knob) -- upstream Megatron capped it at 128; we raise it to 148.
NVLS_MAX_BLOCKS = 148


class NVLSBencher:
    name = "nvls"

    def __init__(self, cfg: Config, group):
        self.cfg = cfg
        self.group = group
        # Fixed block cap passed to AGV/RSV as max_num_blocks (and reported as NVLS's block
        # count). Not a swept knob -- NVLS always uses NVLS_MAX_BLOCKS.
        self.num_sms = NVLS_MAX_BLOCKS
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
                    f"NVLS symmetric-memory init failed for '{key}'. Requires a GPU "
                    f"NVLink domain with torch.distributed._symmetric_memory + multicast "
                    f"and triton."
                )
            return b

        self.agv_h = buf("ep_agv_h", [gmax, H], torch.bfloat16)
        self.agv_r = buf("ep_agv_r", [gmax, K], torch.int64)
        self.agv_p = buf("ep_agv_p", [gmax, K], torch.float32)
        self.rsv = buf("ep_rsv", [gmax, H], torch.float32)
        self.meta = buf("ep_meta", [cfg.ep_size], torch.int32)

        # [valid_tokens, rank_token_offset, ep_max_tokens]; written in-place each step.
        self.step_metadata = torch.zeros(3, dtype=torch.int32, device=self.device)
        # Pre-fill the RSV buffer so combine timing operates on valid data.
        self.rsv["tensor"].normal_()
        self._built = True

    # -- per-batch setup -------------------------------------------------------
    def setup_batch(self, hidden, topk_idx, topk_weights):
        """Store this step's local inputs. Metadata is published inside decode_step
        (once per step, at the first MoE layer)."""
        assert self._built
        self.local_tokens = hidden.shape[0]
        # Local (non-symmetric) AGV inputs; kept stable for graph capture/replay.
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.contiguous()
        self.in_probs = topk_weights.contiguous()
        # Persistent combine output (stable address for graph replay).
        self.out = torch.empty(self.local_tokens, self.cfg.hidden,
                               dtype=torch.float32, device=self.device)

    # -- timed / setup ops -----------------------------------------------------
    def metadata(self):
        """Once-per-step token-count exchange (sum / prefix / max)."""
        fused_metadata_update(
            local_tokens=self.local_tokens,
            local_buf=self.meta["tensor"],
            symm_mem_hdl=self.meta["handle"],
            step_metadata=self.step_metadata,
        )

    def step(self):
        """One MoE layer: AGV-V dispatch -> (identity expert) -> RSV-V combine."""
        self.dispatch()
        self.combine()

    def decode_step(self, num_layers: int):
        """One full decode step across `num_layers` MoE layers: the token-count
        metadata (fused_metadata_update) runs ONCE per step (first MoE layer), then
        each layer does AGV-V dispatch -> RSV-V combine."""
        self.metadata()
        for _ in range(num_layers):
            self.dispatch()
            self.combine()

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Known-value correctness checks with RANDOM per-element tensors -> list of
        (name, ok, detail). Values are random and distinct per element (so they vary
        across the hidden/topk dimension too), which makes the checks sensitive to
        column/stride placement, not just row routing. All ranks generate the SAME full
        [valid,H] tensor from a fixed seed, so each rank's owned slice is consistent and
        the expected gathered result is exactly that full tensor.

        AGV-V: the gather must reproduce the full random tensor bit-exactly (a bf16 copy),
        and likewise the full routing (int64) and probs (fp32).
        RSV-V: rsv is seeded identically on all ranks; the switch-reduced output for this
        rank's owned tokens must equal world * rsv[global_id], element-wise.
        Tested with all ranks populated (B=2*ep) and with 0-token ranks (B=1).
        """
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts, self.cfg.global_cap
        out = []
        for B in (w * 2, 1):
            counts = [B // w + (1 if r < B % w else 0) for r in range(w)]
            n, off, valid = counts[rank], sum(counts[:rank]), sum(counts)
            gen = torch.Generator(device=dev).manual_seed(20260701 + B)  # identical on all ranks
            full_h = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
            full_r = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(valid)])
                      if valid > 0 else torch.empty(0, K, device=dev, dtype=torch.int64)).to(torch.int64)
            full_p = torch.rand(valid, K, generator=gen, device=dev)
            self.setup_batch(full_h[off:off + n].contiguous(),
                             full_r[off:off + n].contiguous(),
                             full_p[off:off + n].contiguous())
            self.metadata()
            self.dispatch()
            torch.cuda.synchronize()
            agv_ok = (torch.equal(self.agv_h["tensor"].view(gcap, H)[:valid], full_h)
                      and torch.equal(self.agv_r["tensor"].view(gcap, K)[:valid], full_r)
                      and torch.equal(self.agv_p["tensor"].view(gcap, K)[:valid], full_p))
            out.append((f"NVLS AGV-V gather   B={B:<4d}", agv_ok,
                        "gathered [valid,H] == full random tensor bit-exact (hidden+routing+probs)"))
            rsv_full = torch.randn(valid, H, generator=gen, device=dev)  # identical on all ranks
            self.rsv["tensor"].view(gcap, H)[:valid] = rsv_full
            self.combine()
            torch.cuda.synchronize()
            rsv_ok = (n == 0) or torch.allclose(self.out, w * rsv_full[off:off + n],
                                                rtol=1e-4, atol=1e-4)
            out.append((f"NVLS RSV-V reduce   B={B:<4d}", rsv_ok,
                        f"output == world*rsv[global_id] element-wise; local_tokens={n}"))
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Full functional dispatch -> (masked identity expert) -> combine, returning the
        combined output [n,H] for this rank's owned tokens. Identity expert: rank r adds a
        source token's (gathered) value to its combine sum iff that token routed >=1 expert
        local to rank r (the routing mask Megatron applies), UNWEIGHTED. So the result is
        (#distinct destination ranks for t) * x[t] -- the SAME quantity DeepEP's rank-layout
        combine yields, which lets run.py cross-check NVLS(fp32) against DeepEP(bf16).

        NOTE: this wires dispatch->combine functionally (rsv derived from the AGV output),
        unlike the timed step() which reduces a disconnected pre-filled rsv."""
        w, rank = self.cfg.ep_size, self.cfg.rank
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        epr = self.cfg.num_experts // w  # contiguous experts per rank
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx,
                         torch.ones(n, K, device=self.device, dtype=torch.float32))
        self.metadata()
        self.dispatch()  # AGV-V: replicate all tokens (+routing) to every rank
        valid = int(self.step_metadata[0].item())
        gh = self.agv_h["tensor"].view(gcap, H)[:valid]  # gathered hidden (bf16), same on all ranks
        gr = self.agv_r["tensor"].view(gcap, K)[:valid]  # gathered routing (expert ids)
        local = ((gr >= rank * epr) & (gr < (rank + 1) * epr)).any(dim=1)  # token hits a local expert
        self.rsv["tensor"].view(gcap, H)[:valid] = torch.where(
            local.view(valid, 1), gh.to(torch.float32), torch.zeros((), device=self.device))
        self.combine()  # RSV-V: sum masked contributions across ranks, scatter to owner
        return self.out

    def dispatch(self):
        """AGV-V: gather all ranks' (hidden, routing, probs) into symm buffers."""
        multimem_all_gatherv_3tensor(
            self.agv_h["tensor"], self.agv_r["tensor"], self.agv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.agv_h["handle"], self.agv_r["handle"], self.agv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            max_num_blocks=self.num_sms,  # fixed 148-block cap (NVLS_MAX_BLOCKS)
        )

    def combine(self):
        """RSV-V: sum expert outputs across EP ranks, scatter to local tokens."""
        multimem_reduce_scatter_v(
            self.out,
            self.rsv["tensor"],
            self.rsv["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            max_num_blocks=self.num_sms,  # fixed 148-block cap (NVLS_MAX_BLOCKS)
        )
