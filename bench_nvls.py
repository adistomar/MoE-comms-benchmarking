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


class NVLSBencher:
    name = "nvls"

    def __init__(self, cfg: Config, group):
        self.cfg = cfg
        self.group = group
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

    def dispatch(self):
        """AGV-V: gather all ranks' (hidden, routing, probs) into symm buffers."""
        multimem_all_gatherv_3tensor(
            self.agv_h["tensor"], self.agv_r["tensor"], self.agv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.agv_h["handle"], self.agv_r["handle"], self.agv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
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
        )
