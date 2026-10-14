# Copyright (c) 2026. Naive (unicast) AllGather-V dispatch / NVLS RSV-V combine bencher.
"""
Isolated bencher for the unicast all-gather-V in nvls/torch_symm_triton/naive_all_gather_v.py.
Identical to bench_nvls.py except for the dispatch transport:

  metadata (once per step) : fused_metadata_update -> [valid, offset, ep_max]
  dispatch (per MoE layer) : naive_all_gatherv_3tensor(hidden, routing, probs)
                             -- every row stored to every rank via peer pointers
  combine  (per MoE layer) : multimem_reduce_scatter_v(rsv)   (same NVLS RSV-V as nvls)

Same buffers, row layout, CTA cap, signal-pad sizing, validation checks and functional
round-trip as NVLSBencher, so naive_agv-vs-nvls isolates what NVLink multicast buys on dispatch.
"""

from nvls.torch_symm_triton.naive_all_gather_v import naive_all_gatherv_3tensor

from bench_nvls import NVLSBencher


class NaiveAllGatherVBencher(NVLSBencher):
    name = "naive_agv"
    # Own buffers (not NVLS's ep_* keys), so both benchers can run in one process.
    buffer_prefix = "naive_agv"
    label = "Naive"

    def dispatch(self):
        """Unicast AGV-V: store all ranks' (hidden, routing, probs) into every rank's buffers."""
        naive_all_gatherv_3tensor(
            self.agv_h["tensor"], self.agv_r["tensor"], self.agv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.agv_h["handle"], self.agv_r["handle"], self.agv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            max_num_blocks=self.num_sms,  # default_num_blocks(device, EP)
        )
