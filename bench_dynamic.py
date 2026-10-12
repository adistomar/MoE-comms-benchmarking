# Copyright (c) 2026. Dynamic (AGV-V/RSV-V or all-to-all-V) dispatch/combine bencher.
"""
Isolated bencher for the dynamic kernels in nvls/torch_symm_triton/dynamic_collectives.py. Same
buffers, metadata and timed structure as bench_a2a.py; each kernel picks AG/RS or A2A on device:

  metadata (once per step) : fused_metadata_update -> [valid, offset, ep_max]
  dispatch (per MoE layer) : dynamic_dispatch_v (AGV-V if valid < threshold, else A2A-V)
  combine  (per MoE layer) : dynamic_combine_v  (RSV-V if valid < threshold, else A2A-V pull)

The threshold is the swept knob (run.py --dynamic-threshold): "ag" / "a2a" pin the mode, an
integer T switches to A2A at global batch B >= T.
"""

import torch
import torch.distributed as dist

from nvls.torch_symm_triton.dynamic_collectives import (
    FORCE_A2A,
    FORCE_AG_RS,
    dynamic_combine_v,
    dynamic_dispatch_v,
    kernel_stats,
)

from bench_a2a import A2ABencher
from common import Config


def parse_thresholds(spec: str) -> "list[tuple[str, int]]":
    """'ag,a2a,4096' -> [('ag', FORCE_AG_RS), ('a2a', FORCE_A2A), ('t4096', 4096)]."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if item == "ag":
            out.append(("ag", FORCE_AG_RS))
        elif item == "a2a":
            out.append(("a2a", FORCE_A2A))
        else:
            out.append((f"t{int(item)}", int(item)))
    return out


class DynamicBencher(A2ABencher):
    name = "dynamic"

    def __init__(self, cfg: Config, group, a2a_threshold: int):
        super().__init__(cfg, group)
        self.a2a_threshold = a2a_threshold

    def set_a2a_threshold(self, a2a_threshold: int):
        """Change the AG/RS -> A2A switch point (a kernel argument; no rebuild)."""
        self.a2a_threshold = a2a_threshold

    def kernel_stats(self) -> str:
        return " ".join(f"{k}: {regs} regs, {spills} B spilled"
                        for k, (regs, spills) in kernel_stats().items())

    def dispatch(self):
        dynamic_dispatch_v(
            self.recv_h["tensor"], self.recv_r["tensor"], self.recv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.recv_h["handle"], self.recv_r["handle"], self.recv_p["handle"],
            valid_tokens=self.step_metadata[0:1],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_local_experts=self.cfg.num_local_experts,
            a2a_threshold=self.a2a_threshold,
            max_num_blocks=self.num_sms,
        )

    def combine(self):
        dynamic_combine_v(
            self.out,
            self.partial["tensor"],
            self.partial["handle"],
            self.in_routing,
            valid_tokens=self.step_metadata[0:1],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_local_experts=self.cfg.num_local_experts,
            a2a_threshold=self.a2a_threshold,
            max_num_blocks=self.num_sms,
        )

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Known-value checks in both pinned modes -> list of (name, ok, detail).

        Dispatch: routing + probs rows [0, valid) bit-exact on every rank; hidden bit-exact on every
        row in AG mode, and on routed rows only in A2A mode (other rows keep a NaN sentinel).
        Combine: each rank exposes partials on the rows routed to it. Other rows hold zeros in RS
        mode (the multicast reduce sums every rank's row) and NaN in A2A mode (a pull from a
        non-destination poisons the output). The result must equal bf16(m * partial[row]),
        m = #distinct destination ranks: bit-exact in A2A mode, within one bf16 rounding step in RS
        mode (the switch's final fp32 -> bf16 rounding is not under our control).
        Tested with all ranks populated (B=2*ep) and with 0-token ranks (B=1).
        """
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        out = []
        saved_threshold = self.a2a_threshold
        for mode, threshold in (("AG/RS", FORCE_AG_RS), ("A2A", FORCE_A2A)):
            self.set_a2a_threshold(threshold)
            a2a = threshold == FORCE_A2A
            for B in (w * 2, 1):
                counts = [B // w + (1 if r < B % w else 0) for r in range(w)]
                n, off, valid = counts[rank], sum(counts[:rank]), sum(counts)
                gen = torch.Generator(device=dev).manual_seed(20260712 + B)  # identical on all ranks
                full_h = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
                full_r = self._validation_routing(valid, gen)
                full_p = torch.rand(valid, K, generator=gen, device=dev)
                local = self._local_rows(full_r)
                received = local if a2a else torch.ones_like(local)

                self.setup_batch(full_h[off:off + n].contiguous(), full_r[off:off + n].contiguous(),
                                 full_p[off:off + n].contiguous())
                self.metadata()
                recv_h = self.recv_h["tensor"].view(gcap, H)
                recv_h[:valid] = float("nan")
                torch.cuda.synchronize()
                dist.barrier(self.group)  # every sentinel is written before anyone dispatches
                self.dispatch()
                torch.cuda.synchronize()
                meta_ok = (torch.equal(self.recv_r["tensor"].view(gcap, K)[:valid], full_r)
                           and torch.equal(self.recv_p["tensor"].view(gcap, K)[:valid], full_p))
                recv_ok = (torch.equal(recv_h[:valid][received], full_h[received])
                           and bool(torch.isnan(recv_h[:valid][~received].float()).all()))
                out.append((f"Dynamic {mode:5s} dispatch B={B:<4d}", meta_ok and recv_ok,
                            f"routing+probs bit-exact on all rows; hidden bit-exact on "
                            f"{int(received.sum())}/{valid} rows, NaN elsewhere"))

                part_full = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
                partial = self.partial["tensor"].view(gcap, H)
                partial[:valid] = float("nan") if a2a else 0.0
                partial[:valid][local] = part_full[local]
                torch.cuda.synchronize()
                dist.barrier(self.group)
                self.combine()
                torch.cuda.synchronize()
                if n > 0:
                    m = self._num_dest_ranks(full_r[off:off + n]).view(n, 1)
                    ref = (m * part_full[off:off + n].float()).to(torch.bfloat16)
                    exact = torch.equal(self.out, ref)
                    comb_ok = exact or (not a2a and torch.allclose(
                        self.out.float(), ref.float(), rtol=2**-7, atol=0.0))
                    diff = float((self.out.float() - ref.float()).abs().max())
                    detail = (f"output == bf16(m*partial) {'bit-exact' if exact else f'max|diff|={diff:.3g}'}; "
                              f"local_tokens={n}, m={m.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
                else:
                    comb_ok, detail = True, "n=0 (idle rank still participates)"
                out.append((f"Dynamic {mode:5s} combine  B={B:<4d}", comb_ok, detail))
        self.set_a2a_threshold(saved_threshold)
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Dispatch -> identity expert -> combine at the current threshold, returning [n,H] ==
        (#distinct destination ranks for t) * x[t] (cross-impl check). Identity expert: each rank
        copies the received rows routed to it into its partials and zeros the rest (RS mode sums
        every rank's row)."""
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx, torch.ones(n, K, device=self.device, dtype=torch.float32))
        self.metadata()
        self.dispatch()
        valid = int(self.step_metadata[0].item())
        recv_h = self.recv_h["tensor"].view(gcap, H)[:valid]
        local = self._local_rows(self.recv_r["tensor"].view(gcap, K)[:valid])
        self.partial["tensor"].view(gcap, H)[:valid] = torch.where(
            local.view(valid, 1), recv_h, torch.zeros((), dtype=recv_h.dtype, device=self.device))
        self.combine()
        return self.out
