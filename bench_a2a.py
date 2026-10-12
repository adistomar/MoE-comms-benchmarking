# Copyright (c) 2026. All-to-all-V (unicast dispatch / pull combine) dispatch/combine bencher.
"""
Isolated bencher for the all-to-all-V kernels in nvls/torch_symm_triton/all_to_all_v.py.
Same structure and buffers as bench_nvls.py (AGV-V row indexing, once-per-step metadata),
but dispatch unicasts each hidden row only to its destination ranks and combine pulls the
partial outputs back from those ranks:

  metadata (once per step) : fused_metadata_update -> [valid, offset, ep_max]
  dispatch (per MoE layer) : a2a_dispatch_v(hidden -> dst ranks; routing, probs -> all ranks)
  combine  (per MoE layer) : a2a_combine_v(pull bf16 partials from dst ranks, fp32 sum)

Like the NVLS bencher, the timed step reduces a pre-filled partial buffer (no expert compute).
The combine buffer is bf16 with fp32 accumulation, matching the NVLS (bf16 RSV) and DeepEP
(bf16 combine) setups.
"""

import torch
import torch.distributed as dist

from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import fused_metadata_update
from nvls.torch_symm_triton.all_to_all_v import a2a_combine_v, a2a_dispatch_v

from bench_nvls import NVLS_MAX_BLOCKS, ensure_signal_pad_size
from common import Config, size_mb


class A2ABencher:
    name = "a2a"

    def __init__(self, cfg: Config, group):
        self.cfg = cfg
        self.group = group
        # Same fixed CTA cap as NVLS (one CTA per token); not swept.
        self.num_sms = NVLS_MAX_BLOCKS
        self.device = torch.device("cuda", cfg.local_rank)
        self._built = False

    # -- one-time allocation (collective) --------------------------------------
    def build(self):
        cfg = self.cfg
        gmax = cfg.global_cap
        K, H = cfg.topk, cfg.hidden

        # Same grid the dispatch/combine wrappers launch; the barrier needs num_blocks * EP
        # signal slots, so size the pad before any symmetric allocation.
        self.num_blocks = min(cfg.per_rank_cap, self.num_sms)
        self.signal_pad_bytes = ensure_signal_pad_size(self.num_blocks, cfg.ep_size)

        def buf(key, shape, dtype):
            b = SymmetricMemoryManager.get_buffer(
                key, process_group=self.group, size_mb=size_mb(shape, dtype)
            ).maybe_get_tensor(shape, dtype=dtype)
            if b["handle"] is None:
                raise RuntimeError(f"A2A symmetric-memory init failed for '{key}'.")
            return b

        # Own buffers (not NVLS's ep_* keys), so both benchers can run in one process.
        self.recv_h = buf("a2a_recv_h", [gmax, H], torch.bfloat16)
        self.recv_r = buf("a2a_recv_r", [gmax, K], torch.int64)
        self.recv_p = buf("a2a_recv_p", [gmax, K], torch.float32)
        self.partial = buf("a2a_partial", [gmax, H], torch.bfloat16)
        self.meta = buf("a2a_meta", [cfg.ep_size], torch.int32)

        # [valid_tokens, rank_token_offset, ep_max_tokens]; written in-place each step.
        self.step_metadata = torch.zeros(3, dtype=torch.int32, device=self.device)
        # Pre-fill the partial buffer so combine timing operates on valid data.
        self.partial["tensor"].normal_()
        self._built = True

    # -- per-batch setup -------------------------------------------------------
    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        self.local_tokens = hidden.shape[0]
        # Local inputs; kept stable for graph capture/replay. Combine reuses the routing.
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.to(torch.int64).contiguous()
        self.in_probs = topk_weights.contiguous()
        self.out = torch.empty(self.local_tokens, self.cfg.hidden,
                               dtype=torch.bfloat16, device=self.device)

    # -- timed / setup ops -----------------------------------------------------
    def metadata(self):
        """Once-per-step token-count exchange (sum / prefix / max)."""
        fused_metadata_update(
            local_tokens=self.local_tokens,
            local_buf=self.meta["tensor"],
            symm_mem_hdl=self.meta["handle"],
            step_metadata=self.step_metadata,
        )

    def dispatch(self):
        a2a_dispatch_v(
            self.recv_h["tensor"], self.recv_r["tensor"], self.recv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.recv_h["handle"], self.recv_r["handle"], self.recv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_local_experts=self.cfg.num_local_experts,
            max_num_blocks=self.num_sms,
        )

    def combine(self):
        a2a_combine_v(
            self.out,
            self.partial["tensor"],
            self.partial["handle"],
            self.in_routing,
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_local_experts=self.cfg.num_local_experts,
            max_num_blocks=self.num_sms,
        )

    def step(self):
        """One MoE layer: A2A-V dispatch -> (identity expert) -> A2A-V combine."""
        self.dispatch()
        self.combine()

    def decode_step(self, num_layers: int):
        """One decode step: metadata once (first MoE layer), then dispatch -> combine per layer."""
        self.metadata()
        for _ in range(num_layers):
            self.dispatch()
            self.combine()

    # -- correctness -----------------------------------------------------------
    def _local_rows(self, routing):
        """[rows] bool: the row has at least one expert on this rank."""
        epr = self.cfg.num_local_experts
        lo, hi = self.cfg.rank * epr, (self.cfg.rank + 1) * epr
        return ((routing >= lo) & (routing < hi)).any(dim=1)

    def _num_dest_ranks(self, routing):
        """[rows] float: number of distinct destination ranks per row (0 for padding rows)."""
        w, epr = self.cfg.ep_size, self.cfg.num_local_experts
        ranks = torch.arange(w, device=routing.device).view(1, w, 1)
        hit = ((routing.unsqueeze(1) >= ranks * epr) & (routing.unsqueeze(1) < (ranks + 1) * epr))
        return hit.any(dim=2).sum(dim=1).to(torch.float32)

    def _validation_routing(self, valid, gen):
        """Routing where each token reaches a random subset of ranks (so holes exist even at
        small EP), with distinct experts per token; the last token is padding (all -1)."""
        w, K, epr = self.cfg.ep_size, self.cfg.topk, self.cfg.num_local_experts
        dev = self.device
        min_ranks = -(-K // epr)  # enough experts on the chosen ranks for K distinct picks
        rows = []
        for _ in range(valid):
            nr = int(torch.randint(min_ranks, w + 1, (1,), generator=gen, device=dev))
            ranks = torch.randperm(w, generator=gen, device=dev)[:nr]
            experts = (ranks.view(-1, 1) * epr + torch.arange(epr, device=dev)).flatten()
            rows.append(experts[torch.randperm(experts.numel(), generator=gen, device=dev)[:K]])
        if not rows:
            return torch.empty(0, K, dtype=torch.int64, device=dev)
        routing = torch.stack(rows).to(torch.int64)
        if valid >= 2:
            routing[-1] = -1
        return routing

    def validate(self):
        """Known-value checks -> list of (name, ok, detail).

        Dispatch: routing + probs rows [0, valid) are bit-exact on every rank; hidden rows routed
        here are bit-exact; hidden rows NOT routed here keep a NaN sentinel (no over-sending).
        Combine: each rank exposes partials only on rows routed to it (NaN elsewhere, so a pull
        from a non-destination poisons the output); the result must equal
        bf16(m * partial[row]) bit-exactly, m = #distinct destination ranks (0 for padding).
        Tested with all ranks populated (B=2*ep) and with 0-token ranks (B=1).
        """
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        out = []
        for B in (w * 2, 1):
            counts = [B // w + (1 if r < B % w else 0) for r in range(w)]
            n, off, valid = counts[rank], sum(counts[:rank]), sum(counts)
            gen = torch.Generator(device=dev).manual_seed(20260709 + B)  # identical on all ranks
            full_h = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
            full_r = self._validation_routing(valid, gen)
            full_p = torch.rand(valid, K, generator=gen, device=dev)
            local = self._local_rows(full_r)

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
            recv_ok = (torch.equal(recv_h[:valid][local], full_h[local])
                       and bool(torch.isnan(recv_h[:valid][~local].float()).all()))
            out.append((f"A2A-V dispatch       B={B:<4d}", meta_ok and recv_ok,
                        f"routing+probs bit-exact on all rows; hidden bit-exact on "
                        f"{int(local.sum())}/{valid} routed rows, NaN holes untouched"))

            part_full = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
            partial = self.partial["tensor"].view(gcap, H)
            partial[:valid] = float("nan")
            partial[:valid][local] = part_full[local]
            torch.cuda.synchronize()
            dist.barrier(self.group)
            self.combine()
            torch.cuda.synchronize()
            if n > 0:
                m = self._num_dest_ranks(full_r[off:off + n]).view(n, 1)
                ref = (m * part_full[off:off + n].float()).to(torch.bfloat16)
                comb_ok = torch.equal(self.out, ref)
                detail = (f"output == bf16(m*partial) bit-exact; local_tokens={n}, "
                          f"m={m.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
            else:
                comb_ok, detail = True, "n=0 (idle rank still participates)"
            out.append((f"A2A-V combine        B={B:<4d}", comb_ok, detail))
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Dispatch -> identity expert -> combine, returning [n,H] == (#distinct destination
        ranks for t) * x[t], the same quantity NVLS and DeepEP produce (cross-impl check).
        Identity expert: each rank copies the received rows routed to it into its partials."""
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx, torch.ones(n, K, device=self.device, dtype=torch.float32))
        self.metadata()
        self.dispatch()
        valid = int(self.step_metadata[0].item())
        recv_h = self.recv_h["tensor"].view(gcap, H)[:valid]
        local = self._local_rows(self.recv_r["tensor"].view(gcap, K)[:valid])
        self.partial["tensor"].view(gcap, H)[:valid][local] = recv_h[local]
        self.combine()
        return self.out
