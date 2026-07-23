# Copyright (c) 2026. One-buffer aliased-multicast variable-multicast dispatch/combine bencher.
"""
The "one-buffer" vmcast variant (the honest, NVLS-structured design).

Instead of a separate padded buffer per nested multicast group, allocate ONE shared buffer per tensor
of shape [P*cap, *] and rendezvous it over EVERY nested aligned group (size-2 {0,1}.., size-4 {0-3}..,
... size-P {0..P-1}) -- torch symm-mem gives a DISTINCT multicast VA per group that ALIASES the one
allocation (confirmed by probe_aliased_mcast.py). Then:

  * Token (source rank r, local index i) always lives at its FIXED GLOBAL row  r*cap + i  -- exactly
    where NVLS's dense AllGather would put it. This is routing-INDEPENDENT: no offset prefix, no scan,
    no per-layer metadata collective. tok_slot is a static array computed once.
  * Per token we pick the smallest aligned group spanning {r} U dest_ranks(token) (tok_size) and multicast
    through THAT group's aliased VA: the store reaches only that group's ranks, but lands at the global
    row in the one shared buffer. Ranks outside the group keep a stale row there -- harmless, because
    they don't host the token's experts AND the combine reduces only over the token's group VA, so a
    stale out-of-group slot is never in the sum.

Honest per-layer layout = ONE elementwise pass computing tok_size -> gather tok_mc_* (device-only,
graph-capturable, NO collective). This runs INSIDE the timed decode_step, per layer -- structurally the
same shape as NVLS (which pays a routing-independent metadata exchange once per step). Reuses the fused
per-token multicast kernels (nvls/torch_symm_triton/fused_vmcast.py) unchanged.
"""
import torch
import torch.distributed as dist

import torch.distributed._symmetric_memory as symm_mem

from nvls.torch_symm_triton.fused_vmcast import (
    fused_vmcast_dispatch, fused_vmcast_combine, vmcast_compute_group_slotless,
)
from common import Config

NVLS_MAX_BLOCKS = 148


def nested_sizes(P, min_size):
    s, out = min_size, []
    while s <= P:
        out.append(s)
        s *= 2
    return out


class VMCastOneBufBencher:
    name = "vmcast_1buf"

    def __init__(self, cfg: Config, group, min_group: int = 2, debug: bool = False):
        self.cfg = cfg
        self.group = group
        self.num_sms = NVLS_MAX_BLOCKS
        self.device = torch.device("cuda", cfg.local_rank)
        self.min_group = min_group if cfg.ep_size >= min_group else 2
        self.debug = debug
        self._dbg_seen = set()
        self._built = False

    # -- one-time (collective) allocation --------------------------------------
    def build(self):
        cfg = self.cfg
        P, rank, cap = cfg.ep_size, cfg.rank, cfg.per_rank_cap
        H, K = cfg.hidden, cfg.topk
        self.epr = cfg.num_experts // P
        self.sizes = nested_sizes(P, self.min_group)
        gcap = P * cap                      # shared buffer rows = full global capacity
        self.gcap = gcap

        # Create every nested aligned group (all ranks, same order); keep this rank's group per size.
        self.pgs = {}
        for s in self.sizes:
            for start in range(0, P, s):
                members = list(range(start, start + s))
                pg = dist.new_group(members)
                if rank in members:
                    self.pgs[s] = pg
        for s in self.sizes:
            symm_mem.enable_symm_mem_for_group(self.pgs[s].group_name)

        # ONE shared buffer per tensor; rendezvous each over every nested group -> aliased VA per size.
        def alloc(numel, dtype):
            return symm_mem.empty(numel, dtype=dtype, device=self.device)

        self._h = alloc(gcap * H, torch.bfloat16)
        self._r = alloc(gcap * K, torch.int64)
        self._p = alloc(gcap * K, torch.float32)
        self._v = alloc(gcap * H, torch.bfloat16)
        self.h_local = self._h.view(gcap, H)
        self.r_local = self._r.view(gcap, K)
        self.v_local = self._v.view(gcap, H)
        self._v.normal_()                   # pre-fill rsv so the TIMED combine reduces real bytes

        mc_h, mc_r, mc_p, mc_v = {}, {}, {}, {}
        for s in self.sizes:                # order: per size, rendezvous all four (consistent on members)
            pg = self.pgs[s]
            hh = symm_mem.rendezvous(self._h, pg)
            hr = symm_mem.rendezvous(self._r, pg)
            hp = symm_mem.rendezvous(self._p, pg)
            hv = symm_mem.rendezvous(self._v, pg)
            mc_h[s], mc_r[s], mc_p[s], mc_v[s] = (int(hh.multicast_ptr), int(hr.multicast_ptr),
                                                  int(hp.multicast_ptr), int(hv.multicast_ptr))
            if s == P:                      # size-P group == global: its signal pads drive the barrier
                self.disp_sig = hh.signal_pad_ptrs_dev
                self.comb_sig = hv.signal_pad_ptrs_dev

        # size-indexed VA tables (gathered per layer by tok_size); the per-token VA arrays (filled in
        # _layout each layer) and the STATIC global-row slots (routing-independent -> computed once).
        dev = self.device
        self._mc_h_by = torch.tensor([mc_h[s] for s in self.sizes], dtype=torch.int64, device=dev)
        self._mc_r_by = torch.tensor([mc_r[s] for s in self.sizes], dtype=torch.int64, device=dev)
        self._mc_p_by = torch.tensor([mc_p[s] for s in self.sizes], dtype=torch.int64, device=dev)
        self._mc_v_by = torch.tensor([mc_v[s] for s in self.sizes], dtype=torch.int64, device=dev)
        self._sizes_t = torch.tensor(self.sizes, dtype=torch.int64, device=dev)
        self.tok_mc_h = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.tok_mc_r = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.tok_mc_p = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.tok_mc_v = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.tok_slot = (rank * cap + torch.arange(cap, device=dev)).to(torch.int32)  # STATIC global rows
        self.in_probs = torch.ones(cap, K, dtype=torch.float32, device=dev)           # identity weights
        # scratch for the fused layout kernel (one launch/layer). counts/intra are computed but UNUSED
        # here (slot is static) -- kept only to reuse the shared vmcast_compute_group kernel.
        nsz = len(self.sizes)
        self._counts = torch.zeros(nsz, dtype=torch.int32, device=dev)
        self._size_idx = torch.zeros(cap, dtype=torch.int32, device=dev)
        self._intra = torch.zeros(cap, dtype=torch.int32, device=dev)
        self._built = True

    # -- DEVICE-ONLY per-token layout: tok_size -> gather group VA. No collective, no scan, no offset.
    #    Runs INSIDE the timed decode_step every layer (honest). tok_slot is static (routing-independent).
    def _layout(self):
        n = self.n
        if n == 0:
            return
        # ONE Triton launch: per token -> tok_size (bit trick) -> gather its group's four multicast VAs.
        # Slotless: no atomic/counts/intra (the global row is static), no collective, no host sync.
        # This IS the honest per-layer layout.
        vmcast_compute_group_slotless(
            self.in_routing[:n],
            (self._mc_h_by, self._mc_r_by, self._mc_p_by, self._mc_v_by),
            self._size_idx[:n],
            self.tok_mc_h[:n], self.tok_mc_r[:n], self.tok_mc_p[:n], self.tok_mc_v[:n],
            self.epr, self.cfg.rank, self.min_group, len(self.sizes))
        self._last_size_idx = self._size_idx[:n].to(torch.int64)

    # -- per-batch: stage local inputs; bake nothing routing-dependent (layout is per-layer, in-graph) --
    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        dev = self.device
        self.n = hidden.shape[0]
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.to(torch.int64).contiguous()
        self.out = torch.zeros(self.n, self.cfg.hidden, dtype=torch.bfloat16, device=dev)
        self._layout()                      # also run once now (warmup/capture arrays + debug)
        if self.debug and self.cfg.rank == 0 and self.n > 0 and self.n not in self._dbg_seen:
            self._dbg_seen.add(self.n)
            hist = {int(s): int((self._sizes_t[self._last_size_idx] == s).sum().item()) for s in self.sizes}
            nd = torch.tensor([torch.unique(self.in_routing[t] // self.epr).numel()
                               for t in range(self.n)], device=dev, dtype=torch.float32)
            print(f"[vmcast_1buf dbg] n={self.n} sizes={self.sizes} tok_size_hist={hist} "
                  f"#dest_ranks/tok mean={float(nd.mean()):.2f} max={int(nd.max())}", flush=True)

    def dispatch(self):
        fused_vmcast_dispatch(
            self.in_hidden, self.in_routing, self.in_probs[:self.n],
            self.tok_mc_h, self.tok_mc_r, self.tok_mc_p, self.tok_slot,
            self.n, self.disp_sig, self.cfg.rank, self.cfg.ep_size, self.cfg.per_rank_cap,
            max_num_blocks=self.num_sms)

    def combine(self):
        fused_vmcast_combine(
            self.out, self.tok_mc_v, self.tok_slot, self.n, self.comb_sig,
            self.cfg.rank, self.cfg.ep_size, self.cfg.per_rank_cap, max_num_blocks=self.num_sms)

    def _mask_into_v(self):
        """Identity-expert mask (functional_roundtrip only, untimed): fill the shared rsv region from the
        gathered hidden iff the token routes to a LOCAL expert. Processes all rows; stale/out-of-group
        rows produce garbage that the group-scoped combine never sums."""
        rank, epr, H = self.cfg.rank, self.epr, self.cfg.hidden
        rv = self.r_local
        local = ((rv >= rank * epr) & (rv < (rank + 1) * epr)).any(dim=1)      # [gcap]
        self.v_local[:] = torch.where(local.view(self.gcap, 1), self.h_local,
                                      torch.zeros((), device=self.device, dtype=torch.bfloat16))

    def step(self):
        self._layout(); self.dispatch(); self.combine()

    def decode_step(self, num_layers: int):
        for _ in range(num_layers):
            self._layout(); self.dispatch(); self.combine()      # honest: layout re-run per layer

    def functional_roundtrip(self, hidden, topk_idx):
        self.setup_batch(hidden, topk_idx, None)
        self.dispatch()
        self._mask_into_v()
        self.combine()
        return self.out[:self.n]

    def validate(self):
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts
        out = []
        for B in (w * 2, 1):
            counts = [B // w + (1 if r < B % w else 0) for r in range(w)]
            n = counts[rank]
            gen = torch.Generator(device=dev).manual_seed(770077 + B + rank * 131)
            x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                 if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
            idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
                   if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
            got = self.functional_roundtrip(x, idx).to(torch.float32)
            torch.cuda.synchronize()
            if n > 0:
                epr = E // w
                mm = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                  device=dev, dtype=torch.float32).view(n, 1)
                ref = mm * x.to(torch.float32)
                ok = bool(torch.allclose(got, ref, rtol=0.03, atol=0.05))
                detail = f"n={n} max|got-m*x|={float((got - ref).abs().max()):.4f}"
            else:
                ok, detail = True, "n=0 (idle rank)"
            out.append((f"VMCAST_1BUF dispatch->combine B={B:<4d}", ok, detail))
        return out
