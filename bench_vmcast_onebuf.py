# Copyright (c) 2026. One-buffer aliased-multicast variable-multicast dispatch/combine bencher.
"""
The "one-buffer" vmcast variant (the honest, NVLS-structured design).

Instead of a separate padded buffer per nested multicast group, allocate ONE shared buffer per tensor
of shape [P*cap, *] and rendezvous it over EVERY nested aligned group (size-2 {0,1}.., size-4 {0-3}..,
... size-P {0..P-1}) -- torch symm-mem gives a DISTINCT multicast VA per group that ALIASES the one
allocation (confirmed by probe_aliased_mcast.py). Then:

  * Token (source rank r, local index i) lives at the COMPACTED global row  offset[r] + i  -- exactly
    where NVLS's dense AllGather puts it. offset[r] = rank_token_offset is the prefix sum of the
    routing-INDEPENDENT source counts, from the SAME once-per-step fused_metadata_update collective NVLS
    runs (NVLS parity; timed inside decode_step). The row is thus routing-independent -- no per-LAYER
    collective -- but does cost one count all-gather per step, like NVLS.
  * Per token we pick the smallest aligned group spanning {r} U dest_ranks(token) (tok_size) and multicast
    through THAT group's aliased VA: the store reaches only that group's ranks, but lands at the global
    row in the one shared buffer. Ranks outside the group keep a stale row there -- harmless, because
    they don't host the token's experts AND the combine reduces only over the token's group VA, so a
    stale out-of-group slot is never in the sum.

Honest per-layer layout = ONE elementwise pass computing tok_size_idx (each token's group index; the
dispatch/combine kernels gather the actual VAs from the per-size menus by it) -- device-only,
graph-capturable, NO per-layer collective. Plus a once-per-step count exchange for the compacted offset.
Both run INSIDE the timed decode_step -- structurally the same shape as NVLS. Reuses the fused per-token
multicast kernels (nvls/torch_symm_triton/fused_vmcast.py) unchanged.
"""
import torch
import torch.distributed as dist

import torch.distributed._symmetric_memory as symm_mem

from nvls.torch_symm_triton.fused_vmcast import (
    fused_vmcast_dispatch, fused_vmcast_combine, vmcast_compute_size,
)
from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import fused_metadata_update
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

    # -- one-time (collective) allocation.  This is the NVLS build() plus ONLY the pieces the multiple-VA
    #    per-token-selection approach needs; every addition is labelled [vmcast].
    def build(self):
        cfg = self.cfg
        gmax = cfg.global_cap                 # NVLS: buffer rows = per_rank_cap * ep_size
        K, H = cfg.topk, cfg.hidden           # NVLS
        dev = self.device

        # [vmcast] size ladder + dest-rank divisor -- needed to pick a token's sub-group (tok_size).
        P, rank = cfg.ep_size, cfg.rank
        self.epr = cfg.num_experts // P                    # dest rank of an expert = expert // epr
        self.sizes = nested_sizes(P, self.min_group)       # multicast-group sizes [2,4,..,P]
        self.gcap = gmax

        # [vmcast] nested aligned sub-groups, so a token can multicast to a SUBSET of ranks (not all P).
        #          Enable ALL before allocating, so ONE buffer can be rendezvoused (aliased) over each.
        #          size-P reuses NVLS's WORLD group; new_group is collective (every rank makes every one).
        self.pgs = {P: self.group}
        for s in self.sizes:
            for start in range(0, P, s):
                pg = dist.new_group(list(range(start, start + s)))
                if s != P and start <= rank < start + s:
                    self.pgs[s] = pg
        for pg in self.pgs.values():        # enable ALL (incl WORLD) BEFORE allocating -> aliasing works
            symm_mem.enable_symm_mem_for_group(pg.group_name)

        # NVLS: four [gmax,*] transport buffers -- agv_h (hidden) / agv_r (routing) / agv_p (probs) / rsv
        # (combine). Our buf() uses symm_mem.empty (not get_buffer) ONLY so the SAME buffer can be
        # rendezvoused over many sub-groups; it returns (flat alloc for rendezvous, shaped view for reads).
        def buf(shape, dtype):
            flat = symm_mem.empty(shape[0] * shape[1], dtype=dtype, device=dev)
            return flat, flat.view(*shape)
        h, self.agv_h = buf((gmax, H), torch.bfloat16)
        r, self.agv_r = buf((gmax, K), torch.int64)
        p, self.agv_p = buf((gmax, K), torch.float32)
        v, self.rsv = buf((gmax, H), torch.bfloat16)
        self.meta = SymmetricMemoryManager.get_buffer(
            "vm_meta", process_group=self.group, size_mb=1).maybe_get_tensor([P], torch.int32)
        if self.meta["handle"] is None:
            raise RuntimeError("vmcast_1buf metadata symm-mem init failed")
        self.step_metadata = torch.zeros(3, dtype=torch.int32, device=dev)   # [valid, offset, ep_max]
        self.rsv.normal_()                    # NVLS: pre-fill rsv so the timed combine reduces real bytes

        # [vmcast] one multicast VA per (buffer, sub-group), ALL aliasing the same allocation.
        def va_table(t):
            return torch.tensor([int(symm_mem.rendezvous(t, self.pgs[s]).multicast_ptr)
                                 for s in self.sizes], dtype=torch.int64, device=dev)
        self.mc_ptrs_agv_h, self.mc_ptrs_agv_r, self.mc_ptrs_agv_p, self.mc_ptrs_rsv = (
            va_table(h), va_table(r), va_table(p), va_table(v))
        self._sizes_t = torch.tensor(self.sizes, dtype=torch.int64, device=dev)   # debug histogram only

        # [vmcast] per-token selection: the ONLY per-token array is tok_size_idx (which group each token
        #          picked). The row is offset + t, computed IN-KERNEL from the once/step scalar offset
        #          (step_metadata[1], exactly like NVLS's AGv) -- no per-token row array. in_probs = id weights.
        cap = cfg.per_rank_cap
        self.tok_size_idx = torch.zeros(cap, dtype=torch.int32, device=dev)
        self.in_probs = torch.ones(cap, K, dtype=torch.float32, device=dev)

        # [vmcast] the single global barrier uses WORLD signal pads (from any size-P rendezvous handle).
        self.disp_sig = symm_mem.rendezvous(h, self.group).signal_pad_ptrs_dev
        self.comb_sig = symm_mem.rendezvous(v, self.group).signal_pad_ptrs_dev
        self._built = True

    # -- DEVICE-ONLY per-layer layout: compute tok_size_idx (each token's group). No per-layer collective,
    #    no scan, no VA gather. Runs INSIDE the timed decode_step every layer (honest). The row (offset + t)
    #    is computed IN-KERNEL from the once/step offset, not here (it is routing-independent).
    def _layout(self):
        n = self.n
        if n == 0:
            return
        # ONE Triton launch: per token -> tok_size_idx (bit trick on the dest-rank span). No VA gather
        # (dispatch/combine gather from the menus by this index), no atomic/counts, no collective, no
        # host sync. This IS the honest per-layer layout.
        vmcast_compute_size(self.in_routing[:n], self.tok_size_idx[:n],
                            self.epr, self.cfg.rank, self.min_group, len(self.sizes))
        self._last_size_idx = self.tok_size_idx[:n].to(torch.int64)

    # -- ONCE-PER-STEP token-count exchange (NVLS parity, timed inside decode_step): fused_metadata_update
    #    -> rank_token_offset (routing-independent source-count prefix sum); dispatch/combine use it for
    #    the compacted row = offset + t in-kernel (no per-token row array).
    def metadata(self):
        # Populates step_metadata = [valid, rank_token_offset, ep_max]. The dispatch/combine kernels read
        # step_metadata[1] (rank_token_offset) and compute each token's row = offset + t inline (NVLS-style).
        fused_metadata_update(local_tokens=self.n, local_buf=self.meta["tensor"],
                              symm_mem_hdl=self.meta["handle"], step_metadata=self.step_metadata)

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
            self.mc_ptrs_agv_h, self.mc_ptrs_agv_r, self.mc_ptrs_agv_p,
            self.tok_size_idx, self.step_metadata[1:2],
            self.n, self.disp_sig, self.cfg.rank, self.cfg.ep_size, self.cfg.per_rank_cap,
            max_num_blocks=self.num_sms)

    def combine(self):
        fused_vmcast_combine(
            self.out, self.mc_ptrs_rsv, self.tok_size_idx, self.step_metadata[1:2], self.n, self.comb_sig,
            self.cfg.rank, self.cfg.ep_size, self.cfg.per_rank_cap, max_num_blocks=self.num_sms)

    def _mask_into_v(self):
        """Identity-expert mask (functional_roundtrip only, untimed): fill the shared rsv region from the
        gathered hidden iff the token routes to a LOCAL expert. Processes all rows; stale/out-of-group
        rows produce garbage that the group-scoped combine never sums."""
        rank, epr = self.cfg.rank, self.epr
        rv = self.agv_r
        local = ((rv >= rank * epr) & (rv < (rank + 1) * epr)).any(dim=1)      # [gcap]
        self.rsv[:] = torch.where(local.view(self.gcap, 1), self.agv_h,
                                  torch.zeros((), device=self.device, dtype=torch.bfloat16))

    def step(self):
        self.metadata(); self._layout(); self.dispatch(); self.combine()

    def decode_step(self, num_layers: int):
        self.metadata()                                          # once/step: rank_token_offset (in-kernel row)
        for _ in range(num_layers):
            self._layout(); self.dispatch(); self.combine()      # honest: layout re-run per layer

    def functional_roundtrip(self, hidden, topk_idx):
        self.setup_batch(hidden, topk_idx, None)
        self.metadata()
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
