# Copyright (c) 2026. Variable-size multicast NVLS dispatch/combine bencher.
"""
A variant of the PURE NVLS multicast dispatch (bench_nvls.py) -- NOT the AGv-in/A2A-out
hierarchy. Instead of one flat size-P AllGather-V multicast to all EP ranks, pre-create
NESTED ALIGNED multicast groups (size 4 {0-3},{4-7},...; size 8 {0-7},...; ... up to {0..P-1})
and route each token through the SMALLEST group that spans {its source rank} U {its dest ranks}.
A token whose experts sit on nearby ranks pays a size-4/8 multicast instead of size-64.

Feasibility of the 31 overlapping multicast groups was confirmed by probe_nested_mcast.py.

Semantics (identical m*x to NVLS/DeepEP): the source rank MUST be in the token's group so the
per-group ReduceScatter-V can return the combined result to it. Each group runs the same
multimem AGv-V (dispatch) + mask + RSv-V (combine) as flat NVLS, just over its members and on
the subset of tokens bucketed to it.

FIRST PROTOTYPE: correctness-first. setup_batch does host-synced bucketing (.nonzero/.item);
the per-group AGv/RSv are the existing multimem kernels. Graph-capture + vectorized bucketing
come after the m*x gate passes.
"""
import torch
import torch.distributed as dist

from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import (
    fused_metadata_update, fused_metadata_update_dev, fused_group_metadata_update,
)
from nvls.torch_symm_triton.variable_collectives import (
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)
from nvls.torch_symm_triton.fused_vmcast import (
    fused_vmcast_dispatch, fused_vmcast_combine, vmcast_compute_group,
)
from common import Config, size_mb

NVLS_MAX_BLOCKS = 148


def nested_sizes(P, min_size):
    s, out = min_size, []
    while s <= P:
        out.append(s)
        s *= 2
    return out


class VMCastBencher:
    name = "vmcast"

    def __init__(self, cfg: Config, group, min_group: int = 4, debug: bool = False,
                 fused: bool = False, inline_layout: bool = False):
        self.cfg = cfg
        self.group = group
        self.num_sms = NVLS_MAX_BLOCKS
        self.device = torch.device("cuda", cfg.local_rank)
        self.min_group = min_group if cfg.ep_size >= min_group else 2
        self.debug = debug
        self.fused = fused           # True -> single fused per-token multicast kernel (1 pass, 1 barrier)
        # inline_layout: compute the per-token group+slot layout ON DEVICE, per layer, INSIDE the timed
        # decode_step (no host sync) -- the production-honest path (routing is produced per-layer by the
        # router inside the whole-model graph, so the layout cannot be hoisted to setup). Requires fused.
        self.inline_layout = inline_layout and fused
        if fused:
            self.name = "vmcast_fused_inl" if self.inline_layout else "vmcast_fused"
        self._dbg_seen = set()
        self._built = False

    # -- one-time (collective) allocation --------------------------------------
    def build(self):
        cfg = self.cfg
        P, rank = cfg.ep_size, cfg.rank
        K, H = cfg.topk, cfg.hidden
        self.epr = cfg.num_experts // P
        self.sizes = nested_sizes(P, self.min_group)

        def gbuf(mgr, key, shape, dtype):
            b = mgr.maybe_get_tensor(shape, dtype=dtype)
            if b["handle"] is None:
                raise RuntimeError(f"vmcast symm-mem init failed for '{key}'")
            return b

        # Create EVERY nested aligned group (all ranks call new_group in the same order);
        # keep + allocate buffers only for the groups THIS rank is a member of.
        self.my_groups = {}          # size -> dict(start, pg, gcap, agv_h/r/p, rsv, meta, step)
        for s in self.sizes:
            for start in range(0, P, s):
                members = list(range(start, start + s))
                pg = dist.new_group(members)
                if rank not in members:
                    continue
                gcap = s * cfg.per_rank_cap          # worst-case tokens in a size-s group
                key = f"vm_{s}_{start}"
                mgr_h = SymmetricMemoryManager.get_buffer(key + "_h", process_group=pg,
                                                          size_mb=size_mb([gcap, H], torch.bfloat16))
                mgr_r = SymmetricMemoryManager.get_buffer(key + "_r", process_group=pg,
                                                          size_mb=size_mb([gcap, K], torch.int64))
                mgr_p = SymmetricMemoryManager.get_buffer(key + "_p", process_group=pg,
                                                          size_mb=size_mb([gcap, K], torch.float32))
                mgr_v = SymmetricMemoryManager.get_buffer(key + "_v", process_group=pg,
                                                          size_mb=size_mb([gcap, H], torch.bfloat16))
                mgr_m = SymmetricMemoryManager.get_buffer(key + "_m", process_group=pg,
                                                          size_mb=size_mb([s], torch.int32))
                self.my_groups[s] = dict(
                    start=start, pg=pg, gcap=gcap,
                    agv_h=gbuf(mgr_h, key, [gcap, H], torch.bfloat16),
                    agv_r=gbuf(mgr_r, key, [gcap, K], torch.int64),
                    agv_p=gbuf(mgr_p, key, [gcap, K], torch.float32),
                    rsv=gbuf(mgr_v, key, [gcap, H], torch.bfloat16),
                    meta=gbuf(mgr_m, key, [s], torch.int32),
                    step=torch.zeros(3, dtype=torch.int32, device=self.device),
                    # stable per-group scratch (graph replay): RSv output + compacted AGv inputs.
                    _out=torch.empty(cfg.per_rank_cap, H, dtype=torch.bfloat16, device=self.device),
                )
                # Pre-fill rsv with valid data so the TIMED combine (raw RSv, no mask -- matches
                # NVLS's timing semantics) operates on real bytes.
                self.my_groups[s]["rsv"]["tensor"].normal_()

        if self.fused:
            # Fused per-token path: precompute each group's multicast VAs; allocate stable per-token
            # arrays (filled per batch in setup_batch) that map token -> its group's VAs + dest row.
            # The global (size-P) group's signal pads serve as the single cross-group barrier (a
            # size-P barrier is a correct superset of every nested subgroup's barrier).
            self.mc = {s: (g["agv_h"]["handle"].multicast_ptr, g["agv_r"]["handle"].multicast_ptr,
                           g["agv_p"]["handle"].multicast_ptr, g["rsv"]["handle"].multicast_ptr)
                       for s, g in self.my_groups.items()}
            cap = cfg.per_rank_cap
            self.tok_mc_h = torch.zeros(cap, dtype=torch.int64, device=self.device)
            self.tok_mc_r = torch.zeros(cap, dtype=torch.int64, device=self.device)
            self.tok_mc_p = torch.zeros(cap, dtype=torch.int64, device=self.device)
            self.tok_mc_v = torch.zeros(cap, dtype=torch.int64, device=self.device)
            self.tok_slot = torch.zeros(cap, dtype=torch.int32, device=self.device)
            self.in_probs = torch.ones(cap, K, dtype=torch.float32, device=self.device)  # identity weights
            self.disp_sig = self.my_groups[P]["agv_h"]["handle"].signal_pad_ptrs_dev
            self.comb_sig = self.my_groups[P]["rsv"]["handle"].signal_pad_ptrs_dev
            # Size-indexed lookup tables for the on-device layout (_compute_layout_device):
            # sizes ascending [min_group..P]; mc_by_size[i] = size sizes[i]'s group multicast VAs.
            self._sizes_t = torch.tensor(self.sizes, dtype=torch.int64, device=self.device)
            self._ar_sizes = torch.arange(len(self.sizes), device=self.device)
            self._mc_by_size_h = torch.tensor([self.mc[s][0] for s in self.sizes], dtype=torch.int64, device=self.device)
            self._mc_by_size_r = torch.tensor([self.mc[s][1] for s in self.sizes], dtype=torch.int64, device=self.device)
            self._mc_by_size_p = torch.tensor([self.mc[s][2] for s in self.sizes], dtype=torch.int64, device=self.device)
            self._mc_by_size_v = torch.tensor([self.mc[s][3] for s in self.sizes], dtype=torch.int64, device=self.device)
            self._tok_size = torch.zeros(cap, dtype=torch.int64, device=self.device)  # last layout's sizes (debug)
            # scratch for the fused layout kernel (per-layer device layout):
            nsz = len(self.sizes)
            self._counts = torch.zeros(nsz, dtype=torch.int32, device=self.device)
            self._size_idx = torch.zeros(cap, dtype=torch.int32, device=self.device)
            self._intra = torch.zeros(cap, dtype=torch.int32, device=self.device)
            self._offsets = torch.zeros(nsz, dtype=torch.int32, device=self.device)  # rank_token_offset/size
            self._gvalid = torch.zeros(nsz, dtype=torch.int32, device=self.device)   # total tokens/size (mask)
            # symm buffer for the SINGLE group-metadata collective ([P, NSZ] int32 over the global group).
            mgr_gm = SymmetricMemoryManager.get_buffer("vm_gmeta", process_group=self.my_groups[P]["pg"],
                                                       size_mb=size_mb([P * nsz], torch.int32))
            self._gmeta = mgr_gm.maybe_get_tensor([P * nsz], dtype=torch.int32)
            if self._gmeta["handle"] is None:
                raise RuntimeError("vmcast group-metadata symm-mem init failed")
        self._built = True

    # -- DEVICE-ONLY per-token layout (no host sync -> graph-capturable, runs per layer when inline) --
    def _compute_layout_device(self):
        """Fill tok_mc_*/tok_slot from in_routing entirely on device. This is what a real deployment
        must do: the router emits topk_idx per layer inside the model graph, so the group+slot layout
        can't be hoisted to setup -- it runs here, per layer, with no .item()/.nonzero().

        1. tok_size = smallest aligned pow2 block spanning {rank} U dest_ranks   (integer arithmetic)
        2. size_idx = index of tok_size in sizes                                 (searchsorted)
        3. counts   = per-size local token counts                               (scatter_add)
        4. intra    = exclusive count of earlier same-size tokens               (segmented cumsum)
        5. rank_token_offset per size via device-count metadata collective       (fused_metadata_update_dev)
        6. tok_slot = offset[size_idx] + intra ; tok_mc_* = mc_by_size[size_idx] (gathers)
        """
        n, rank, epr, nsz = self.n, self.cfg.rank, self.epr, len(self.sizes)
        # 1-4,6a. ONE Triton kernel: tok_size (bit arithmetic) + size_idx + atomic intra-position +
        #         per-size counts + gather tok_mc_* -- replaces ~15 tiny torch ops.
        self._counts.zero_()
        if n > 0:
            vmcast_compute_group(
                self.in_routing,
                (self._mc_by_size_h, self._mc_by_size_r, self._mc_by_size_p, self._mc_by_size_v),
                self._counts, self._size_idx, self._intra,
                self.tok_mc_h, self.tok_mc_r, self.tok_mc_p, self.tok_mc_v,
                epr, rank, self.min_group, nsz)
        # 5. ONE group-metadata collective -> rank_token_offset (+valid total) per size.
        fused_group_metadata_update(self._counts, self._gmeta["tensor"], self._gmeta["handle"],
                                    self._offsets, self._gvalid, self.min_group, nsz)
        # 6b. slot = rank_token_offset[size] + intra-rank position
        if n > 0:
            si = self._size_idx[:n]
            self.tok_slot[:n] = self._offsets[si.to(torch.int64)] + self._intra[:n]
            self._tok_size[:n] = self._sizes_t[si.to(torch.int64)]                            # for debug

    # -- per-batch: compute each token's group + bucket (host-synced, prototype) ----
    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        cfg, rank, P = self.cfg, self.cfg.rank, self.cfg.ep_size
        dev, K = self.device, self.cfg.topk
        self.n = hidden.shape[0]
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.to(torch.int64).contiguous()
        self.out = torch.zeros(self.n, cfg.hidden, dtype=torch.bfloat16, device=dev)

        if self.inline_layout:
            # Device-only layout: fill tok arrays now (for warmup/capture arrays + correctness + debug);
            # decode_step recomputes it per layer inside the timed region. Bake per-group _valid (untimed
            # host sync here, NOT in the timed path) so the correctness mask can bound its reads.
            self._compute_layout_device()
            for i, s in enumerate(self.sizes):
                self.my_groups[s]["_valid"] = int(self._gvalid[i].item())
            self._debug_hist(self._tok_size[:self.n])
            return

        # Per-token group SIZE = smallest nested block containing {rank} U dest_ranks(token).
        # dest ranks = unique(expert // epr).  block size = smallest s with lo//s == hi//s
        # where [lo,hi] = span of {rank} U dest_ranks.
        if self.n > 0:
            dest = (self.in_routing // self.epr)                  # [n,K] dest rank per expert
            lo = torch.minimum(dest.min(dim=1).values,
                               torch.full((self.n,), rank, device=dev))
            hi = torch.maximum(dest.max(dim=1).values,
                               torch.full((self.n,), rank, device=dev))
            tok_size = torch.full((self.n,), P, dtype=torch.int64, device=dev)
            for s in reversed(self.sizes):                        # largest->smallest, keep smallest fit
                fits = (lo // s) == (hi // s)
                tok_size = torch.where(fits, torch.full_like(tok_size, s), tok_size)
        else:
            tok_size = torch.empty(0, dtype=torch.int64, device=dev)

        # Bucket local tokens by size -> compact per group; publish per-group metadata.
        self._buckets = {}
        for s in self.sizes:
            g = self.my_groups[s]
            idx = (tok_size == s).nonzero(as_tuple=True)[0] if self.n > 0 else \
                torch.empty(0, dtype=torch.int64, device=dev)
            ns = int(idx.numel())
            g["agv_h"]["tensor"].view(g["gcap"], cfg.hidden)  # ensure view exists
            self._buckets[s] = dict(idx=idx, ns=ns)
            # fused_metadata_update over this group -> step = [valid, offset, ep_max]
            fused_metadata_update(local_tokens=ns, local_buf=g["meta"]["tensor"],
                                  symm_mem_hdl=g["meta"]["handle"], step_metadata=g["step"])
            # Bake the group total here (untimed host sync) so the TIMED combine has no .item().
            g["_valid"] = int(g["step"][0].item())
            # stage this rank's compacted inputs for the AGv
            H, Kk = cfg.hidden, K
            g["_in_h"] = self.in_hidden[idx] if ns > 0 else torch.empty(0, H, dtype=torch.bfloat16, device=dev)
            g["_in_r"] = self.in_routing[idx] if ns > 0 else torch.empty(0, Kk, dtype=torch.int64, device=dev)
            g["_in_p"] = torch.ones(ns, Kk, dtype=torch.float32, device=dev)
            g["_ns"] = ns

        if self.fused:
            # Fill the per-token arrays: token t (in group s at bucket position j) -> group s's
            # multicast VAs and dest row (rank_token_offset_s + j). Read directly from the local
            # input rows in the fused kernel (no compaction). Untimed host-side setup.
            for s in self.sizes:
                b = self._buckets[s]
                idx, ns = b["idx"], b["ns"]
                if ns == 0:
                    continue
                mc_h, mc_r, mc_p, mc_v = self.mc[s]
                self.tok_mc_h[idx] = mc_h
                self.tok_mc_r[idx] = mc_r
                self.tok_mc_p[idx] = mc_p
                self.tok_mc_v[idx] = mc_v
                self.tok_slot[idx] = (self.my_groups[s]["step"][1].to(torch.int32)
                                      + torch.arange(ns, device=dev, dtype=torch.int32))

        self._debug_hist(tok_size)

    # -- diagnostic: GLOBAL tok_size histogram + dest-rank stats (all ranks; rank 0's own window
    #    clamps to an aligned position, so per-rank-0 numbers hide the unaligned group-size inflation).
    #    All ranks participate in the reduces; rank 0 prints once per global token total. Untimed.
    def _debug_hist(self, tok_size):
        if not self.debug:
            return
        cfg, rank, dev = self.cfg, self.cfg.rank, self.device
        n, nsz = self.n, len(self.sizes)
        if n > 0:
            dest = self.in_routing // self.epr
            lh = torch.tensor([int((tok_size == s).sum()) for s in self.sizes], device=dev, dtype=torch.long)
            nd = torch.tensor([torch.unique(dest[t]).numel() for t in range(n)], device=dev, dtype=torch.long)
            agg = torch.tensor([n, int(nd.sum())], device=dev, dtype=torch.long)
            mx = torch.tensor([int(nd.max())], device=dev, dtype=torch.long)
        else:
            lh = torch.zeros(nsz, device=dev, dtype=torch.long)
            agg = torch.zeros(2, device=dev, dtype=torch.long)
            mx = torch.zeros(1, device=dev, dtype=torch.long)
        dist.all_reduce(lh, group=self.group)
        dist.all_reduce(agg, group=self.group)
        dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=self.group)
        key = int(agg[0].item())
        if rank == 0 and key not in self._dbg_seen:
            self._dbg_seen.add(key)
            hist = {int(s): int(lh[i].item()) for i, s in enumerate(self.sizes)}
            loc = f"mix={cfg.routing_mix}" if cfg.routing_mix else f"block={cfg.routing_block}"
            if cfg.routing_unaligned:
                loc += " UNALIGNED"
            print(f"[vmcast dbg GLOBAL] Btot={key} topk={cfg.topk} {loc} sizes={self.sizes} "
                  f"tok_size_hist={hist} #dest_ranks/tok mean="
                  f"{float(agg[1].item())/max(1, key):.2f} max={int(mx[0].item())}", flush=True)

    # -- dispatch: per-group multimem AGv-V (multicast only within the group) -------
    def dispatch(self):
        if self.fused:
            # ONE fused kernel: every token multicast-stores to its own group's buffer, then a
            # single global barrier. No per-size passes (the whole point vs the unfused path).
            fused_vmcast_dispatch(
                self.in_hidden, self.in_routing, self.in_probs[:self.n] if self.n else self.in_probs,
                self.tok_mc_h, self.tok_mc_r, self.tok_mc_p, self.tok_slot,
                self.n, self.disp_sig, self.cfg.rank, self.cfg.ep_size, self.cfg.per_rank_cap,
                max_num_blocks=self.num_sms)
            return
        for s in self.sizes:
            g = self.my_groups[s]
            if g["_valid"] == 0:       # no member routed a token to this group -> skip its whole pass.
                continue               # `_valid` is group-wide (all members agree) -> no barrier mismatch.
            multimem_all_gatherv_3tensor(
                g["agv_h"]["tensor"], g["agv_r"]["tensor"], g["agv_p"]["tensor"],
                g["_in_h"], g["_in_r"], g["_in_p"],
                g["agv_h"]["handle"], g["agv_r"]["handle"], g["agv_p"]["handle"],
                rank_token_offset=g["step"][1:2], ep_max_tokens=g["step"][2:3],
                per_rank_max_tokens=self.cfg.per_rank_cap, max_num_blocks=self.num_sms)

    def _mask_into_rsv(self):
        """Identity-expert mask (functional_roundtrip only, NOT timed -- matches how NVLS keeps
        the mask out of its timed step): fill each group's rsv from its gathered hidden iff the
        token routes to a local expert."""
        cfg, rank, H = self.cfg, self.cfg.rank, self.cfg.hidden
        for s in self.sizes:
            g = self.my_groups[s]
            valid = g["_valid"]
            if valid == 0:
                continue
            gh = g["agv_h"]["tensor"].view(g["gcap"], H)
            gr = g["agv_r"]["tensor"].view(g["gcap"], cfg.topk)
            local = ((gr[:valid] >= rank * self.epr) & (gr[:valid] < (rank + 1) * self.epr)).any(dim=1)
            g["rsv"]["tensor"].view(g["gcap"], H)[:valid] = torch.where(
                local.view(valid, 1), gh[:valid], torch.zeros((), device=self.device))

    # -- combine (TIMED): per-group RSv-V (on pre-filled rsv, NO mask) + scatter back to source.
    #    No host sync: `_valid`/`_ns`/idx baked in setup -> whole decode_step is graph-capturable.
    def combine(self):
        if self.fused:
            # ONE fused kernel: global barrier, then every token multimem.ld_reduce-loads its row
            # from its own group's RSv buffer straight into out[t] (no per-group scatter).
            fused_vmcast_combine(
                self.out if self.n else self.out.new_zeros(1, self.cfg.hidden),
                self.tok_mc_v, self.tok_slot, self.n, self.comb_sig,
                self.cfg.rank, self.cfg.ep_size, self.cfg.per_rank_cap, max_num_blocks=self.num_sms)
            return
        H = self.cfg.hidden
        for s in self.sizes:
            g = self.my_groups[s]
            if g["_valid"] == 0:       # skip empty groups (matches dispatch) -> no barrier mismatch.
                continue
            out_g = g["_out"][:g["_ns"]]
            multimem_reduce_scatter_v(
                out_g, g["rsv"]["tensor"], g["rsv"]["handle"],
                rank_token_offset=g["step"][1:2], ep_max_tokens=g["step"][2:3],
                per_rank_max_tokens=self.cfg.per_rank_cap, max_num_blocks=self.num_sms)
            if g["_ns"] > 0:           # scatter this group's results back to original token positions
                self.out[self._buckets[s]["idx"]] = out_g

    def step(self):
        if self.inline_layout:
            self._compute_layout_device()
        self.dispatch(); self.combine()

    def decode_step(self, num_layers: int):
        # inline_layout: recompute the group+slot layout ON DEVICE each layer, INSIDE the timed region
        # -- the production-honest cost (per-layer router output can't be hoisted out of the model graph).
        if self.inline_layout:
            for _ in range(num_layers):
                self._compute_layout_device(); self.dispatch(); self.combine()
        else:
            for _ in range(num_layers):
                self.dispatch(); self.combine()

    def functional_roundtrip(self, hidden, topk_idx):
        self.setup_batch(hidden, topk_idx, None)
        self.dispatch()
        self._mask_into_rsv()      # identity expert (untimed here; excluded from step()/decode_step)
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
            out.append((f"VMCAST dispatch->combine B={B:<4d}", ok, detail))
        return out
