# Copyright (c) 2026. Hierarchical (AGv-inner / A2A-outer) MoE dispatch/combine bencher.
"""
K2: the full hierarchical dispatcher, un-pipelined (sequential hops), composing the
hardware-validated building blocks:

  dispatch = outer directed-P2P scatter (column, directed_p2p.py)
             -> row fused_metadata_update + multimem AGv-V (nvls/)  [inner]
             -> mask to local experts (identity expert)
  combine  = row multimem RSv-V (nvls/)  [inner]
             -> outer directed-P2P gather (column) -> local fp32 reduce over G groups

Correctness gate: functional_roundtrip(x, idx) == (#distinct dest ranks) * x == the same
`m*x` every other bencher yields, checked element-wise and cross-impl in run.py.

Group-major placement + the two-hop routing decomposition live in hier_common.py. This
K2 path computes the routing decomposition / send plans in torch (eager); K3 fuses them
into device kernels and pipelines the two hops.
"""

import torch

from common import Config, size_mb, time_region
from hier_common import (
    HierGrid, HierPlacement, make_groups, dest_group_mask, device_send_plan, local_mask,
)
from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.metadata import fused_metadata_update
from nvls.torch_symm_triton.variable_collectives import (
    multimem_all_gatherv_3tensor, multimem_reduce_scatter_v,
)
from nvls.torch_symm_triton.directed_p2p import (
    directed_a2a_scatter, directed_a2a_gather, fused_dispatch, masked_copy,
)

NVLS_MAX_BLOCKS = 148


class HierBencher:
    name = "hier"

    def __init__(self, cfg: Config, group, g: int):
        self.cfg = cfg
        self.group = group
        self.g = g
        self.num_sms = NVLS_MAX_BLOCKS
        self.device = torch.device("cuda", cfg.local_rank)
        self._built = False
        self.fused = False        # if True, step()/roundtrip use the fused barrier-free dispatch
        self.fused_max_n = 48     # ...but only when max per-rank tokens <= this (fused's per-token
                                  # overhead dominates at large B; bulk staged dispatch wins there)

    # -- one-time (collective) allocation --------------------------------------
    def build(self):
        cfg = self.cfg
        self.grid = HierGrid(cfg.ep_size, self.g, cfg.rank)
        self.placement = HierPlacement(cfg.num_experts, cfg.ep_size, self.g)
        self.row_group, self.col_group = make_groups(self.grid)
        G, g = self.grid.G, self.g
        H, K = cfg.hidden, cfg.topk
        cap = cfg.per_rank_cap
        self.cap = cap
        self.col_cap = G * cap          # max tokens landing at one (group,pos) in a column
        self.row_cap = g * self.col_cap  # AGv global capacity over the g positions of a group
        self.epr = cfg.num_experts // cfg.ep_size

        def cols(key, specs):
            b = SymmetricMemoryManager.get_buffer(
                key, process_group=self.col_group,
                size_mb=sum(size_mb([n], d) for n, d in specs) + 4).maybe_get_tensors(specs)
            assert b["handle"] is not None, f"col symm alloc failed for {key}"
            return b

        def rows(key, specs):
            b = SymmetricMemoryManager.get_buffer(
                key, process_group=self.row_group,
                size_mb=sum(size_mb([n], d) for n, d in specs) + 4).maybe_get_tensors(specs)
            assert b["handle"] is not None, f"row symm alloc failed for {key}"
            return b

        # Column dispatch-recv (this rank as a destination): hidden/routing/srctok + a
        # per-landed-row flag region (int32) used by the fused barrier-free dispatch to
        # hand off each token from the outer scatter (producer) to the inner AGv (consumer).
        d = cols("hier_disp", [(self.col_cap * H, torch.bfloat16),
                               (self.col_cap * K, torch.int64),
                               (self.col_cap, torch.int32),
                               (self.col_cap, torch.int32)])
        self.disp_hdl = d["handle"]
        (rh, self.dh_off), (rr, self.dr_off), (rs, self.ds_off), (rf, self.df_off) = d["tensors"]
        self.dh = rh.view(torch.bfloat16).view(self.col_cap, H)
        self.dr = rr.view(torch.int64).view(self.col_cap, K)
        self.ds = rs.view(torch.int32).view(self.col_cap)
        self.dflag = rf.view(torch.int32).view(self.col_cap)
        self.dflag.zero_()                       # flags self-reset (CAS 1->0) after each layer
        self.dbg = torch.full((self.col_cap,), -99, dtype=torch.int32, device=self.device)

        # Row AGv output (gathered group tokens): hidden/routing/probs.
        a = rows("hier_agv", [(self.row_cap * H, torch.bfloat16),
                              (self.row_cap * K, torch.int64),
                              (self.row_cap * K, torch.float32)])
        self.agv_hdl = a["handle"]
        (ah, self.ah_off), (ar, self.ar_off), (ap, self.ap_off) = a["tensors"]
        self.ah = ah.view(torch.bfloat16).view(self.row_cap, H)
        self.ar = ar.view(torch.int64).view(self.row_cap, K)
        self.ap = ap.view(torch.float32).view(self.row_cap, K)

        # Row RSv buffer.
        rv = rows("hier_rsv", [(self.row_cap * H, torch.bfloat16)])
        self.rsv_hdl = rv["handle"]
        (rvt, _), = rv["tensors"]
        self.rsv = rvt.view(torch.bfloat16).view(self.row_cap, H)

        # Row metadata buffer + step scalars.
        mt = rows("hier_meta", [(g, torch.int32)])
        self.meta_hdl = mt["handle"]
        (mtt, _), = mt["tensors"]
        self.meta = mtt.view(torch.int32).view(g)
        self.step_metadata = torch.zeros(3, dtype=torch.int32, device=self.device)

        # Column combine-recv (this rank as a source): partials from G dest groups, per token slot.
        c = cols("hier_comb", [(G * cap * H, torch.bfloat16)])
        self.comb_hdl = c["handle"]
        (ct, self.comb_off), = c["tensors"]
        self.comb = ct.view(torch.bfloat16).view(G, cap, H)

        # Local AGv probs input (unweighted) and combine output.
        self.probs_in = torch.ones(self.col_cap, K, dtype=torch.float32, device=self.device)
        self.out = torch.empty(cap, H, dtype=torch.float32, device=self.device)
        self._built = True

    # -- per-batch: compute ALL routing-dependent plans + metadata ONCE (untimed).
    #    The per-layer step() below is then a pure fixed-shape kernel sequence with NO
    #    host sync / collective, so the whole num_layers decode_step is CUDA-graph capturable.
    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        cfg, G, cap, dev = self.cfg, self.grid.G, self.cap, self.device
        my_group = self.grid.group_id
        self.in_hidden = hidden.contiguous()
        self.in_routing = topk_idx.to(torch.int64).contiguous()
        self.n = hidden.shape[0]
        self.srctok = torch.arange(self.n, dtype=torch.int32, device=dev)
        # Max per-rank token count (consistent across ranks) -> pick fused vs bulk dispatch.
        # Done once here (untimed); the branch resolves at graph-capture time, so it is stable.
        nmax = torch.tensor([self.n], dtype=torch.int64, device=dev)
        torch.distributed.all_reduce(nmax, op=torch.distributed.ReduceOp.MAX, group=self.group)
        self._max_n = int(nmax.item())

        # Routing decomposition + column count-matrix exchange (device). The only host syncs
        # (.item for fixed slice sizes) live here, OUTSIDE the timed/graph decode_step.
        mask = dest_group_mask(self.in_routing, self.placement)
        my_M_row = mask.sum(dim=0).to(torch.int64)
        M = torch.empty(G, G, dtype=torch.int64, device=dev)
        torch.distributed.all_gather_into_tensor(M.view(-1), my_M_row, group=self.col_group)
        self.d_send = device_send_plan(mask, M, my_group)                # (dest, tok, row) [n*G]
        R = int(M[:, my_group].sum().item())
        self._R = R

        # Once-per-step row metadata (routing-fixed across all layers) -> step_metadata.
        fused_metadata_update(local_tokens=R, local_buf=self.meta,
                              symm_mem_hdl=self.meta_hdl, step_metadata=self.step_metadata)
        self._valid = int(self.step_metadata[0].item())                  # group total (host, in setup)

        # Combine plan: source group per landed row (return slot uses ds, read at runtime).
        boundaries = torch.cumsum(M[:, my_group], dim=0)
        self.c_send_dest = torch.searchsorted(
            boundaries, torch.arange(R, device=dev), right=True).to(torch.int32)

        # Fixed-size intermediates (stable addresses for graph replay). comb is zeroed ONCE:
        # each layer's gather overwrites only the visited [d,t] slots (same set every layer),
        # so unvisited slots stay zero without a per-layer memset.
        self.out_inner = torch.empty(R, cfg.hidden, dtype=torch.bfloat16, device=dev)
        self.comb.zero_()
        self.dflag.zero_()                  # ensure fused-dispatch flags start clean this batch
        self.rsv.zero_()                    # masked_copy only writes OWNED tokens; unowned stay 0

    # -- the two hops: pure fixed-shape kernels, no host sync (graph-capturable) ----
    # -- phase sub-ops (split so the pipeline can put OUTER on one stream, INNER on another) --
    def _scatter(self, mb=None):                                        # OUTER: token -> dest groups
        d_dest, d_tok, d_row = self.d_send
        directed_a2a_scatter(
            self.disp_hdl.buffer_ptrs_dev, self.disp_hdl.signal_pad_ptrs_dev,
            self.in_hidden, self.in_routing, self.srctok,
            d_dest, d_tok, d_row, self.dh_off, self.dr_off, self.ds_off,
            my_group=self.grid.group_id, gcol=self.grid.G, hidden_size=self.cfg.hidden,
            topk=self.cfg.topk, max_blocks=(mb or self.num_sms))

    def _inner_agv(self, mb=None):                                      # INNER: gather within group
        multimem_all_gatherv_3tensor(
            self.ah, self.ar, self.ap, self.dh[:self._R], self.dr[:self._R], self.probs_in[:self._R],
            self.agv_hdl, self.agv_hdl, self.agv_hdl,
            rank_token_offset=self.step_metadata[1:2], ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.col_cap, output_byte_offset_0=self.ah_off,
            output_byte_offset_1=self.ar_off, output_byte_offset_2=self.ap_off,
            max_num_blocks=(mb or self.num_sms))

    def _mask_rsv(self, mb=None):                                       # INNER: identity + reduce
        valid = self._valid
        lo = self.cfg.rank * self.epr
        masked_copy(self.ah, self.ar, self.rsv, valid, lo, lo + self.epr,
                    self.cfg.topk, self.cfg.hidden, max_blocks=(mb or self.num_sms))
        multimem_reduce_scatter_v(
            self.out_inner, self.rsv, self.rsv_hdl,
            rank_token_offset=self.step_metadata[1:2], ep_max_tokens=self.step_metadata[2:3],
            per_rank_max_tokens=self.col_cap, max_num_blocks=(mb or self.num_sms))

    def _gather_reduce(self, mb=None):                                  # OUTER: return partials + reduce
        send_row = self.grid.group_id * self.cap + self.ds[:self._R]    # ds is int32; small, no overflow
        directed_a2a_gather(
            self.comb_hdl.buffer_ptrs_dev, self.comb_hdl.signal_pad_ptrs_dev,
            self.out_inner, self.c_send_dest, send_row, self.comb_off,
            my_group=self.grid.group_id, gcol=self.grid.G, hidden_size=self.cfg.hidden, max_blocks=(mb or self.num_sms))
        # Reduce over the G dest-group slots for THIS rank's n tokens only (comb is padded to
        # [G, cap, H] for worst-case B; reducing the full cap every layer is pure waste at small B).
        self.out[:self.n] = self.comb[:, :self.n, :].to(torch.float32).sum(dim=0)

    def _dispatch(self):
        self._scatter(); self._inner_agv()

    def _dispatch_fused(self, scatter_ctas=96, no_flags=False, skip_wait=False, wait_iters=50000):
        """K3 increment 1: barrier-free fused outer-scatter + inner-AGv (hidden+routing)
        in one launch, per-row release/acquire flags instead of the scatter column barrier
        + AGv row entry. Gathers ah/ar (probs unused on the identity-expert m*x path). The
        single retained row barrier lives at the end of the fused kernel."""
        d_dest, d_tok, d_row = self.d_send
        fused_dispatch(
            self.disp_hdl.buffer_ptrs_dev,
            self.in_hidden, self.in_routing, self.srctok,
            d_dest, d_tok, d_row, self.dh_off, self.dr_off, self.ds_off, self.df_off,
            self.dh, self.dr, self.dflag,
            self.agv_hdl.multicast_ptr, self.agv_hdl.signal_pad_ptrs_dev,
            self.step_metadata[1:2], self._R, self.ah_off, self.ar_off, self.dbg,
            my_group=self.grid.group_id, gcol=self.grid.G,
            row_rank=self.agv_hdl.rank, row_world=self.agv_hdl.world_size,
            hidden_size=self.cfg.hidden, topk=self.cfg.topk,
            scatter_ctas=scatter_ctas, max_blocks=self.num_sms,
            no_flags=no_flags, skip_wait=skip_wait, wait_iters=wait_iters)

    def _combine(self):
        self._mask_rsv(); self._gather_reduce()

    # -- correctness -----------------------------------------------------------
    def _use_fused(self):
        # Fused per-token dispatch only when it wins (small-mid B); bulk staged above the
        # threshold, where the fused kernel's per-token spin-wait overhead dominates.
        return self.fused and (self._max_n <= self.fused_max_n)

    def functional_roundtrip(self, hidden, topk_idx):
        self.setup_batch(hidden, topk_idx, None)
        (self._dispatch_fused if self._use_fused() else self._dispatch)()
        self._combine()
        return self.out[:self.n]

    def validate(self):
        world, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E, epr = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts, self.epr
        out = []
        for B in (world * 2, 1):
            counts = [B // world + (1 if r < B % world else 0) for r in range(world)]
            n = counts[rank]
            gen = torch.Generator(device=dev).manual_seed(880701 + B + rank * 151)
            x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                 if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
            idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
                   if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
            got = self.functional_roundtrip(x, idx).to(torch.float32)
            torch.cuda.synchronize()
            if n > 0:
                mm = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                  device=dev, dtype=torch.float32).view(n, 1)
                ref = mm * x.to(torch.float32)
                ok = bool(torch.allclose(got, ref, rtol=0.03, atol=0.05))
                detail = (f"n={n} max|got-m*x|={float((got - ref).abs().max()):.4f} "
                          f"m={mm.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
            else:
                ok, detail = True, "n=0 (idle rank)"
            out.append((f"HIER dispatch->combine B={B:<4d}", ok, detail))
        return out

    # -- timed unit: one full decode step = num_layers paired dispatch->combine.
    #    setup_batch precomputed the plans/metadata, so this is pure fixed-shape kernels
    #    (no host sync) and CUDA-graph capturable. K3b will pipeline the two hops.
    def step(self):
        (self._dispatch_fused if self._use_fused() else self._dispatch)()
        self._combine()

    def decode_step(self, num_layers: int):
        for _ in range(num_layers):
            self.step()

    def set_num_sms(self, num_sms: int):
        pass

    # -- K3b go/no-go: comm-comm overlap probe --------------------------------
    def overlap_probe(self, reps=50, sm_each=None):
        """Do the OUTER directed-P2P scatter (column) and the INNER multicast AGv (row)
        overlap when run concurrently on two streams, each capped to `sm_each` blocks so
        they can co-reside? Returns per-op us for outer-alone / inner-alone / both-concurrent
        and the overlap fraction (1=perfect: both==max; 0=none: both==sum). Needs setup_batch."""
        import torch.distributed as dist
        dev, R = self.device, self._R
        d_dest, d_tok, d_row = self.d_send
        mb = sm_each if sm_each is not None else max(2, self.num_sms // 2)

        def outer():
            directed_a2a_scatter(
                self.disp_hdl.buffer_ptrs_dev, self.disp_hdl.signal_pad_ptrs_dev,
                self.in_hidden, self.in_routing, self.srctok, d_dest, d_tok, d_row,
                self.dh_off, self.dr_off, self.ds_off, my_group=self.grid.group_id,
                gcol=self.grid.G, hidden_size=self.cfg.hidden, topk=self.cfg.topk, max_blocks=mb)

        def inner():
            multimem_all_gatherv_3tensor(
                self.ah, self.ar, self.ap, self.dh[:R], self.dr[:R], self.probs_in[:R],
                self.agv_hdl, self.agv_hdl, self.agv_hdl,
                rank_token_offset=self.step_metadata[1:2], ep_max_tokens=self.step_metadata[2:3],
                per_rank_max_tokens=self.col_cap, output_byte_offset_0=self.ah_off,
                output_byte_offset_1=self.ar_off, output_byte_offset_2=self.ap_off, max_num_blocks=mb)

        def timed(body):
            for _ in range(5):
                body()
            torch.cuda.synchronize(); dist.barrier(self.group)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                body()
            e.record(); torch.cuda.synchronize()
            t = torch.tensor([s.elapsed_time(e) / reps * 1000.0], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.group)   # critical path across ranks
            return float(t.item())

        t_o = timed(outer)
        t_i = timed(inner)

        # Concurrent: issue ALL `reps` outers on sA and ALL `reps` inners on sB with a SINGLE
        # join at the end (NO per-iteration fork/join) — the two streams run freely, so we
        # measure genuine overlap rather than per-step stream-sync bubbles.
        sA, sB, cur = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.current_stream()

        def concurrent():
            sA.wait_stream(cur); sB.wait_stream(cur)
            with torch.cuda.stream(sA):
                for _ in range(reps):
                    outer()
            with torch.cuda.stream(sB):
                for _ in range(reps):
                    inner()
            cur.wait_stream(sA); cur.wait_stream(sB)

        concurrent()                                   # warmup
        torch.cuda.synchronize(); dist.barrier(self.group)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        concurrent()
        e.record(); torch.cuda.synchronize()
        t_b = torch.tensor([s.elapsed_time(e) / reps * 1000.0], device=dev)
        dist.all_reduce(t_b, op=dist.ReduceOp.MAX, group=self.group)
        t_b = float(t_b.item())

        denom = min(t_o, t_i)
        overlap = (t_o + t_i - t_b) / denom if denom > 0 else 0.0
        return {"sms_each": mb, "outer_us": t_o, "inner_us": t_i, "both_us": t_b,
                "sum_us": t_o + t_i, "max_us": max(t_o, t_i), "overlap": overlap}

    def pipeline_bound(self, num_layers=20, reps=3, sm_each=None):
        """Best-case pipelined per-layer latency (us): all OUTER ops (scatter+gather) on stream P
        concurrent with all INNER ops (AGv+RSv) on stream M, each SM-capped so they co-reside.
        Ignores the outer->inner dependency => an UNACHIEVABLE LOWER BOUND. If even this loses to
        the pure schemes, hier cannot win. Also returns sequential (full SMs) for reference."""
        import torch.distributed as dist
        dev = self.device
        mb = sm_each if sm_each is not None else max(2, self.num_sms // 2)

        def all_outer():
            for _ in range(num_layers):
                self._scatter(mb); self._gather_reduce(mb)

        def all_inner():
            for _ in range(num_layers):
                self._inner_agv(mb); self._mask_rsv(mb)

        def all_seq():
            for _ in range(num_layers):
                self._scatter(); self._inner_agv(); self._mask_rsv(); self._gather_reduce()

        def timeit(body):
            body()
            torch.cuda.synchronize(); dist.barrier(self.group)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                body()
            e.record(); torch.cuda.synchronize()
            t = torch.tensor([s.elapsed_time(e) / reps / num_layers * 1000.0], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.group)
            return float(t.item())

        t_seq = timeit(all_seq)
        sP, sM, cur = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.current_stream()

        def both():
            sP.wait_stream(cur); sM.wait_stream(cur)
            with torch.cuda.stream(sP):
                all_outer()
            with torch.cuda.stream(sM):
                all_inner()
            cur.wait_stream(sP); cur.wait_stream(sM)

        t_bound = timeit(both)
        return {"sms_each": mb, "seq_full_us": t_seq, "pipe_bound_us": t_bound}

    # -- fused-kernel go/no-go: where does the sequential per-layer latency go? --
    def phase_decompose(self, num_layers=88, reps=20, warmup=6):
        """Split the sequential per-layer latency into its component phases, EACH
        graph-captured under identical conditions to the full decode_step (so the
        numbers are apples-to-apples with run.py). Every comm phase carries exactly
        ONE symm-mem barrier + one grid launch, so at tiny B (payload ~free) a phase's
        us is ~pure launch+barrier overhead — the part a FUSED single kernel removes.
        Returns per-layer us per phase, their sum, and the full step for cross-check.
          - transfer-bound share  -> irreducible; a fused kernel can't beat it
          - launch/barrier share   -> fusible; motivates the single-kernel build
        Requires setup_batch() first. Timed as max across ranks (critical path)."""
        nl = num_layers

        def rep(fn):
            def body():
                for _ in range(nl):
                    fn()
            (_, mx) = time_region(body, self.group, warmup, reps, use_graph=True)
            return mx / nl                                    # per-layer us

        phases = [("scatter", self._scatter), ("inner_agv", self._inner_agv),
                  ("mask_rsv", self._mask_rsv), ("gather_reduce", self._gather_reduce)]
        out = {name: rep(fn) for name, fn in phases}
        out["sum_phases"] = sum(out[n] for n, _ in phases)
        (_, full) = time_region(lambda: self.decode_step(nl), self.group, warmup, reps, use_graph=True)
        out["full_step"] = full / nl
        return out
