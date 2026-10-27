# Copyright (c) 2026. Hybrid (multicast or unicast, chosen on device) dispatch/combine bencher.
"""
Isolated bencher for the hybrid kernels in nvls/torch_symm_triton/hybrid_v.py. Same buffers,
row layout and once-per-step metadata as bench_a2a.py, but the metadata kernel also evaluates the
mode-selection rule (parametrized by a device tensor, mode_params) and writes two mode flags that
the kernels read on device:

  metadata (once per step) : fused_metadata_update(mode_params)
                             -> [valid, offset, ep_max, dispatch_multicast, combine_reduce]
  dispatch (per MoE layer) : hybrid_dispatch_v (hidden multicast as AGV-V, or unicast to the
                             destination ranks as A2A-V; routing, probs multicast either way)
  combine  (per MoE layer) : hybrid_combine_v (multimem.ld_reduce as RSV-V, or pull as A2A-V)

Modes (set_mode / run.py --hybrid-modes). The first three are mode-selection rules, evaluated
by the metadata kernel (nvls/metadata.py) with a separate threshold for dispatch and combine:
  global            every rank: multicast iff valid_tokens - ep_max_weight * ep_max_tokens <=
                    threshold (ep_max_weight from --hybrid-ep-max-weight, default 0: a plain
                    threshold on the global token count).
  global_imbalance  global, ep_max_weight = u (a token's expected remote destinations): also
                    multicasts every rank once the heaviest rank's unicast fan-out u * ep_max
                    exceeds valid_tokens. The threshold is scaled by (1 - u / EP) so a balanced
                    batch switches at the same B.
  local             per rank: multicast if valid_tokens <= threshold (latency regime), else if
                    this rank is one of the heaviest senders, u * n_s > q * N + (1 - q) * S_s
                    (bandwidth model); flags may then differ across ranks.
  mc_rs             force multicast dispatch + multicast-reduce combine (the AGV-V / RSV-V pair)
  uc_pull           force unicast dispatch + pull combine               (the A2A-V pair)
  mc_pull, uc_rs    the mixed pairs.
Every mode uses the same kernels, buffers and launch grid; only the device mode_params tensor
changes, so a captured graph can switch modes between replays.

With --expert vllm the expert always writes a zero row for tokens with no local expert
(write_unrouted_rows=True): the multicast-reduce combine sums every rank's row.
"""

import math

import torch
import torch.distributed as dist

from nvls.metadata import (
    ALWAYS_MULTICAST,
    ALWAYS_UNICAST,
    HYBRID_STEP_METADATA_LEN,
    MODE_PARAMS_LEN,
    fused_metadata_update,
    hybrid_mode_flags,
    hybrid_mode_params,
)
from nvls.torch_symm_triton.hybrid_v import hybrid_combine_v, hybrid_dispatch_v

from bench_a2a import A2ABencher
from common import Config, all_rank_counts

# mode -> (dispatch_multicast, combine_reduce) for forced modes; None for the selection rules.
MODES = {
    "global": None,
    "global_imbalance": None,
    "local": None,
    "mc_rs": (1, 1),
    "uc_pull": (0, 0),
    "mc_pull": (1, 0),
    "uc_rs": (0, 1),
}
FORCED_MODES = tuple(m for m, flags in MODES.items() if flags is not None)
# Prototype per-token split: mode "split<f>" (e.g. split0.5) forces both flags on and lets only
# the first round(f * active CTAs) CTAs of each rank multicast / multicast-reduce; the rest of
# the rank's tokens (round-robin over CTAs) are unicast / pulled.
SPLIT_PREFIX = "split"
NO_SPLIT = 2**31 - 1


def split_fraction(mode: str):
    """f of a "split<f>" mode, else None."""
    if not mode.startswith(SPLIT_PREFIX):
        return None
    f = float(mode[len(SPLIT_PREFIX):])
    assert 0.0 <= f <= 1.0, f"split fraction {f} outside [0, 1]"
    return f


def is_valid_mode(mode: str) -> bool:
    try:
        return mode in MODES or split_fraction(mode) is not None
    except (ValueError, AssertionError):
        return False

# Auto thresholds (dispatch, combine) in global tokens: multicast at B <= threshold. EP=4 and
# EP=64 are per-phase crossovers on GB200 (communication only; the largest swept B, on a
# sqrt(2) grid, at which the multicast side of that collective was still faster:
# run.py --phases dispatch,combine --hybrid-modes mc_rs,uc_pull). EP=16/32 are not yet measured
# per phase: the largest power-of-two B at which AGV-V/RSV-V beat A2A-V in plots/comm, for
# both. EPs not listed use the nearest listed EP (log scale).
DEFAULT_THRESHOLDS = {
    "super": {4: (2896, 4096), 16: (4096, 4096), 32: (2048, 2048), 64: (1448, 1448)},
    "ultra": {4: (1448, 2896), 16: (2048, 2048), 32: (1024, 1024), 64: (724, 724)},
}


def default_thresholds(model: str, ep_size: int) -> "tuple[int, int]":
    table = DEFAULT_THRESHOLDS[model]
    nearest = min(table, key=lambda ep: (abs(math.log2(ep) - math.log2(ep_size)), ep))
    return table[nearest]


def destination_probability(num_experts: int, topk: int, ep_size: int) -> float:
    """q = P(a given rank holds at least one of a token's top-k experts), uniform routing:
    1 - C(E - E/EP, k) / C(E, k). A token's expected remote destinations are u = (EP - 1) q."""
    local = num_experts // ep_size
    return 1.0 - math.comb(num_experts - local, topk) / math.comb(num_experts, topk)


class HybridBencher(A2ABencher):
    name = "hybrid"
    buffer_prefix = "hyb"

    def __init__(self, cfg: Config, group, thresholds=None, ep_max_weight=(0.0, 0.0)):
        super().__init__(cfg, group)
        # (dispatch, combine) global rule: multicast iff valid - ep_max_weight * ep_max <= threshold.
        self.thresholds = (tuple(thresholds) if thresholds
                           else default_thresholds(cfg.model, cfg.ep_size))
        self.ep_max_weight = tuple(ep_max_weight)
        # Bandwidth-model constants of the global_imbalance / local rules (uniform routing):
        # q = P(a given rank is one of a token's destinations), u = expected remote destinations.
        self.dest_prob = destination_probability(cfg.num_experts, cfg.topk, cfg.ep_size)
        self.remote_fanout = (cfg.ep_size - 1) * self.dest_prob  # u
        self.mode = "global"

    # -- one-time allocation (collective) --------------------------------------
    def build(self):
        super().build()
        # [valid_tokens, rank_token_offset, ep_max_tokens, dispatch_multicast, combine_reduce]
        self.step_metadata = torch.zeros(HYBRID_STEP_METADATA_LEN, dtype=torch.int32,
                                         device=self.device)
        self.mode_params = torch.zeros(MODE_PARAMS_LEN, dtype=torch.int32, device=self.device)
        # [dispatch, combine] multicasting-CTA counts of a split mode (NO_SPLIT otherwise).
        self.mc_split = torch.full((2,), NO_SPLIT, dtype=torch.int32, device=self.device)
        self.split = None
        self.set_mode("global")

    # -- modes ---------------------------------------------------------------
    def mode_params_for(self, mode: str):
        """Host mode parameters (nvls.metadata.hybrid_mode_params) for `mode`."""
        t_d, t_c = self.thresholds
        if mode == "global":
            return hybrid_mode_params(t_d, t_c, *self.ep_max_weight)
        if mode == "global_imbalance":
            # Balanced batch: valid - u * valid / EP <= T * (1 - u / EP) <=> valid <= T.
            u = self.remote_fanout
            scale = 1.0 - u / self.cfg.ep_size
            return hybrid_mode_params(round(t_d * scale), round(t_c * scale), u, u)
        if mode == "local":
            return hybrid_mode_params(t_d, t_c, local_rule=True, dest_prob=self.dest_prob)
        flags = (1, 1) if split_fraction(mode) is not None else MODES[mode]
        t = [ALWAYS_MULTICAST if f else ALWAYS_UNICAST for f in flags]
        return hybrid_mode_params(t[0], t[1])

    def set_mode(self, mode: str):
        """Select a mode (collective: checks the mode parameters are identical on every
        rank)."""
        assert is_valid_mode(mode), (
            f"unknown hybrid mode '{mode}' (choose from {tuple(MODES)} or split<f>)")
        self.mode = mode
        self.split = split_fraction(mode)
        self.name = (f"hybrid-split{self.split:g}" if self.split is not None
                     else f"hybrid-{mode}")
        self.set_mode_params(self.mode_params_for(mode))
        self._update_split()

    def _update_split(self):
        """This rank's multicasting-CTA count for a split mode: round(f * active CTAs), where
        active CTAs = min(grid, local tokens). Set before graph capture (setup_batch)."""
        if self.split is None or not hasattr(self, "local_tokens"):
            return
        active = min(self.num_blocks, self.local_tokens)
        m = round(self.split * active)
        self.mc_split.copy_(torch.tensor([m, m], dtype=torch.int32))

    def setup_batch(self, hidden, topk_idx, topk_weights):
        super().setup_batch(hidden, topk_idx, topk_weights)
        self._update_split()

    def set_mode_params(self, values):
        """Write the device mode_params tensor. Every rank must use the same parameters (the
        flags may still differ per rank under the local rule); checked with a MIN/MAX
        all-reduce."""
        self.mode_params.copy_(torch.tensor(values, dtype=torch.int32))
        lo, hi = self.mode_params.clone(), self.mode_params.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=self.group)
        assert torch.equal(lo, hi), f"hybrid mode_params differ across ranks: {lo} vs {hi}"

    def describe_modes(self) -> str:
        return (f"thresholds (dispatch, combine) = {self.thresholds}; global: ep_max_weight = "
                f"{self.ep_max_weight}; global_imbalance: ep_max_weight = u = "
                f"{self.remote_fanout:.2f}; local: q = {self.dest_prob:.4f}")

    # -- timed / setup ops -----------------------------------------------------
    def metadata(self):
        """Once-per-step token-count exchange + mode flags (mode_params read on device)."""
        fused_metadata_update(
            local_tokens=self.local_tokens,
            local_buf=self.meta["tensor"],
            symm_mem_hdl=self.meta["handle"],
            step_metadata=self.step_metadata,
            mode_params=self.mode_params,
        )

    def dispatch(self):
        hybrid_dispatch_v(
            self.recv_h["tensor"], self.recv_r["tensor"], self.recv_p["tensor"],
            self.in_hidden, self.in_routing, self.in_probs,
            self.recv_h["handle"], self.recv_r["handle"], self.recv_p["handle"],
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            dispatch_mode=self.step_metadata[3:4],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_local_experts=self.cfg.num_local_experts,
            max_num_blocks=self.num_sms,
            mc_split=self.mc_split[0:1] if self.split is not None else None,
        )

    def combine(self):
        hybrid_combine_v(
            self.out,
            self.partial["tensor"],
            self.partial["handle"],
            self.in_routing,
            rank_token_offset=self.step_metadata[1:2],
            ep_max_tokens=self.step_metadata[2:3],
            combine_mode=self.step_metadata[4:5],
            per_rank_max_tokens=self.cfg.per_rank_cap,
            num_local_experts=self.cfg.num_local_experts,
            max_num_blocks=self.num_sms,
            mc_split=self.mc_split[1:2] if self.split is not None else None,
        )

    def run_expert(self):
        """As A2A-V, but always writing a zero row for tokens with no local expert, which the
        multicast-reduce combine needs (and the pull combine ignores)."""
        cfg = self.cfg
        self.expert(self.recv_h["tensor"], self.recv_p["tensor"], self.recv_r["tensor"],
                    self.step_metadata[0:1], cfg.rank * cfg.num_local_experts,
                    self.partial["tensor"], self.local_tokens * cfg.ep_size,
                    write_unrouted_rows=True)

    # -- correctness -----------------------------------------------------------
    def _inputs(self, counts, seed):
        """Inputs identical on all ranks for per-rank token `counts`: (n, off, valid, full_h,
        full_r, full_p, part_full, local). Routing reaches a random subset of ranks per token,
        with a padding token (all -1) at the end; `local` = rows with an expert on this rank."""
        rank, dev = self.cfg.rank, self.device
        H, K = self.cfg.hidden, self.cfg.topk
        n, off, valid = counts[rank], sum(counts[:rank]), sum(counts)
        gen = torch.Generator(device=dev).manual_seed(seed)
        full_h = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
        full_r = self._validation_routing(valid, gen)
        full_p = torch.rand(valid, K, generator=gen, device=dev)
        part_full = torch.randn(valid, H, generator=gen, device=dev).to(torch.bfloat16)
        self.setup_batch(full_h[off:off + n].contiguous(), full_r[off:off + n].contiguous(),
                         full_p[off:off + n].contiguous())
        return n, off, valid, full_h, full_r, full_p, part_full, self._local_rows(full_r)

    def _check_dispatch(self, valid, full_h, full_r, full_p, local, multicast):
        """Routing + probs bit-exact on every row. Hidden: multicast -> every row bit-exact;
        unicast -> routed rows bit-exact and the other rows keep their NaN sentinel."""
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        recv_h = self.recv_h["tensor"].view(gcap, H)[:valid]
        meta_ok = (torch.equal(self.recv_r["tensor"].view(gcap, K)[:valid], full_r)
                   and torch.equal(self.recv_p["tensor"].view(gcap, K)[:valid], full_p))
        if multicast:
            hid_ok = torch.equal(recv_h, full_h)
            what = f"hidden bit-exact on all {valid} rows"
        else:
            hid_ok = (torch.equal(recv_h[local], full_h[local])
                      and bool(torch.isnan(recv_h[~local].float()).all()))
            what = f"hidden bit-exact on {int(local.sum())}/{valid} routed rows, NaN holes kept"
        return meta_ok and hid_ok, f"routing+probs bit-exact; {what}"

    def _fill_partial(self, valid, part_full, local, other):
        """This rank's partial rows: part_full where routed here, `other` elsewhere."""
        H, gcap = self.cfg.hidden, self.cfg.global_cap
        partial = self.partial["tensor"].view(gcap, H)
        partial[:valid] = other
        partial[:valid][local] = part_full[local]

    def _combine_ref(self, n, off, full_r, part_full, unrouted_value, reduce):
        """fp32 expected combine output for this rank's tokens. Pull sums the m destination
        ranks' partials (m*x); multicast-reduce sums every rank's row, so the other w - m
        ranks add their unrouted_value."""
        m = self._num_dest_ranks(full_r[off:off + n]).view(n, 1)
        ref = m * part_full[off:off + n].float()
        if reduce:
            ref = ref + (self.cfg.ep_size - m) * unrouted_value
        return ref

    def _sync_all(self):
        torch.cuda.synchronize()
        dist.barrier(self.group)

    def _validate_mode_flags(self):
        """The metadata kernel's flags == hybrid_mode_flags for a spread of mode parameters
        (thresholds around valid_tokens, ep_max_weight 0 .. EP, local rule on/off with random
        q), on balanced, imbalanced and all-on-one-rank count vectors."""
        w, K, H, dev = self.cfg.ep_size, self.cfg.topk, self.cfg.hidden, self.device
        gen = torch.Generator().manual_seed(77)  # CPU, identical on all ranks
        out = []
        for label, counts in (("B=2EP", all_rank_counts(2 * w, w)),
                              ("B=1", all_rank_counts(1, w)),
                              ("rho=3", all_rank_counts(4 * w, w, min(3.0, w))),
                              ("rho=EP", all_rank_counts(4 * w, w, w))):
            n, B = counts[self.cfg.rank], sum(counts)
            self.setup_batch(torch.zeros(n, H, dtype=torch.bfloat16, device=dev),
                             torch.zeros(n, K, dtype=torch.int64, device=dev),
                             torch.zeros(n, K, dtype=torch.float32, device=dev))
            bad, n_rank = [], 0
            for _ in range(24):
                thresholds = [int(torch.randint(-2, 2 * B + 3, (1,), generator=gen))
                              for _ in range(2)]
                weights = [float(torch.randint(0, 4 * w + 1, (1,), generator=gen)) / 4
                           for _ in range(2)]
                local_rule = bool(torch.randint(0, 2, (1,), generator=gen))
                q = float(torch.rand(1, generator=gen))
                n_rank += local_rule
                vals = hybrid_mode_params(*thresholds, *weights, local_rule, q)
                self.set_mode_params(vals)
                self.metadata()
                got = tuple(self.step_metadata[3:5].tolist())
                want = hybrid_mode_flags(vals, counts, self.cfg.rank)
                if got != want:
                    bad.append((vals, got, want))
            out.append((f"hybrid mode flags    {label:6s}", not bad,
                        f"24 random mode_params ({n_rank} local), counts={counts[:4]}..."
                        + (f"; mismatches {bad[:2]}" if bad else "")))
        return out

    def _validate_local(self):
        """The local rule on an imbalanced batch, where ranks pick different modes. Every
        rank's flags match the host reference; rows of multicasting senders reach every rank
        and rows of unicasting senders only their destinations (NaN elsewhere); each owner's
        combine follows its own flag (unrouted partial rows hold 1.0, which only the
        multicast-reduce adds)."""
        w, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        # Rank 0 holds h tokens, every other rank 1. The rule multicasts rank 0 iff
        # h * (u - 1) > u, and never a rank holding 1 token here; h must fit the buffers.
        u = self.remote_fanout
        h_min = math.floor(u / (u - 1)) + 1 if u > 1 else None
        if h_min is None or h_min > self.cfg.per_rank_cap:
            return [("hybrid local          ", True,
                     f"skipped: no imbalance within per_rank_cap={self.cfg.per_rank_cap} "
                     f"makes ranks differ (u={u:.2f})")]
        counts = [min(self.cfg.per_rank_cap, max(h_min, 8))] + [1] * (w - 1)
        n, off, valid, full_h, full_r, full_p, part_full, local = self._inputs(counts, 31337)
        # The bandwidth rule alone (no latency-regime override): the heavy rank multicasts.
        vals = hybrid_mode_params(ALWAYS_UNICAST, ALWAYS_UNICAST,
                                  local_rule=True, dest_prob=self.dest_prob)
        self.set_mode_params(vals)
        self.metadata()
        recv_h = self.recv_h["tensor"].view(gcap, H)
        recv_h[:valid] = float("nan")
        self._sync_all()
        self.dispatch()
        torch.cuda.synchronize()
        gathered = [torch.empty(2, dtype=torch.int32, device=dev) for _ in range(w)]
        dist.all_gather(gathered, self.step_metadata[3:5].clone(), group=self.group)
        flags = [tuple(f.tolist()) for f in gathered]
        want = [hybrid_mode_flags(vals, counts, r) for r in range(w)]
        mixed = len({f for f, c in zip(flags, counts) if c > 0}) > 1
        owner = torch.repeat_interleave(torch.arange(w, device=dev),
                                        torch.tensor(counts, device=dev))
        sender_mc = torch.tensor([f[0] for f in flags], device=dev, dtype=torch.bool)[owner]
        present = local | sender_mc
        got_h = recv_h[:valid]
        disp_ok = (flags == want and mixed
                   and torch.equal(got_h[present], full_h[present])
                   and bool(torch.isnan(got_h[~present].float()).all())
                   and torch.equal(self.recv_r["tensor"].view(gcap, K)[:valid], full_r)
                   and torch.equal(self.recv_p["tensor"].view(gcap, K)[:valid], full_p))
        out = [("hybrid local dispatch   ", disp_ok,
                f"counts={counts[:4]}..., flags={flags[:4]}..., mixed={mixed}; rows of "
                f"multicasting senders on every rank, of unicasting senders only on their "
                f"destinations")]

        unrouted = 1.0
        self._fill_partial(valid, part_full, local, unrouted)
        self._sync_all()
        self.combine()
        torch.cuda.synchronize()
        reduce = flags[rank][1]
        if n > 0:
            ref = self._combine_ref(n, off, full_r, part_full, unrouted, reduce)
            comb_ok = (torch.allclose(self.out.float(), ref, rtol=2**-7, atol=1e-3) if reduce
                       else torch.equal(self.out, ref.to(torch.bfloat16)))
        else:
            comb_ok = True
        out.append(("hybrid local combine    ", comb_ok,
                    f"this rank {'multicast-reduce' if reduce else 'pull'}s its {n} tokens"))
        return out

    def _validate_forced(self, mode):
        """Dispatch and combine known-value checks with `mode` forced, at B=2*EP and B=1."""
        multicast, reduce = MODES[mode]
        w = self.cfg.ep_size
        out = []
        for B in (w * 2, 1):
            n, off, valid, full_h, full_r, full_p, part_full, local = self._inputs(
                all_rank_counts(B, w), 20261025 + B)
            self.metadata()
            self.recv_h["tensor"].view(self.cfg.global_cap, self.cfg.hidden)[:valid] = float("nan")
            self._sync_all()  # every sentinel is written before anyone dispatches
            self.dispatch()
            torch.cuda.synchronize()
            flags = tuple(self.step_metadata[3:5].tolist())
            ok, detail = self._check_dispatch(valid, full_h, full_r, full_p, local, multicast)
            out.append((f"hybrid[{mode}] dispatch  B={B:<4d}", ok and flags == (multicast, reduce),
                        f"flags={flags}; {detail}"))

            # Multicast-reduce needs zero rows where the token has no expert here (what the
            # expert writes with write_unrouted_rows=True); pull must never read them (NaN).
            self._fill_partial(valid, part_full, local, 0.0 if reduce else float("nan"))
            self._sync_all()
            self.combine()
            torch.cuda.synchronize()
            if n > 0:
                ref = self._combine_ref(n, off, full_r, part_full, 0.0, reduce).to(torch.bfloat16)
                if reduce:
                    # multimem.ld_reduce (acc::f32) is not always bit-equal to bf16(m*x): allow
                    # one bf16 ulp (as RSV-V's own check allows rounding) and report how close.
                    ulps = (self.out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
                    comb_ok = int(ulps.max()) <= 1
                    detail = (f"output within 1 bf16 ulp of bf16(m*partial): "
                              f"{float((ulps == 0).float().mean()):.2%} bit-exact, "
                              f"max {int(ulps.max())} ulp; local_tokens={n}")
                else:
                    comb_ok = torch.equal(self.out, ref)
                    detail = f"output == bf16(m*partial) bit-exact; local_tokens={n}"
            else:
                comb_ok, detail = True, "n=0 (idle rank still participates)"
            out.append((f"hybrid[{mode}] combine   B={B:<4d}", comb_ok, detail))
        return out

    def _validate_graph_switch(self):
        """One captured graph (metadata -> dispatch -> combine), replayed with mode_params
        changed between replays: each replay must follow the mode its flags select. Unrouted
        partial rows hold 1.0, so the multicast-reduce combine (which adds them) and the pull
        combine (which never reads them) give different results; unicast dispatch leaves NaN
        holes that multicast dispatch overwrites."""
        w = self.cfg.ep_size
        B = 2 * w
        n, off, valid, full_h, full_r, full_p, part_full, local = self._inputs(
            all_rank_counts(B, w), 4242)
        unrouted = 1.0
        cases = [(m, self.mode_params_for(m), MODES[m]) for m in FORCED_MODES]
        # Threshold boundary (ep_max_weight = 0): multicast at valid <= threshold, unicast one
        # token below.
        cases += [("global threshold=valid", hybrid_mode_params(valid, valid), (1, 1)),
                  ("global threshold=valid-1", hybrid_mode_params(valid - 1, valid - 1), (0, 0)),
                  ("global mixed", hybrid_mode_params(valid, valid - 1), (1, 0))]

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up on a side stream before capture
            self.metadata(); self.dispatch(); self.combine()
        torch.cuda.current_stream().wait_stream(side)
        self._sync_all()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.metadata(); self.dispatch(); self.combine()
        self._sync_all()

        H, gcap = self.cfg.hidden, self.cfg.global_cap
        bad = []
        for label, vals, want in cases:
            self.set_mode_params(vals)
            self.recv_h["tensor"].view(gcap, H)[:valid] = float("nan")
            self._fill_partial(valid, part_full, local, unrouted)
            self.out.fill_(float("nan"))
            self._sync_all()
            g.replay()
            torch.cuda.synchronize()
            flags = tuple(self.step_metadata[3:5].tolist())
            disp_ok, _ = self._check_dispatch(valid, full_h, full_r, full_p, local, want[0])
            comb_ok = True
            if n > 0:
                ref = self._combine_ref(n, off, full_r, part_full, unrouted, want[1])
                # Multicast-reduce adds the 1.0 rows in fp32 before one bf16 rounding: within
                # one bf16 ulp of the fp32 reference (the 1.0 rows themselves are >> 1 ulp).
                comb_ok = torch.allclose(self.out.float(), ref, rtol=2**-7, atol=1e-3)
            if not (flags == want and disp_ok and comb_ok):
                bad.append(f"{label}: flags={flags} want={want} dispatch_ok={disp_ok} "
                           f"combine_ok={comb_ok}")
        del g
        return [(f"hybrid graph switch  B={B:<4d}", not bad,
                 f"{len(cases)} mode_params replayed in one captured graph"
                 + (f"; {bad}" if bad else ""))]

    def _validate_split(self):
        """Per-token mixing (mode split0.5): on every rank the first half of the active CTAs
        multicast and multicast-reduce, the others unicast and pull. Rows of multicast tokens
        reach every rank and rows of unicast tokens only their destinations (NaN elsewhere);
        each token's combine follows its CTA (unrouted partial rows hold 1.0, which only the
        multicast-reduce adds)."""
        w, dev = self.cfg.ep_size, self.device
        H, K, gcap = self.cfg.hidden, self.cfg.topk, self.cfg.global_cap
        n_per = min(self.cfg.per_rank_cap, 6)
        if n_per < 2:
            return [("hybrid split           ", True, "skipped: per_rank_cap < 2")]
        mode = self.mode
        self.set_mode("split0.5")
        counts = [n_per] * w
        n, off, valid, full_h, full_r, full_p, part_full, local = self._inputs(counts, 2718)
        m = round(0.5 * min(self.num_blocks, n_per))
        self.metadata()
        recv_h = self.recv_h["tensor"].view(gcap, H)
        recv_h[:valid] = float("nan")
        self._sync_all()
        self.dispatch()
        torch.cuda.synchronize()
        flags = tuple(self.step_metadata[3:5].tolist())
        token_index = torch.arange(n_per, device=dev).repeat(w)  # every row's local index
        present = local | ((token_index % self.num_blocks) < m)
        got_h = recv_h[:valid]
        disp_ok = (flags == (1, 1)
                   and torch.equal(got_h[present], full_h[present])
                   and bool(torch.isnan(got_h[~present].float()).all())
                   and torch.equal(self.recv_r["tensor"].view(gcap, K)[:valid], full_r)
                   and torch.equal(self.recv_p["tensor"].view(gcap, K)[:valid], full_p))
        out = [("hybrid split dispatch  ", disp_ok,
                f"{m} of {n_per} tokens per rank multicast, the rest unicast")]

        unrouted = 1.0
        self._fill_partial(valid, part_full, local, unrouted)
        self._sync_all()
        self.combine()
        torch.cuda.synchronize()
        reduce = (torch.arange(n, device=dev) % self.num_blocks) < m
        ref_rs = self._combine_ref(n, off, full_r, part_full, unrouted, True)
        ref_pull = self._combine_ref(n, off, full_r, part_full, unrouted, False)
        comb_ok = (torch.allclose(self.out.float()[reduce], ref_rs[reduce], rtol=2**-7, atol=1e-3)
                   and torch.equal(self.out[~reduce], ref_pull[~reduce].to(torch.bfloat16)))
        out.append(("hybrid split combine   ", comb_ok,
                    f"{int(reduce.sum())} tokens multicast-reduced, {int((~reduce).sum())} pulled"))
        self.set_mode(mode)
        return out

    def validate(self):
        """Known-value checks -> list of (name, ok, detail): the mode flags, every forced
        mode's dispatch and combine, mode switching inside one captured graph, and the
        local rule with different modes on different ranks."""
        mode = self.mode
        out = self._validate_mode_flags()
        for m in FORCED_MODES:
            self.set_mode(m)
            out += self._validate_forced(m)
        out += self._validate_graph_switch()
        out += self._validate_local()
        out += self._validate_split()
        self.set_mode(mode)
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """Dispatch -> identity expert -> combine in the current mode, returning [n,H] ==
        (#distinct destination ranks for t) * x[t] (cross-impl check). The identity expert
        copies the received rows routed here into the partials and zeroes the others, as the
        expert does with write_unrouted_rows=True."""
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
        self.combine()  # its leading barrier orders every rank's partial writes before reads
        return self.out
