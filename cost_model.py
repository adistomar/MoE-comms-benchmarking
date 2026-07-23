#!/usr/bin/env python3
# Copyright (c) 2026. Analytical cost model for MoE dispatch/combine schemes.
"""
First-order latency cost model for the three MoE dispatch/combine schemes, plus a
HYBRID "AllGather-V inner / All-to-All outer" scheme, so the AGv/A2A/hybrid
crossover can be studied at LARGE EP (e.g. EP=32, EP=63) WITHOUT a 32/72-GPU
allocation. Pure Python + stdlib (matplotlib only if --plot); no torch, no CUDA.

This does NOT replace the real benchmark — it extrapolates it. Calibrate the five
hardware constants from a measured EP=4 results.csv with --fit-csv (grounds four of
them); the fifth, the per-A2A-peer cost `tau`, cannot be recovered from a single-EP
run (it needs >=2 EP points) and is THE lever that decides whether the hybrid wins,
so it is a first-class swept parameter here (--tau-sweep).

Model (per MoE layer = paired dispatch + combine; matches results.csv "decode_step"/num_layers)
--------------------------------------------------------------------------------
Routing: each token picks k distinct experts ~uniformly; experts split evenly over
the R destinations (ranks or groups), E/R each. Then
    f(R) = 1 - (1 - 1/R)**k          # P(a token touches a given destination)
    phi(R) = R * f(R)                # expected DISTINCT destinations (dedup'd fan-out / peers)
Per-rank received token-payloads: AllGather-V gets ALL B tokens (k-independent);
All-to-All gets B*f(R) (sparsity-exploiting). token_bytes = hidden * dtype_bytes.

Per-layer latency of each primitive (constants absorb the 2 collectives disp+combine):
    AGv(vol)        = A_mc  + Bt_mc  * vol
    A2A(phi, vol)   = A_a2a + T*phi  + Bt_a2a * vol
so
    L_agv(P)   = AGv( B * token_bytes )                                     # all tokens to all
    L_a2a(P)   = A2A( phi(P), B*f(P)*token_bytes )
    L_hyb(g)   = A2A( phi(G), B*f(G)/g   * token_bytes )   # inter-group, G=P/g groups
               + AGv(         B*f(G)*(g-1)/g * token_bytes )   # intra-group multicast
with the boundaries collapsing exactly: g=P (G=1) -> L_agv, g=1 (G=P) -> L_a2a.
NVLS runs its metadata kernel once per step (not per layer): --a-meta adds it to the
per-STEP AGv/hybrid totals only.

The hybrid can never beat pure A2A on BYTES (its volume B*f(G) >= B*f(P) for G<=P);
its only edge is LATENCY -- capping the A2A radix from phi(P) to phi(G) and doing the
last hop as a cheap multicast. That edge exists only when phi(P) is large, i.e. large EP.
"""

import argparse
import csv
import math
from dataclasses import dataclass


# ---- routing combinatorics ---------------------------------------------------
def f_touch(R: int, k: int) -> float:
    """P(a token routes >=1 of its k experts to a given destination of radix R)."""
    if R <= 1:
        return 1.0
    return 1.0 - (1.0 - 1.0 / R) ** k


def phi_peers(R: int, k: int) -> float:
    """Expected number of DISTINCT destinations a token touches at radix R."""
    return R * f_touch(R, k)


def divisors(n: int):
    ds = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
    return sorted(ds)


# ---- hardware constants (per-LAYER; each absorbs the 2 collectives) -----------
@dataclass
class HW:
    A_mc: float = 3.0        # fixed multicast (AGv/RSv) launch+sync, us/layer
    Bt_mc: float = 3.3e-6    # multicast per-byte time, us/byte (~600 GB/s eff, x2 collectives)
    A_a2a: float = 3.0       # fixed all-to-all launch, us/layer
    T: float = 0.8           # per-A2A-peer cost (msg startup + count exchange), us/peer  <-- KEY UNKNOWN
    Bt_a2a: float = 3.3e-6   # all-to-all per-byte time, us/byte
    A_meta: float = 2.0      # NVLS once-per-STEP metadata (AGv/hybrid only), us/step

    def agv(self, vol_bytes: float) -> float:
        return self.A_mc + self.Bt_mc * vol_bytes

    def a2a(self, phi: float, vol_bytes: float) -> float:
        return self.A_a2a + self.T * phi + self.Bt_a2a * vol_bytes


# ---- per-layer latencies -----------------------------------------------------
def l_agv(hw: HW, P, B, k, token_bytes) -> float:
    return hw.agv(B * token_bytes)  # every rank receives all B tokens (k-independent)


def l_a2a(hw: HW, P, B, k, token_bytes) -> float:
    return hw.a2a(phi_peers(P, k), B * f_touch(P, k) * token_bytes)


def hybrid_stages(hw: HW, P, B, k, token_bytes, g: int):
    """The two per-layer stage latencies of the hybrid at group size g (G=P/g groups):
    (T_inter = inter-group A2A, T_intra = intra-group AGv multicast).
    The two stages' per-rank byte volumes sum to exactly B*f(G)*token_bytes."""
    G = P // g
    fG = f_touch(G, k)
    t_inter = hw.a2a(phi_peers(G, k), B * fG / g * token_bytes)        # sparse, few peers
    t_intra = hw.agv(B * fG * (g - 1) / g * token_bytes)               # dense multicast last hop
    return t_inter, t_intra


def l_hybrid_g(hw: HW, P, B, k, token_bytes, g: int,
               pipeline: str = "none", overlap: float = 0.7) -> float:
    """Hybrid with groups of size g. Boundaries reproduce the pure schemes.

    pipeline models how the inter-group A2A and intra-group multicast compose:
      none   : T_inter + T_intra (SEQUENTIAL -- pays both latencies + both launches).
      ideal  : max(T_inter, T_intra) (FULL overlap on disjoint resources; upper bound
               on the benefit -- almost certainly optimistic for comm+comm).
      fabric : SHARED-FABRIC pipeline -- the realistic middle. Launch overhead overlaps
               to max(A_mc,A_a2a); the inter-group A2A per-peer LATENCY hides behind the
               multicast by fraction `overlap`; but both stages contend for the SAME
               NVLink bandwidth so the BYTE terms still ADD (pipelining fills the link,
               it does not create bandwidth). This is the key model for a pipelined design.
    """
    assert P % g == 0, "group size g must divide EP size P"
    G = P // g
    if G == 1:                      # one group == whole domain -> pure AGv (no inter A2A)
        return l_agv(hw, P, B, k, token_bytes)
    if g == 1:                      # each rank its own group -> pure A2A (no intra AGv)
        return l_a2a(hw, P, B, k, token_bytes)
    ti, tx = hybrid_stages(hw, P, B, k, token_bytes, g)
    if pipeline == "none":
        return ti + tx
    if pipeline == "ideal":
        return max(ti, tx)
    if pipeline == "fabric":
        fG = f_touch(G, k)
        bytes_inter = hw.Bt_a2a * (B * fG / g * token_bytes)
        bytes_intra = hw.Bt_mc * (B * fG * (g - 1) / g * token_bytes)
        launch = max(hw.A_mc, hw.A_a2a)                # two launches overlap
        peer = (1.0 - overlap) * hw.T * phi_peers(G, k)  # A2A peer latency partly hidden
        return launch + peer + bytes_inter + bytes_intra
    raise ValueError(f"unknown pipeline mode {pipeline!r}")


def best_hybrid(hw: HW, P, B, k, token_bytes, pipeline="none", overlap=0.7):
    """Min over INTERIOR group sizes (1<g<P) -> (latency, g). Excludes the pure endpoints."""
    interior = [g for g in divisors(P) if 1 < g < P]
    if not interior:
        return math.inf, None
    best = min((l_hybrid_g(hw, P, B, k, token_bytes, g, pipeline, overlap), g)
               for g in interior)
    return best


# ---- calibration from a measured EP=4 results.csv ----------------------------
def _linfit(xs, ys):
    """Closed-form least squares y = intercept + slope*x."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return ybar - slope * xbar, slope


def fit_from_csv(path, k, token_bytes, num_layers, fit_ep, tau, verbose=True):
    """Fit A_mc,Bt_mc (from nvls rows) and A_a2a,Bt_a2a (from deepep rows) of a
    results.csv measured at EP=fit_ep. Per-layer latency = decode_step_us/num_layers,
    regressed vs global_B. tau cannot be separated from A_a2a at a single EP, so the
    caller-supplied tau is used to split the A2A intercept: A_a2a = intercept - tau*phi(P)."""
    per = {"nvls": ([], []), "deepep": ([], [])}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("phase") != "decode_step":
                continue
            impl = r["impl"]
            if impl in per:
                per[impl][0].append(int(r["global_B"]))
                per[impl][1].append(float(r["latency_us"]) / num_layers)
    hw = HW(T=tau)
    if per["nvls"][0]:
        a, c = _linfit(*per["nvls"])
        hw.A_mc, hw.Bt_mc = max(0.0, a), max(0.0, c / token_bytes)
    if per["deepep"][0]:
        a, c = _linfit(*per["deepep"])
        fP = f_touch(fit_ep, k)
        hw.Bt_a2a = max(0.0, c / (fP * token_bytes))
        hw.A_a2a = max(0.0, a - tau * phi_peers(fit_ep, k))
    if verbose:
        print(f"# fit from {path} at EP={fit_ep} (tau={tau} us/peer assumed):")
        print(f"#   A_mc={hw.A_mc:.3f}us Bt_mc={hw.Bt_mc:.3e}us/B  "
              f"A_a2a={hw.A_a2a:.3f}us Bt_a2a={hw.Bt_a2a:.3e}us/B")
        print(f"#   (Bt -> {1e-6 / hw.Bt_mc:.0f}/{1e-6 / hw.Bt_a2a:.0f} GB/s eff mc/a2a)")
    return hw


# ---- sweep / reporting -------------------------------------------------------
def winner(hw, P, B, k, token_bytes, pipeline="none", overlap=0.7):
    la = l_agv(hw, P, B, k, token_bytes)
    lt = l_a2a(hw, P, B, k, token_bytes)
    lh, g = best_hybrid(hw, P, B, k, token_bytes, pipeline, overlap)
    cands = {"agv": la, "a2a": lt, "hybrid": lh}
    win = min(cands, key=cands.get)
    return win, cands, g


def run(args, hw):
    k, tb = args.topk, args.hidden * args.bytes
    pipe, ov = args.pipeline, args.overlap
    Ps = [int(x) for x in args.ep.split(",")]
    Bs = [int(x) for x in args.batch_sizes.split(",")]
    print(f"# hybrid pipeline model = {pipe}" + (f" (overlap={ov})" if pipe == "fabric" else ""))
    rows = []
    for P in Ps:
        print(f"\n# EP={P}  f(P)={f_touch(P, k):.3f}  phi(P)={phi_peers(P, k):.1f} peers  "
              f"AGv-redundancy={1 / f_touch(P, k):.2f}x  group sizes g={divisors(P)}")
        print(f"#   {'B':>7} {'AGv':>9} {'A2A':>9} {'hybrid':>9} {'g*':>4} {'winner':>7} "
              f"{'margin':>8}")
        prev = None
        for B in Bs:
            win, c, g = winner(hw, P, B, k, tb, pipe, ov)
            others = sorted(v for kk, v in c.items() if kk != win)
            margin = (others[0] - c[win]) / c[win] * 100 if others and c[win] else 0.0
            flag = "  <-- crossover" if prev and prev != win else ""
            prev = win
            print(f"#   {B:>7} {c['agv']:>9.2f} {c['a2a']:>9.2f} {c['hybrid']:>9.2f} "
                  f"{str(g):>4} {win:>7} {margin:>7.1f}%{flag}")
            rows.append((P, B, c["agv"], c["a2a"], c["hybrid"], g, win))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ep", "global_B", "agv_us", "a2a_us", "hybrid_us",
                        "hybrid_g", "winner"])
            w.writerows(rows)
        print(f"\n# wrote {args.out}")
    if args.plot:
        _plot(rows, Ps, args)
    return rows


def _plot(rows, Ps, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("# matplotlib not available; skipping --plot")
        return
    fig, axes = plt.subplots(1, len(Ps), figsize=(6.5 * len(Ps), 5.0), squeeze=False)
    for ax, P in zip(axes[0], Ps):
        sub = [r for r in rows if r[0] == P]
        xs = [r[1] for r in sub]
        for j, (lab, col) in enumerate([("AGv (AllGather-V)", "tab:orange"),
                                        ("A2A (all-to-all)", "tab:blue"),
                                        ("hybrid (AGv-in/A2A-out)", "tab:green")]):
            ax.plot(xs, [r[2 + j] for r in sub], marker="o", ms=4, lw=2, color=col, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"EP={P}")
        ax.set_xlabel("global batch B")
        ax.set_ylabel("latency / layer (us)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"# wrote {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ep", default="32,63", help="comma EP sizes to study (default 32,63)")
    p.add_argument("--batch-sizes",
                   default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
                   help="comma GLOBAL token counts")
    p.add_argument("--topk", type=int, default=22)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--bytes", type=int, default=2, help="bytes/elem of the dispatched hidden (bf16=2)")
    p.add_argument("--num-layers", type=int, default=88, help="for per-step scaling / CSV fit")
    p.add_argument("--pipeline", choices=["none", "ideal", "fabric"], default="fabric",
                   help="how the hybrid's inter-A2A and intra-AGv compose: none=sequential, "
                        "ideal=full overlap (max, upper bound), fabric=shared-fabric realistic")
    p.add_argument("--overlap", type=float, default=0.7,
                   help="fabric mode: fraction of inter-group A2A peer latency hidden (0..1)")
    # hardware constants (per-layer); overridden by --fit-csv for all but tau
    p.add_argument("--tau", type=float, default=0.8, help="per-A2A-peer cost us/peer (KEY unknown)")
    p.add_argument("--tau-sweep", default=None,
                   help="comma taus to sweep (repeats the whole table per tau)")
    p.add_argument("--a-mc", type=float, default=None)
    p.add_argument("--bt-mc", type=float, default=None)
    p.add_argument("--a-a2a", type=float, default=None)
    p.add_argument("--bt-a2a", type=float, default=None)
    p.add_argument("--fit-csv", default=None, help="measured results.csv to calibrate from")
    p.add_argument("--fit-ep", type=int, default=4, help="EP the --fit-csv was measured at")
    p.add_argument("--out", default=None, help="write sweep CSV")
    p.add_argument("--plot", default=None, help="write per-EP latency plot PNG")
    return p.parse_args()


def build_hw(args, tau):
    tb = args.hidden * args.bytes
    if args.fit_csv:
        hw = fit_from_csv(args.fit_csv, args.topk, tb, args.num_layers, args.fit_ep, tau)
    else:
        hw = HW(T=tau)
    for name, val in [("A_mc", args.a_mc), ("Bt_mc", args.bt_mc),
                      ("A_a2a", args.a_a2a), ("Bt_a2a", args.bt_a2a)]:
        if val is not None:
            setattr(hw, name, val)
    return hw


def main():
    args = parse_args()
    print(f"# MoE dispatch cost model | experts={args.experts} topk={args.topk} "
          f"hidden={args.hidden} bytes={args.bytes} num_layers={args.num_layers}")
    if not args.fit_csv:
        print("# WARNING: using PLACEHOLDER hardware constants. Pass --fit-csv results.csv "
              "(EP=4 run) to ground A_mc/Bt_mc/A_a2a/Bt_a2a; tau still needs an assumption.")
    taus = ([float(x) for x in args.tau_sweep.split(",")] if args.tau_sweep else [args.tau])
    for tau in taus:
        if len(taus) > 1:
            print(f"\n########## tau = {tau} us/peer ##########")
        run(args, build_hw(args, tau))


if __name__ == "__main__":
    main()
