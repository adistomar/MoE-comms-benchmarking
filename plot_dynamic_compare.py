#!/usr/bin/env python3
# Copyright (c) 2026. Plot the AGv/RSv vs A2Av vs dynamic microbenchmark comparison.
"""Three-way comparison plot for the regime-adaptive (dynamic) MoE collectives.

Usage:
  python3 plot_dynamic_compare.py --csv results_dynamic.csv --out results_dynamic.png \
      [--dispatch-threshold 2048] [--combine-threshold 8192]

Reads a run.py CSV containing `nvls`, `a2av_push` and `dynamic` rows and emits a
two-panel figure:

  left   per-layer latency vs global token count (log/log), the quantity that actually
         shows up in a served step, with the two switch thresholds marked.
  right  the same data as speedup over NVLS (higher is better), which is where the
         point of the dynamic kernel is visible: it should sit at or above 1.0
         everywhere, tracking whichever fixed strategy is winning in that regime.

Also prints a markdown table (per-layer us and speedup vs NVLS) to stdout.
"""

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "nvls": ("NVLS AGv/RSv (multicast + in-switch reduce)", "tab:orange", "o", "-"),
    "a2av_push": ("A2Av (unicast dispatch + push combine)", "tab:blue", "s", "-"),
    "dynamic": ("Dynamic (switches per step, on device)", "black", "D", "-"),
    "deepep": ("DeepEP-v2 (all-to-all, bf16 combine)", "tab:green", "^", "--"),
}
ORDER = ["nvls", "a2av_push", "dynamic", "deepep"]


def load(path):
    """-> ({impl: {B: per_layer_us}}, ep) ; ep inferred from the per_rank_counts length."""
    data = defaultdict(dict)
    ep = None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] != "decode_step":
                continue
            if ep is None:
                parts = [
                    c
                    for c in r.get("per_rank_counts", "").strip().strip("[]").split(",")
                    if c.strip() != ""
                ]
                if parts:
                    ep = len(parts)
            data[r["impl"]][int(r["global_B"])] = float(r["latency_us"])
    return data, ep


def _ktick(v):
    """Binary tick label; non-power-of-two values keep a fraction (1792 -> '1.75k')."""
    if v >= 1 << 20:
        return f"{v / (1 << 20):g}M"
    if v > 999:
        return f"{v / 1024:g}k"
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results_dynamic.csv")
    p.add_argument("--out", default="results_dynamic.png")
    p.add_argument("--num-layers", type=int, default=88)
    p.add_argument("--dispatch-threshold", type=int, default=2048)
    p.add_argument("--combine-threshold", type=int, default=8192)
    p.add_argument("--buffer-dtype", default="fp32", choices=["fp32", "bf16"],
                   help="expert-output buffer dtype the CSV was produced with; only used "
                        "to label the plot (run.py's --out-dtype sets the real thing)")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data, ep = load(args.csv)
    nl = args.num_layers
    impls = [i for i in ORDER if i in data] + [i for i in data if i not in ORDER]
    batches = sorted({b for d in data.values() for b in d})

    # ---- table ----
    base = data.get("nvls", {})
    hdr = " | ".join(f"{i} us/layer" for i in impls)
    spd = " | ".join(f"{i} vs nvls" for i in impls if i != "nvls")
    print(f"\n| global B | tokens/rank | {hdr} | {spd} |")
    print("|" + "---|" * (3 + len(impls) + len(impls) - 1))
    for b in batches:
        cells = [f"{data[i][b] / nl:.2f}" if b in data[i] else "-" for i in impls]
        sp = [
            f"{base[b] / data[i][b]:.3f}x" if b in base and b in data[i] else "-"
            for i in impls
            if i != "nvls"
        ]
        print(f"| {b} | {b // (ep or 1)} | " + " | ".join(cells) + " | " + " | ".join(sp) + " |")

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    for impl in impls:
        label, color, marker, ls = STYLE.get(impl, (impl, "gray", "^", "--"))
        pts = sorted(data[impl].items())
        xs = [b for b, _ in pts]
        ys = [us / nl for _, us in pts]
        ax1.plot(xs, ys, marker=marker, ls=ls, lw=2, ms=5, color=color, label=label)
        if impl != "nvls" and base:
            sx = [b for b in xs if b in base]
            sy = [base[b] / data[impl][b] for b in sx]
            ax2.plot(sx, sy, marker=marker, ls=ls, lw=2, ms=5, color=color, label=label)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels([_ktick(b) for b in batches], fontsize=8)
        ax.set_xlabel("step size: tokens across all EP ranks")
        ax.grid(alpha=0.3)
        # Mark where each half of the dynamic kernel flips strategy. When both halves
        # flip at the same step size (the GB200/EP=4 fp32 case) draw a single line.
        if args.dispatch_threshold == args.combine_threshold:
            ax.axvline(args.dispatch_threshold, color="tab:red", ls=":", lw=1.5)
        else:
            ax.axvline(args.dispatch_threshold, color="tab:green", ls=":", lw=1.5)
            ax.axvline(args.combine_threshold, color="tab:red", ls=":", lw=1.5)

    ax1.set_yscale("log")
    ax1.set_ylabel("per-layer dispatch+combine latency (us)")
    ax1.set_title("absolute latency (lower is better)")
    ax1.legend(fontsize=8, loc="upper left")
    if args.dispatch_threshold == args.combine_threshold:
        ax1.annotate(
            f"switch @ {args.dispatch_threshold} tokens",
            xy=(args.dispatch_threshold, ax1.get_ylim()[0]),
            xytext=(3, 4), textcoords="offset points", fontsize=7, color="tab:red",
        )
    else:
        ax1.annotate(
            f"dispatch switch\n@ {_ktick(args.dispatch_threshold)}",
            xy=(args.dispatch_threshold, ax1.get_ylim()[0]),
            xytext=(3, 4), textcoords="offset points", fontsize=7, color="tab:green",
        )
        ax1.annotate(
            f"combine switch\n@ {_ktick(args.combine_threshold)}",
            xy=(args.combine_threshold, ax1.get_ylim()[0]),
            xytext=(3, 22), textcoords="offset points", fontsize=7, color="tab:red",
        )

    ax2.axhline(1.0, color="tab:orange", lw=2, label=STYLE["nvls"][0])
    ax2.set_ylabel("speedup over NVLS AGv/RSv  (>1 = faster)")
    ax2.set_title("relative to NVLS (higher is better)")
    ax2.legend(fontsize=8, loc="upper left")

    node_str = ""
    if ep:
        nodes = max(1, ep // 4)
        node_str = f"EP={ep} · {nodes} node{'s' if nodes > 1 else ''} (4×GB200/node) · "
    title = args.title or (
        f"MoE dispatch+combine: AGv/RSv vs A2Av vs regime-adaptive kernel\n"
        f"{node_str}512 experts · top-k=22 · hidden=1024 · "
        f"{args.buffer_dtype} expert-output buffer for NVLS/dynamic"
        + (" (DeepEP and A2Av combine in bf16)" if "fp32" in args.buffer_dtype else "")
        + f" · {nl} MoE layers in one CUDA graph"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
