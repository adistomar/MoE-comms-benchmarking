#!/usr/bin/env python3
# Copyright (c) 2026. Plot MoE dispatch/combine latency vs CTA (block) count.
"""
Plot a block-count sweep from sweep_ctas_nsys.sh: per-layer latency vs number of CTAs, one
line per implementation, one figure per global batch size B.

Each figure has three panels on a shared x-axis:
  total    end-to-end per-layer latency from run.py (decode step / num_layers, max across ranks)
  dispatch dispatch-side kernel medians from the rank-0 nsys trace (DeepEP: + copy epilogue)
  combine  combine-side kernel medians (DeepEP: + reduce epilogue)
The kernel panels are drawn only when the analyze_nsys_sweep.py CSV is present.

x is the grid size: CTAs launched for NVLS / A2A-V (one token per CTA, grid-stride), and
num_sms for DeepEP (one CTA per SM, so it stops at the device SM count). Outputs go to
plots/block-size-scaling/: <tag>_B<B>.png per batch size and <tag>_overview.png (totals only).

  python3 plot_block_scaling.py --csv ctasweep_ep4_<jobid>.csv
  python3 plot_block_scaling.py --csv blocks_ep4_deepep.csv,blocks_ep4_nvls.csv,blocks_ep4_a2a.csv
"""

import argparse
import csv
import math
import os
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
IMPLS = ("nvls", "deepep", "a2a")
IMPL_NAME = {"deepep": "DeepEP-v2", "nvls": "NVLS (AGV/RSV)", "a2a": "A2A-V"}
# First three slots of the dataviz reference categorical palette (validated all-pairs).
IMPL_COLOR = {"deepep": "#2a78d6", "nvls": "#eb6834", "a2a": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")
REF_SMS = 148  # NVLS/A2A default cap = 1 CTA per SM; marks the "k CTAs per SM" guides
# Kernel columns of analyze_nsys_sweep.py summed into each side, per impl.
SIDES = {
    "dispatch": {"a2a": ["a2a_dispatch"], "nvls": ["nvls_dispatch"],
                 "deepep": ["dispatch", "dispatch_epilogue"]},
    "combine": {"a2a": ["a2a_combine"], "nvls": ["nvls_combine"],
                "deepep": ["combine", "combine_epilogue"]},
}


def _resolve(path, default_dir):
    return path if os.path.dirname(path) else os.path.join(default_dir, path)


def load_totals(path, num_layers):
    """{(impl, B): {ctas: per-layer us}} and the EP size (from per_rank_counts)."""
    out, ep = defaultdict(dict), None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] != "decode_step" or r["impl"] not in IMPLS:
                continue
            if ep is None:
                ep = len([c for c in r["per_rank_counts"].strip("[]").split(",") if c.strip()])
            out[(r["impl"], int(r["global_B"]))][int(r["num_sms"])] = (
                float(r["latency_us"]) / num_layers)
    return out, ep


def load_kernels(path):
    """{side: {(impl, B): {ctas: us}}} from the analyze_nsys_sweep.py CSV."""
    out = {side: defaultdict(dict) for side in SIDES}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            impl = r["impl"]
            if impl not in IMPLS:
                continue
            for side, cols in SIDES.items():
                vals = [r.get(f"{c}_us", "") for c in cols[impl]]
                if vals and all(v not in ("", None) for v in vals):
                    out[side][(impl, int(r["B"]))][int(r["ctas"])] = sum(map(float, vals))
    return out


def _tick_labels(xs, min_ratio=1.25):
    """Labels for the x ticks, dropping ones too close (in log space) to an already-labeled
    tick; REF_SMS and the extremes win ties, so 148 is kept over 128/152."""
    order = sorted(xs, key=lambda x: (x != REF_SMS, x not in (min(xs), max(xs)), x))
    kept = []
    for x in order:
        if all(max(x, k) / min(x, k) >= min_ratio for k in kept):
            kept.append(x)
    return [str(x) if x in kept else "" for x in xs]


def _style(ax, xs):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(_tick_labels(xs), fontsize=7, color=MUTED)
    # Plain-number y ticks at 1/2/5 x 10^k (no "3x10^1" labels).
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelsize=7, colors=MUTED, which="both")
    ax.tick_params(axis="x", colors=MUTED)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="-")
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    # "k CTAs per SM" guides for the one-token-per-CTA kernels.
    lo, hi = min(xs), max(xs)
    for k in (1, 2, 4, 8):
        x = REF_SMS * k
        if lo <= x <= hi:
            ax.axvline(x, color=AXIS, linewidth=0.6, linestyle=":", zorder=0)
            ax.text(x, 1.0, f" {k}/SM", transform=ax.get_xaxis_transform(), fontsize=6,
                    color=MUTED, va="bottom", ha="left")


def _draw(ax, series, title):
    """series: {impl: {ctas: us}}. Lines with end markers; direct labels when they don't collide."""
    ends = []
    for impl in IMPLS:
        pts = sorted(series.get(impl, {}).items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=1.2, solid_joinstyle="round",
                solid_capstyle="round", marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.0, label=IMPL_NAME[impl], zorder=3)
        ends.append((ys[-1], xs[-1], impl))
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    # Direct labels at line ends (text in secondary ink), skipped if any two ends are close.
    logs = sorted(math.log10(y) for y, _, _ in ends)
    if all(b - a > 0.08 for a, b in zip(logs, logs[1:])):
        for y, x, impl in ends:
            ax.annotate(IMPL_NAME[impl], (x, y), xytext=(5, 0), textcoords="offset points",
                        fontsize=7, color=INK_2, va="center")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="run.py CSV(s) of the sweep, comma-separated (bare names -> results/)")
    p.add_argument("--kernels-csv", default=None,
                   help="analyze_nsys_sweep.py CSV (default: <csv stem>_kernels.csv if present)")
    p.add_argument("--num-layers", type=int, default=88)
    p.add_argument("--min-blocks", type=int, default=0,
                   help="drop points with fewer CTAs (DeepEP: num_sms) than this")
    p.add_argument("--tag", default=None, help="output file prefix (default: ep<EP>)")
    p.add_argument("--results-dir", default=os.path.join(REPO_DIR, "results"))
    p.add_argument("--out-dir", default=os.path.join(REPO_DIR, "plots", "block-size-scaling"))
    args = p.parse_args()

    csv_paths = [_resolve(c, args.results_dir) for c in args.csv.split(",")]
    totals, ep = defaultdict(dict), None
    for path in csv_paths:  # e.g. one CSV per impl from sweep_blocks.sh
        if not os.path.exists(path):
            print(f"[plot_block_scaling] skipping missing {path}")
            continue
        part, part_ep = load_totals(path, args.num_layers)
        for key, pts in part.items():
            totals[key].update(pts)
        ep = ep or part_ep
    kern_path = (_resolve(args.kernels_csv, args.results_dir) if args.kernels_csv
                 else os.path.splitext(csv_paths[0])[0] + "_kernels.csv")
    kernels = load_kernels(kern_path) if os.path.exists(kern_path) else None
    for src in [totals] + (list(kernels.values()) if kernels else []):
        for key in src:
            src[key] = {c: us for c, us in src[key].items() if c >= args.min_blocks}
    tag = args.tag or f"ep{ep}"
    os.makedirs(args.out_dir, exist_ok=True)

    batches = sorted({B for _, B in totals})
    xs_all = sorted({c for d in totals.values() for c in d})
    note = ("x = CTAs launched for NVLS / A2A-V (4 warps, one token per CTA); "
            "num_sms for DeepEP (one CTA per SM, 8-32 warps). Total = run.py per-layer "
            "(max over ranks)")
    note += ("; dispatch / combine = rank-0 kernel medians (DeepEP incl. epilogues)."
             if kernels else ".")
    panels = [("total", "Total per layer (dispatch + combine)")]
    if kernels:
        panels += [("dispatch", "Dispatch kernels"), ("combine", "Combine kernels")]

    for B in batches:
        fig, axes = plt.subplots(1, len(panels), figsize=(max(4.6 * len(panels), 8.0), 4.6),
                                 sharex=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, (key, title) in zip(axes[0], panels):
            src = totals if key == "total" else kernels[key]
            _style(ax, xs_all)
            _draw(ax, {impl: src.get((impl, B), {}) for impl in IMPLS}, title)
            ax.set_xlabel("CTAs (blocks)", fontsize=8, color=INK_2)
        axes[0][0].set_ylabel("latency per MoE layer (µs, log)", fontsize=8, color=INK_2)
        axes[0][0].legend(fontsize=7, frameon=False, loc="best", labelcolor=INK_2)
        n = -(-B // ep)
        fig.suptitle(f"EP={ep} · global B={B} ({n} tokens/rank) — latency vs CTA count",
                     fontsize=10.5, color=INK, x=0.01, ha="left")
        fig.text(0.01, 0.005, textwrap.fill(note, 150 if len(panels) > 1 else 120),
                 fontsize=6.5, color=MUTED, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        out = os.path.join(args.out_dir, f"{tag}_B{B}.png")
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out}")

    # Overview: totals only, one panel per B.
    fig, axes = plt.subplots(1, len(batches), figsize=(4.0 * len(batches), 3.8),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, B in zip(axes[0], batches):
        _style(ax, xs_all)
        _draw(ax, {impl: totals.get((impl, B), {}) for impl in IMPLS},
              f"B={B} ({-(-B // ep)} tokens/rank)")
        ax.set_xlabel("CTAs (blocks)", fontsize=8, color=INK_2)
    axes[0][0].set_ylabel("latency per MoE layer (µs, log)", fontsize=8, color=INK_2)
    axes[0][0].legend(fontsize=7, frameon=False, loc="best", labelcolor=INK_2)
    fig.suptitle(f"EP={ep} — per-layer dispatch+combine latency vs CTA count",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, textwrap.fill(note, 220), fontsize=6.5, color=MUTED, ha="left",
             va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = os.path.join(args.out_dir, f"{tag}_overview.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
