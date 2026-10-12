#!/usr/bin/env python3
# Copyright (c) 2026. Plot the DeepEP dispatch-warp sweep (sweep_deepep_warps.sh).
"""
Per-layer DeepEP latency vs EP_DISPATCH_TOTAL_WARPS (total dispatch warps per GPU; upstream
DeepEP hardcodes 512), one line per num_sms, one panel per global batch size B. Optionally
overlays the best NVLS / A2A-V latency over all block counts from a block sweep, as dashed
reference lines, to answer "does any DeepEP setting beat them?".

  python3 plot_deepep_warps.py --prefix deepwarps_ep4_<jobid> \
      --ref-csv blocks_ep4_<jobid>_nvls.csv,blocks_ep4_<jobid>_a2a.csv
  -> plots/block-size-scaling/deepep_warps_ep4.png
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_block_scaling import (  # noqa: E402
    AXIS, GRID, IMPL_COLOR, IMPL_NAME, INK, INK_2, MUTED, REPO_DIR, SURFACE, _resolve,
    load_totals)
from matplotlib.ticker import (  # noqa: E402
    FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator)

# Ordinal blue ramp (dataviz reference sequential steps 250 / 450 / 650): more SMs = darker.
SMS_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
DEFAULT_WARPS = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="results/<prefix>_w<warps>.csv files")
    p.add_argument("--ref-csv", default=None,
                   help="comma-separated block-sweep CSVs whose best latency per B is overlaid")
    p.add_argument("--num-layers", type=int, default=88)
    p.add_argument("--results-dir", default=os.path.join(REPO_DIR, "results"))
    p.add_argument("--out-dir", default=os.path.join(REPO_DIR, "plots", "block-size-scaling"))
    args = p.parse_args()

    data, ep = {}, None  # (B, sms) -> {warps: us}
    for path in glob.glob(os.path.join(args.results_dir, f"{args.prefix}_w*.csv")):
        warps = int(re.search(r"_w(\d+)\.csv$", path).group(1))
        part, ep = load_totals(path, args.num_layers)
        for (impl, B), pts in part.items():
            for sms, us in pts.items():
                data.setdefault((B, sms), {})[warps] = us
    if not data:
        raise SystemExit(f"no results/{args.prefix}_w*.csv files")
    refs = {}  # (impl, B) -> best us over block counts
    for path in (args.ref_csv.split(",") if args.ref_csv else []):
        path = _resolve(path, args.results_dir)
        if os.path.exists(path):
            for (impl, B), pts in load_totals(path, args.num_layers)[0].items():
                refs[(impl, B)] = min(pts.values())

    batches = sorted({B for B, _ in data})
    sms_list = sorted({s for _, s in data})
    warps_all = sorted({w for pts in data.values() for w in pts})
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(batches), figsize=(4.0 * len(batches), 3.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, B in zip(axes[0], batches):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(warps_all))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(w) for w in warps_all], fontsize=7, color=MUTED)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=7, colors=MUTED, which="both")
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.axvline(DEFAULT_WARPS, color=AXIS, linewidth=0.6, linestyle=":", zorder=0)
        ax.text(DEFAULT_WARPS, 1.0, " DeepEP default", transform=ax.get_xaxis_transform(),
                fontsize=6, color=MUTED, va="bottom")
        for color, sms in zip(SMS_COLORS[-len(sms_list):], sms_list):
            pts = sorted(data.get((B, sms), {}).items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linewidth=1.2, marker="o", markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=f"DeepEP, {sms} SMs", zorder=3)
        for impl in ("nvls", "a2a"):
            if (impl, B) in refs:
                ax.axhline(refs[(impl, B)], color=IMPL_COLOR[impl], linewidth=1.0,
                           linestyle="--", label=f"best {IMPL_NAME[impl]} (any block count)")
        ax.set_title(f"B={B} ({-(-B // ep)} tokens/rank)", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("EP_DISPATCH_TOTAL_WARPS (dispatch warps per GPU)", fontsize=8,
                      color=INK_2)
    axes[0][0].set_ylabel("latency per MoE layer (µs, log)", fontsize=8, color=INK_2)
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="best", labelcolor=INK_2)
    fig.suptitle(f"EP={ep} — DeepEP latency vs dispatch warps (combine fixed at 32 warps/SM)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Warps per CTA = min(ceil(total / num_sms), 28) + 4 notify warps, so "
             "small num_sms saturate early (32 SMs: 28 warps from 1024 on).",
             fontsize=6.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = os.path.join(args.out_dir, f"deepep_warps_ep{ep}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
