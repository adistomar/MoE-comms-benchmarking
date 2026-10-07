#!/usr/bin/env python3
# Copyright (c) 2026. Plot the per-phase (dispatch / combine) breakdown.
"""Plot bench_phase_breakdown.py output: dispatch and combine, per strategy.

Usage:
  python3 plot_phase_breakdown.py --csv phase_fine.csv --out phase_breakdown.png \\
      [--dispatch-threshold 4096] [--combine-threshold 1792]

Two panels, one per phase. This is the view that actually justifies having SEPARATE
dispatch and combine thresholds: the two phases cross over at different step sizes, so
between the two crossings the fastest configuration is neither fixed strategy but
multicast dispatch combined with the push combine. The vertical lines mark where the
dynamic kernel flips each half.

`nvls` is the reference fixed strategy; `dynamic[AGv]` / `dynamic[A2Av]` are the dynamic
kernels with their thresholds pinned, so each is exactly the code path the real
dispatcher runs in that regime. `dynamic[AGv]` tracking `nvls` is the correctness check
on the adaptive kernel: it means the switch costs nothing in the regime where the
in-switch collectives already win.
"""

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "nvls": ("NVLS fixed (AGv / RSv)", "tab:orange", "o", "-"),
    "dynamic[AGv]": ("dynamic, NVLS branch", "black", "D", "--"),
    "dynamic[A2Av]": ("dynamic, A2Av branch", "tab:blue", "s", "-"),
    "a2av_push": ("standalone A2Av (separate builder launch)", "tab:purple", "v", ":"),
}


def _ktick(v):
    if v >= 1 << 20:
        return f"{v >> 20}M"
    if v > 999:
        return f"{v / 1024:g}k"
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="phase_fine.csv")
    p.add_argument("--out", default="phase_breakdown.png")
    p.add_argument("--dispatch-threshold", type=int, default=4096)
    p.add_argument("--combine-threshold", type=int, default=1792)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = defaultdict(lambda: defaultdict(dict))  # phase -> impl -> {B: us}
    for r in csv.DictReader(open(args.csv)):
        data[r["phase"]][r["impl"]][int(r["global_B"])] = float(r["latency_us"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, phase, thr, thr_label in (
        (axes[0], "dispatch", args.dispatch_threshold, "dispatch switch"),
        (axes[1], "combine", args.combine_threshold, "combine switch"),
    ):
        series = data.get(phase, {})
        batches = sorted({b for d in series.values() for b in d})
        for impl in ("nvls", "dynamic[AGv]", "dynamic[A2Av]", "a2av_push"):
            if impl not in series:
                continue
            label, color, marker, ls = STYLE[impl]
            pts = sorted(series[impl].items())
            ax.plot([b for b, _ in pts], [v for _, v in pts], marker=marker, ls=ls,
                    lw=2, ms=5, color=color, label=label)
        ax.axvline(thr, color="tab:red", ls=":", lw=1.5)
        ax.annotate(f"{thr_label}\n@ {thr}", xy=(thr, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color="tab:red")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches)
        ax.set_xticklabels([_ktick(b) for b in batches], fontsize=8)
        ax.set_xlabel("step size: tokens across all EP ranks")
        ax.set_ylabel(f"per-layer {phase} latency (us)")
        ax.set_title(f"{phase} (lower is better)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

        # Print the table for this phase.
        print(f"\n### {phase} (per-layer us)\n")
        impls = [i for i in ("nvls", "dynamic[AGv]", "dynamic[A2Av]", "a2av_push") if i in series]
        print("| step tokens | " + " | ".join(impls) + " | A2Av branch vs NVLS |")
        print("|" + "---|" * (2 + len(impls)))
        for b in batches:
            cells = [f"{series[i][b]:.2f}" if b in series[i] else "-" for i in impls]
            base = series.get("nvls", {}).get(b)
            a2 = series.get("dynamic[A2Av]", {}).get(b)
            rel = f"{base / a2:.3f}x" if base and a2 else "-"
            print(f"| {b} | " + " | ".join(cells) + f" | {rel} |")

    fig.suptitle(
        args.title
        or "MoE dispatch and combine, measured separately — EP=4, 4xGB200, 512 experts, "
           "top-k=22, hidden=1024, fp32 expert-output buffer",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
