#!/usr/bin/env python3
# Copyright (c) 2026. Plot MoE decode-step dispatch/combine latency (all layers).
"""
Plot the full-decode-step (all MoE layers) dispatch/combine latency written by
run.py. Latency is reported in MILLISECONDS (max across ranks).

  --x B         latency vs global batch size, one line per implementation.
  --x num_sms   latency vs DeepEP num_sms (one line per batch size), with an
                NVLS SM-independent reference band+line.

Depends only on matplotlib + stdlib csv (no torch).

  python3 plot_results.py --csv results.csv --out results.png
  python3 plot_results.py --csv results_sms.csv --x num_sms --out results_sms.png
"""

import argparse
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

IMPL_NAME = {"deepep": "DeepEP-v2 (A2A)", "nvls": "NVLS (AGv/RSv)",
             "nccl": "NCCL (AllGather)"}
IMPL_COLOR = {"deepep": "tab:blue", "nvls": "tab:orange", "nccl": "tab:green"}
_SUBTITLE = ("EP=4 on 4×B200 · 512 experts · top-k=22 · hidden=1024 · bf16 · "
             "CUDA-graph timing")


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] != "decode_step":
                continue
            rows.append(dict(impl=r["impl"], B=int(r["global_B"]),
                             num_sms=int(r.get("num_sms", -1)),
                             ms=float(r["latency_us"]) / 1000.0))  # us -> ms
    return rows


def _ktick(v):
    """Batch-size tick label: values over 999 in binary-k (1024->'1k', 2048->'2k', ...)."""
    return f"{v // 1024}k" if v > 999 else str(v)


def plot_vs_B(rows, ax):
    for impl in sorted({r["impl"] for r in rows}):
        pts = sorted((r["B"], r["ms"]) for r in rows if r["impl"] == impl)
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, marker="o", lw=2, ms=6, color=IMPL_COLOR.get(impl, "gray"),
                label=IMPL_NAME.get(impl, impl))
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=IMPL_COLOR.get(impl, "gray"))
    allB = sorted({r["B"] for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(allB)
    ax.set_xticklabels([_ktick(b) for b in allB])
    ax.set_xlabel("global batch size B  (decode tokens across all 4 EP ranks)")


def plot_vs_sms(rows, ax):
    dee = [r for r in rows if r["impl"] == "deepep"]
    Bs = sorted({r["B"] for r in dee})
    sms = sorted({r["num_sms"] for r in dee})
    for i, B in enumerate(Bs):
        pts = sorted((r["num_sms"], r["ms"]) for r in dee if r["B"] == B)
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, marker="o", lw=1.8, ms=5,
                color=cm.viridis(i / max(1, len(Bs) - 1)), label=f"DeepEP  B={B}")
    # NVLS (fixed 148-block cap) and NCCL (NCCL-internal grid) have no swept num_sms axis,
    # so each is drawn as a horizontal reference band+line across the DeepEP sweep.
    for ref in ("nvls", "nccl"):
        vals = [r["ms"] for r in rows if r["impl"] == ref]
        if vals:
            m = statistics.mean(vals)
            ax.axhspan(min(vals), max(vals), color=IMPL_COLOR[ref], alpha=0.12)
            ax.axhline(m, color=IMPL_COLOR[ref], ls="--", lw=2,
                       label=f"{IMPL_NAME[ref]} (num_sms-independent, ~{m:.2f} ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sms)
    ax.set_xticklabels([str(s) for s in sms])
    ax.set_xlabel("DeepEP num_sms  (SMs for dispatch/combine; rightmost = device max)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--out", default="results.png")
    p.add_argument("--x", choices=["B", "num_sms"], default="B")
    p.add_argument("--num-layers", type=int, default=88, help="for the title only")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rows = load(args.csv)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Impl names live in the legend; keep the title short so it never clips.
    if args.x == "num_sms":
        plot_vs_sms(rows, ax)
        default_title = (f"MoE decode dispatch+combine vs DeepEP num_sms — "
                         f"{args.num_layers} MoE layers/step\n{_SUBTITLE}")
    else:
        plot_vs_B(rows, ax)
        default_title = (f"MoE decode dispatch+combine latency — "
                         f"{args.num_layers} MoE layers/step\n{_SUBTITLE}")
    ax.set_ylabel("latency per decode step (ms)")
    # Log y: latency spans ~1 ms (NVLS) to ~40 ms (NCCL), so a linear axis would squash the
    # fast impls. Log keeps all curves legible.
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_title(args.title or default_title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
