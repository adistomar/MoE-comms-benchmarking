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

Bare file names are resolved against --results-dir (CSV input, default results/) and
--plots-dir (PNG output, default plots/); paths with a directory are used as given.
"""

import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

IMPL_NAME = {"deepep": "DeepEP A2Av", "nvls": "Megatron AGv/RSv", "a2a": "Megatron A2Av",
             "nccl": "NCCL (AllGather)", "naive_agv": "Naive unicast AGv / NVLS RSv"}
IMPL_COLOR = {"deepep": "tab:blue", "nvls": "tab:orange", "nccl": "tab:green",
              "a2a": "tab:red", "naive_agv": "tab:purple"}


def _subtitle(ep):
    """Subtitle line. EP size and node count are derived from the data (per_rank_counts
    length in load()), so the plot reflects the actual run rather than a hardcoded config."""
    if not ep:
        return "512 experts · top-k=22 · hidden=1024 · bf16 · CUDA-graph timing"
    nodes = max(1, ep // 4)  # 4 GB300 GPUs per node
    node_str = "1 node" if nodes == 1 else f"{nodes} nodes"
    return (f"EP={ep} · {node_str} (4×B300/node) · 512 experts · top-k=22 · "
            f"hidden=1024 · bf16 · CUDA-graph timing")


def load(path, phase="decode_step"):
    rows = []
    ep = None  # world size = EP = #ranks, inferred from the per_rank_counts list length
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] != phase:
                continue
            if ep is None:
                parts = [c for c in r.get("per_rank_counts", "").strip().strip("[]").split(",")
                         if c.strip() != ""]
                if parts:
                    ep = len(parts)
            rows.append(dict(impl=r["impl"], B=int(r["global_B"]),
                             num_sms=int(r.get("num_sms", -1)),
                             ms=float(r["latency_us"]) / 1000.0))  # us -> ms
    return rows, ep


def _ktick(v):
    """Batch-size tick label in binary units: 1024->'1k', 131072->'128k', 1048576->'1M'."""
    if v >= 1 << 20:
        return f"{v >> 20}M"
    if v > 999:
        return f"{v >> 10}k"
    return str(v)


def _label_ends(ax, ends, min_gap_pt=10):
    """Label each line's last value to its right, nudged apart vertically so close values
    (e.g. all impls at B=1M) do not overlap. ends = [(x, y, color)]."""
    fig = ax.figure
    fig.canvas.draw()  # finalize limits/transforms before measuring
    px_to_pt = 72.0 / fig.dpi
    ys_pt = sorted((ax.transData.transform((x, y))[1] * px_to_pt, x, y, c) for x, y, c in ends)
    placed = []
    for y_pt, x, y, c in ys_pt:
        target = y_pt if not placed else max(y_pt, placed[-1] + min_gap_pt)
        placed.append(target)
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, target - y_pt), va="center", fontsize=8, color=c)


def plot_vs_B(rows, ax):
    impls = {r["impl"] for r in rows}
    ends = []
    # Legend order follows IMPL_NAME (DeepEP first); unknown impls go last.
    for impl in [i for i in IMPL_NAME if i in impls] + sorted(impls - IMPL_NAME.keys()):
        pts = sorted((r["B"], r["ms"]) for r in rows if r["impl"] == impl)
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        color = IMPL_COLOR.get(impl, "gray")
        ax.plot(xs, ys, marker="o", lw=2, ms=6, color=color, label=IMPL_NAME.get(impl, impl))
        ends.append((xs[-1], ys[-1], color))
    allB = sorted({r["B"] for r in rows})
    ax.set_xscale("log", base=2)
    # Room on the right (and above) for the end-of-line value labels.
    ax.set_xlim(allB[0] / 1.5, allB[-1] * 4)
    ax.margins(y=0.08)
    ax.set_xticks(allB)
    # ha="center" centers each rotated label under its tick (ha="right" hangs it to the left).
    ax.set_xticklabels([_ktick(b) for b in allB], rotation=45, ha="center", fontsize=8)
    ax.set_xlabel("global batch size B  (decode tokens across all EP ranks)")
    return ends


def plot_vs_sms(rows, ax):
    dee = [r for r in rows if r["impl"] == "deepep"]
    Bs = sorted({r["B"] for r in dee})
    sms = sorted({r["num_sms"] for r in dee})
    for i, B in enumerate(Bs):
        pts = sorted((r["num_sms"], r["ms"]) for r in dee if r["B"] == B)
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, marker="o", lw=1.8, ms=5,
                color=cm.viridis(i / max(1, len(Bs) - 1)), label=f"DeepEP  B={B}")
    # NVLS (own CTA cap, default_num_blocks) and NCCL (NCCL-internal grid) have no swept num_sms axis,
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
    p.add_argument("--phase", choices=["decode_step", "dispatch"], default="decode_step",
                   help="which timed phase to plot (CSV phase column)")
    p.add_argument("--num-layers", type=int, default=88, help="for the title only")
    p.add_argument("--title", default=None)
    p.add_argument("--results-dir", default=os.path.join(REPO_DIR, "results"),
                   help="where bare --csv names are read from")
    p.add_argument("--plots-dir", default=os.path.join(REPO_DIR, "plots"),
                   help="where bare --out names are written (created if missing)")
    args = p.parse_args()
    csv_path = args.csv if os.path.dirname(args.csv) else os.path.join(args.results_dir, args.csv)
    out_path = args.out if os.path.dirname(args.out) else os.path.join(args.plots_dir, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    rows, ep = load(csv_path, args.phase)
    subtitle = _subtitle(ep)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Impl names live in the legend; keep the title short so it never clips.
    ends = []
    if args.x == "num_sms":
        plot_vs_sms(rows, ax)
        default_title = (f"MoE decode dispatch+combine vs DeepEP num_sms — "
                         f"{args.num_layers} MoE layers/step\n{subtitle}")
    else:
        ends = plot_vs_B(rows, ax)
        what = "dispatch-only" if args.phase == "dispatch" else "dispatch+combine"
        default_title = (f"MoE decode {what} latency — "
                         f"{args.num_layers} MoE layers/step\n{subtitle}")
    ax.set_ylabel("latency per decode step (ms)")
    # Log y: latency spans ~1 ms (NVLS) to ~40 ms (NCCL), so a linear axis would squash the
    # fast impls. Log keeps all curves legible.
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(args.title or default_title, fontsize=10)
    fig.tight_layout()
    _label_ends(ax, ends)  # after layout, so label offsets use the final axes geometry
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
