"""EP=64 microbench (e2e library stack) next to the e2e decode benchmark.

    python3 plot_ep64_vs_e2e.py results/<job> ../ep64_summary.json --out ../microbench_ep64.png

Panels: (1) communication only, B up to 1M; (2) communication + Megatron's vllm_fused_moe
experts at the e2e buffer size; (3) DeepEP minus A2A per decode step for both microbench
modes and the e2e forward time.
"""

import argparse
import collections
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

# Reference palette (dataviz skill), light mode; hue follows the dispatcher.
COLOR = {"nvls": "#2a78d6", "a2a": "#eb6834", "deepep": "#1baf7a"}
LABEL = {"nvls": "AG/RS (nvls)", "a2a": "A2A (nvl_a2a)", "deepep": "DeepEP LL"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DPI = 150
PX = 72 / DPI
LW = 2 * PX * 1.2

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(path):
    """{(impl, num_sms): {B: ms}}"""
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[(r["impl"], int(r["num_sms"]))][int(r["global_B"])] = float(r["latency_us"]) / 1e3
    return d


def series(d, impl, sms=None):
    keys = [k for k in d if k[0] == impl and (sms is None or k[1] == sms)]
    assert len(keys) == 1, (impl, sms, list(d))
    pts = sorted(d[keys[0]].items())
    return [b for b, _ in pts], [v for _, v in pts]


def style(ax, xs, logy):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([(f"{x // 1048576:g}M" if x >= 1048576 else f"{x // 1024:g}k")
                        if x >= 1024 else f"{x:g}" for x in xs])
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", which="major", color=GRID, linewidth=2 * PX)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=4)
    ax.set_xlabel("global batch (decode tokens across 64 GPUs)")


def line(ax, xs, ys, color, dashes=None):
    kw = dict(color=color, linewidth=LW, marker="o", markersize=8 * PX,
              markeredgecolor=SURFACE, markeredgewidth=2 * PX, zorder=3)
    if dashes:
        kw["dashes"] = dashes
    ax.plot(xs, ys, **kw)


def end_labels(ax, ends, gap_px=8 * DPI / 72 * 1.3):
    """Label line ends in secondary ink, spaced so labels never overlap (call after layout)."""
    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform
    pts = sorted((to_disp((x, y))[1], x, y, t) for x, y, t in ends)
    placed = []
    for yd, x, y, t in pts:
        if placed and yd - placed[-1][0] < gap_px:
            yd = placed[-1][0] + gap_px
        placed.append((yd, x, y, t))
    over = max(0.0, placed[-1][0] - ax.bbox.y1)
    for yd, x, y, t in placed:
        _, yl = to_data((to_disp((x, y))[0], yd - over))
        ax.annotate(t, xy=(x, yl), xytext=(8, 0), textcoords="offset points", va="center",
                    ha="left", color=INK_2, fontsize=8, annotation_clip=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("e2e_summary")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    comm = load(os.path.join(args.results_dir, "comm_1m.csv"))
    comm_cap = load(os.path.join(args.results_dir, "comm_cap2048.csv"))
    vllm = load(os.path.join(args.results_dir, "vllm_cap2048.csv"))
    auto_sms = min(k[1] for k in comm if k[0] == "deepep")
    e2e = {r["dispatcher"]: {row["global_batch"]: row["forward_ms"] for row in r["rows"]}
           for r in json.load(open(args.e2e_summary))}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), dpi=DPI)
    labels = []

    ax = axes[0]
    ends = []
    for impl in ("nvls", "a2a"):
        xs, ys = series(comm, impl)
        line(ax, xs, ys, COLOR[impl])
        ends.append((xs[-1], ys[-1], LABEL[impl].split(" ")[0]))
    xs, ys = series(comm, "deepep", auto_sms)
    line(ax, xs, ys, COLOR["deepep"])
    ends.append((xs[-1], ys[-1], f"DeepEP {auto_sms} SMs"))
    xs, ys = series(comm, "deepep", 152)
    line(ax, xs, ys, COLOR["deepep"], dashes=(4, 2))
    ends.append((xs[-1], ys[-1], "DeepEP 152 SMs"))
    style(ax, xs[::2], logy=True)
    ax.set_ylabel("ms per decode step (40 MoE layers)")
    ax.set_title("Communication only (identity expert)", pad=10)
    labels.append((ax, ends))

    ax = axes[1]
    ends = []
    for impl in ("nvls", "a2a", "deepep"):
        xs, ys = series(vllm, impl)
        line(ax, xs, ys, COLOR[impl])
        ends.append((xs[-1], ys[-1], LABEL[impl].split(" ")[0]))
    style(ax, xs, logy=True)
    ax.set_ylabel("ms per decode step (40 MoE layers)")
    ax.set_title("Communication + Megatron vllm_fused_moe experts", pad=10)
    labels.append((ax, ends))

    ax = axes[2]
    ends = []
    e2e_max = max(b for b in e2e["deepep"] if b in e2e["nvl_a2a"])  # same B range as e2e
    variants = [
        ("comm only", comm_cap, None, MUTED, (1.5, 2)),
        ("comm + experts", vllm, None, INK_2, (4, 2)),
    ]
    for name, d, _, color, dash in variants:
        xa, ya = series(d, "a2a")
        xd, yd = series(d, "deepep")
        a = dict(zip(xa, ya))
        xs = [x for x in xd if x in a and x <= e2e_max]
        ys = [dict(zip(xd, yd))[x] - a[x] for x in xs]
        line(ax, xs, ys, color, dashes=dash)
        ends.append((xs[-1], ys[-1], f"micro: {name}"))
    xs = sorted(b for b in e2e["deepep"] if b in e2e["nvl_a2a"] and b >= 64)
    ys = [e2e["deepep"][b] - e2e["nvl_a2a"][b] for b in xs]
    line(ax, xs, ys, INK)
    ends.append((xs[-1], ys[-1], "e2e forward"))
    ax.axhline(0, color=AXIS, linewidth=2 * PX, zorder=1)
    style(ax, [x for x in series(vllm, "a2a")[0] if x <= e2e_max], logy=False)
    ax.set_ylabel("DeepEP minus A2A, ms per decode step")
    ax.set_title("DeepEP vs A2A, e2e batch range (below 0 = DeepEP faster)", pad=10)
    labels.append((ax, ends))

    handles = [Line2D([], [], color=COLOR[i], linewidth=LW, marker="o", markersize=8 * PX,
                      markeredgecolor=SURFACE) for i in ("nvls", "a2a", "deepep")]
    fig.legend(handles, [LABEL[i] for i in ("nvls", "a2a", "deepep")], loc="upper left",
               ncol=3, frameon=False, bbox_to_anchor=(0.005, 0.95), labelcolor=INK_2)
    fig.suptitle("Super v3 MoE at EP=64 (16 GB200 nodes, one NVL72 domain), e2e image: "
                 f"DeepEP 2.5 EPBuffer (auto = {auto_sms} SMs), NCCL 2.32.3; uniform routing, "
                 "graph-replayed, max across ranks", x=0.01, y=0.995, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.88), w_pad=9)
    for ax, ends in labels:
        end_labels(ax, ends)
    fig.savefig(args.out, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
