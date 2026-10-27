#!/usr/bin/env python3
# Copyright (c) 2026. Plot MoE decode-step dispatch/combine latency (all layers).
"""
Plot the full-decode-step (all MoE layers) dispatch/combine latency written by
run.py. Latency is reported in MILLISECONDS (max across ranks).

  --x B         latency vs global batch size, one line per implementation.
  --x num_sms   latency vs DeepEP num_sms (one line per batch size), with an
                NVLS SM-independent reference band+line.

The title and subtitle (model, MoE layers per step, communication only or with vLLM experts,
experts, top-k, dispatched row width) come from the CSV's model / num_layers / expert columns
(written by run.py; CSVs without an expert column are communication only).

Depends only on matplotlib + stdlib csv + models.py (no torch).

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

from models import get_model  # noqa: E402

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

IMPL_NAME = {"deepep": "DeepEP A2Av", "nvls": "Megatron AGv/RSv", "a2a": "Megatron A2Av",
             "nccl": "NCCL (AllGather)",
             "hybrid-global": "Hybrid (global)",
             "hybrid-local": "Hybrid (local)",
             "hybrid-global_imbalance": "Hybrid (global + imbalance term)",
             "hybrid-mc_rs": "Hybrid: multicast + ld_reduce",
             "hybrid-uc_pull": "Hybrid: unicast + pull",
             "hybrid-mc_pull": "Hybrid: multicast + pull",
             "hybrid-uc_rs": "Hybrid: unicast + ld_reduce"}
IMPL_COLOR = {"deepep": "tab:blue", "nvls": "tab:orange", "nccl": "tab:green",
              "a2a": "tab:red", "hybrid-global": "black", "hybrid-local": "tab:green",
              "hybrid-global_imbalance": "tab:purple", "hybrid-mc_rs": "tab:olive",
              "hybrid-uc_pull": "tab:pink", "hybrid-mc_pull": "tab:cyan",
              "hybrid-uc_rs": "tab:brown"}
# Mode names in older CSVs: hybrid (= global, then called auto), hybrid-per_rank / hybrid-rank
# (= local), hybrid-kappa (= global_imbalance).
for _old, _new in (("hybrid", "hybrid-global"), ("hybrid-per_rank", "hybrid-local"),
                   ("hybrid-rank", "hybrid-local"), ("hybrid-kappa", "hybrid-global_imbalance")):
    IMPL_NAME[_old], IMPL_COLOR[_old] = IMPL_NAME[_new], IMPL_COLOR[_new]
PHASE_WHAT = {"dispatch": "dispatch", "combine": "combine"}  # decode_step: see main()
# A second DeepEP line (fixed num_sms next to auto) gets its own color.
DEEPEP_FIXED_COLOR = "tab:purple"


def _subtitle(ep, spec, gpu):
    """Subtitle line. EP size and node count are derived from the data (per_rank_counts
    length in load()), the GPU from the CSV's gpu column, and the MoE shape from the model
    preset, so the plot reflects the actual run rather than a hardcoded config."""
    shape = (f"{spec.num_experts} experts · top-k={spec.topk} · "
             f"latent={spec.moe_latent_size} (hidden={spec.hidden_size}) · bf16 · "
             f"CUDA-graph timing")
    if not ep:
        return shape
    nodes = max(1, ep // 4)  # 4 GPUs per node
    node_str = "1 node" if nodes == 1 else f"{nodes} nodes"
    return f"EP={ep} · {node_str} (4×{gpu}/node) · {shape}"


def load(path, phase="decode_step", impls=None):
    """-> (rows, ep, model, num_layers, expert, gpu, imbalance) for the rows of `phase` (and
    `impls`); imbalance is run.py's --imbalance ("1" for CSVs without the column)."""
    rows = []
    ep = None  # world size = EP = #ranks, inferred from the per_rank_counts list length
    models, layers, experts, gpus, imbalances = set(), set(), set(), set(), set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] != phase or (impls and r["impl"] not in impls):
                continue
            if ep is None:
                parts = [c for c in r.get("per_rank_counts", "").strip().strip("[]").split(",")
                         if c.strip() != ""]
                if parts:
                    ep = len(parts)
            models.add(r["model"])
            layers.add(int(r["num_layers"]))
            experts.add(r.get("expert") or "identity")
            gpus.add(r.get("gpu") or "GB200")  # CSVs before the gpu column: GB200 runs
            imbalances.add(float(r.get("imbalance") or 1.0))
            rows.append(dict(impl=r["impl"], B=int(r["global_B"]),
                             num_sms=int(r.get("num_sms", -1)),
                             sms_setting=r.get("sms_setting") or "",
                             ms=float(r["latency_us"]) / 1000.0))  # us -> ms
    if len(models) != 1 or len(layers) != 1 or len(experts) != 1:
        raise SystemExit(f"{path}: expected one model / num_layers / expert, got "
                         f"models={sorted(models)}, num_layers={sorted(layers)}, "
                         f"expert={sorted(experts)}")
    imbalance = "/".join(f"{x:g}" for x in sorted(imbalances))
    return (rows, ep, models.pop(), layers.pop(), experts.pop(), "/".join(sorted(gpus)),
            imbalance)


def _ktick(v):
    """Batch-size tick label in binary units: 1024->'1k', 131072->'128k', 1048576->'1M'."""
    if v >= 1 << 20:
        return f"{v >> 20}M"
    if v > 999:
        # Non-power-of-two sweeps (e.g. 1448, 2896): one decimal, so ticks stay distinct.
        return f"{v >> 10}k" if v % 1024 == 0 else f"{v / 1024:.1f}k"
    return str(v)


def _label_ends(ax, ends, min_gap_pt=10):
    """Label each line's last value to its right, nudged apart vertically so close values
    (e.g. all impls at B=1M) do not overlap. Labels stack downward from the highest value, so
    a cluster at the top of the axes never runs into the title. ends = [(x, y, color)]."""
    fig = ax.figure
    fig.canvas.draw()  # finalize limits/transforms before measuring
    px_to_pt = 72.0 / fig.dpi
    ys_pt = sorted(((ax.transData.transform((x, y))[1] * px_to_pt, x, y, c) for x, y, c in ends),
                   reverse=True)
    placed = []
    for y_pt, x, y, c in ys_pt:
        target = y_pt if not placed else min(y_pt, placed[-1] - min_gap_pt)
        placed.append(target)
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, target - y_pt), va="center", fontsize=8, color=c)


def _series(rows):
    """[(label, color, rows)] in legend order (IMPL_NAME order, unknown impls last). DeepEP
    run at several num_sms gets one line per value, labelled with its SM count and whether
    it was DeepEP's theoretical ('auto') count."""
    impls = {r["impl"] for r in rows}
    out = []
    for impl in [i for i in IMPL_NAME if i in impls] + sorted(impls - IMPL_NAME.keys()):
        mine = [r for r in rows if r["impl"] == impl]
        name, color = IMPL_NAME.get(impl, impl), IMPL_COLOR.get(impl, "gray")
        variants = sorted({(r["sms_setting"] != "auto", r["num_sms"]) for r in mine})
        if impl != "deepep" or len(variants) == 1:
            if impl == "deepep" and variants[0][0]:  # one fixed SM count: show it
                name = f"{name} ({variants[0][1]} SMs)"
            out.append((name, color, mine))
            continue
        for i, (fixed, sms) in enumerate(variants):
            setting = f"{sms} SMs" if fixed else f"auto: {sms} SMs"
            out.append((f"{name} ({setting})", DEEPEP_FIXED_COLOR if i else color,
                        [r for r in mine if (r["sms_setting"] != "auto", r["num_sms"])
                         == (fixed, sms)]))
    return out


def plot_vs_B(rows, ax):
    ends = []
    for label, color, mine in _series(rows):
        pts = sorted((r["B"], r["ms"]) for r in mine)
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, marker="o", lw=2, ms=6, color=color, label=label)
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
    p.add_argument("--phase", choices=["decode_step", "dispatch", "combine"],
                   default="decode_step",
                   help="which timed phase to plot (run.py --phases); dispatch / combine are "
                        "that collective alone, all layers of a step")
    p.add_argument("--impls", default=None,
                   help="comma-separated impl names to plot (CSV impl column, e.g. "
                        "nvls,a2a,hybrid); default: all")
    p.add_argument("--powers-of-two-only", action="store_true",
                   help="plot only power-of-two B, dropping the in-between points (e.g. the "
                        "sqrt(2) steps of a crossover calibration sweep)")
    p.add_argument("--title", default=None)
    p.add_argument("--results-dir", default=os.path.join(REPO_DIR, "results"),
                   help="where bare --csv names are read from")
    p.add_argument("--plots-dir", default=os.path.join(REPO_DIR, "plots"),
                   help="where bare --out names are written (created if missing)")
    args = p.parse_args()
    csv_path = args.csv if os.path.dirname(args.csv) else os.path.join(args.results_dir, args.csv)
    out_path = args.out if os.path.dirname(args.out) else os.path.join(args.plots_dir, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    impls = set(args.impls.split(",")) if args.impls else None
    rows, ep, model, num_layers, expert, gpu, imbalance = load(csv_path, args.phase, impls)
    if args.powers_of_two_only:
        rows = [r for r in rows if r["B"] & (r["B"] - 1) == 0]
    spec = get_model(model)
    subtitle = _subtitle(ep, spec, gpu)
    # run.py --imbalance: rank 0 holds rho x the mean token count.
    rho = "" if imbalance == "1" else f" · imbalance ρ={imbalance}"
    what = ("dispatch+combine" if expert == "identity"
            else f"dispatch+{ {'vllm': 'vLLM'}.get(expert, expert)} experts+combine")
    what = PHASE_WHAT.get(args.phase, what)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Impl names live in the legend; keep the title short so it never clips.
    ends = []
    if args.x == "num_sms":
        plot_vs_sms(rows, ax)
        default_title = (f"{spec.name} MoE decode {what} vs DeepEP num_sms — "
                         f"{num_layers} MoE layers/step{rho}\n{subtitle}")
    else:
        ends = plot_vs_B(rows, ax)
        default_title = (f"{spec.name} MoE decode {what} latency — "
                         f"{num_layers} MoE layers/step{rho}\n{subtitle}")
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
