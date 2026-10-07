# Plot the NVLS AGv bus-bandwidth saturation sweep (bench_agv_bw.py output).
"""Two panels from agv_bw.csv:
  left : bus bandwidth vs output message size (per row width H), 148 CTAs.
  right: bus bandwidth vs CTA count at a fixed 1 GiB (bandwidth-bound) message.
Reference lines: 900 GB/s (B200 NVLink-5 per-direction peak) and 675 GB/s
(= (g-1)/g * 900, the multicast-AG ingress-bound ceiling at g=4 — a multimem.st
delivers the sender's own copy back through the switch, so per-rank ingress
carries the full output while bus BW counts only (g-1)/g of it).
"""

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--csv", default="agv_bw.csv")
p.add_argument("--out", default="agv_bw.png")
args = p.parse_args()

rows = list(csv.DictReader(open(args.csv)))
msg = defaultdict(list)   # hidden -> [(out_mb, busbw)]
cta = defaultdict(list)   # hidden -> [(num_blocks, busbw)]
for r in rows:
    h, bw = int(r["hidden"]), float(r["busbw_gbs"])
    if r["sweep"] == "msg":
        msg[h].append((int(r["out_mb"]), bw))
    else:
        cta[h].append((int(r["num_blocks"]), bw))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

for h in sorted(msg):
    xs, ys = zip(*sorted(msg[h]))
    thr = min(-(-h * 2 // 16) and (1 << (max(1, -(-h * 2 // 16)) - 1).bit_length()), 1024)
    ax1.plot(xs, ys, marker="o", ms=4, label=f"H={h} ({thr} thr/CTA)")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("output message size (MiB)")
ax1.set_ylabel("bus bandwidth (GB/s)")
ax1.set_title("NVLS AGv bus BW vs message size (148 CTAs, 4×GB200)")

for h in sorted(cta):
    xs, ys = zip(*sorted(cta[h]))
    ax2.plot(xs, ys, marker="s", ms=4, label=f"H={h}")
ax2.set_xlabel("CTA count (max_num_blocks)")
ax2.set_ylabel("bus bandwidth (GB/s)")
ax2.set_title("NVLS AGv bus BW vs CTA count (1 GiB output)")

for ax in (ax1, ax2):
    ax.axhline(900, color="k", ls="--", lw=1)
    ax.axhline(675, color="crimson", ls=":", lw=1.2)
    ax.text(0.02, 0.965, "900 GB/s: NVLink-5 per-direction peak", transform=ax.transAxes,
            fontsize=8, va="top")
    ax.text(0.02, 0.90, "675 GB/s: multicast-AG ingress ceiling (¾·900)",
            transform=ax.transAxes, fontsize=8, va="top", color="crimson")
    ax.set_ylim(0, 950)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"wrote {args.out}")
