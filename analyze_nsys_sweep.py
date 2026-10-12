#!/usr/bin/env python3
"""Per-config kernel breakdown of an nsys sweep (sweep_ctas_nsys.sh), from its SQLite export.

run.py opens one NVTX range "<impl> sms=<n> B=<B>" around each timed config. Each kernel is
assigned to the range containing its launching API call (for CUDA-graph node kernels that is
the cudaGraphLaunch). Per config this prints the median duration of each dispatch/combine
kernel (median over warmup + replays, so first-launch outliers drop out) with its launch
shape, and the per-layer sum. Stdlib only:

  python3 analyze_nsys_sweep.py profiles/ctasweep_ep4_<jobid>.sqlite [--csv out.csv]
"""

import argparse
import bisect
import csv
import re
import sqlite3
import statistics
from collections import defaultdict

# (phase label, kernel-name substring); first match wins, so epilogues precede their mains.
PHASES = [
    ("a2a_dispatch", "_a2a_dispatch_v_kernel"),
    ("a2a_combine", "_a2a_combine_v_kernel"),
    ("nvls_dispatch", "_multimem_all_gatherv_3tensor_kernel"),
    ("nvls_combine", "_multimem_reduce_scatter_v_kernel"),
    ("dispatch_epilogue", "dispatch_copy_epilogue"),
    ("combine_epilogue", "combine_reduce_epilogue"),
    ("dispatch", "dispatch_impl"),
    ("combine", "combine_impl"),
]
RANGE_RE = re.compile(r"^(\w+) sms=(-?\d+) B=(\d+)$")


def phase_of(name):
    for phase, sub in PHASES:
        if sub in name:
            return phase
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--csv", default=None, help="optional CSV of the per-config breakdown")
    args = p.parse_args()
    con = sqlite3.connect(args.sqlite)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))

    nvtx_cols = {r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text_expr = "text, textId" if "textId" in nvtx_cols else "text, NULL"
    ranges = []  # (start, end, label) for push/pop ranges matching run.py's format
    for start, end, text, text_id in con.execute(
            f"SELECT start, end, {text_expr} FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        label = text if text is not None else strings.get(text_id)
        if label and RANGE_RE.match(label):
            ranges.append((start, end, label))
    ranges.sort()
    starts = [r[0] for r in ranges]

    rows = con.execute("""
        SELECT k.start, k.end, k.shortName, k.gridX, k.blockX, k.registersPerThread,
               k.staticSharedMemory + k.dynamicSharedMemory, r.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId""")
    durs = defaultdict(list)  # (label, phase) -> [us]
    shape = {}                # (label, phase) -> (grid, block, regs, smem)
    for kstart, kend, name_id, grid, block, regs, smem, api_start in rows:
        phase = phase_of(strings.get(name_id, ""))
        if phase is None:
            continue
        i = bisect.bisect_right(starts, api_start) - 1
        if i < 0 or api_start > ranges[i][1]:
            continue
        label = ranges[i][2]
        durs[(label, phase)].append((kend - kstart) / 1e3)
        shape[(label, phase)] = (grid, block, regs, smem)

    configs = sorted({lab for lab, _ in durs},
                     key=lambda s: (RANGE_RE.match(s).group(1), int(RANGE_RE.match(s).group(3)),
                                    int(RANGE_RE.match(s).group(2))))
    out = []
    for label in configs:
        impl, sms, B = RANGE_RE.match(label).groups()
        parts = {ph: statistics.median(durs[(label, ph)])
                 for ph, _ in PHASES if (label, ph) in durs}
        total = sum(parts.values())
        desc = "  ".join(
            f"{ph}={us:8.1f} [grid={shape[(label, ph)][0]} blk={shape[(label, ph)][1]} "
            f"regs={shape[(label, ph)][2]}]" for ph, us in parts.items())
        print(f"{impl:6s} B={int(B):7d} ctas={int(sms):4d}  sum={total:8.1f} us/layer   {desc}")
        out.append(dict(impl=impl, B=int(B), ctas=int(sms), sum_us=round(total, 2),
                        **{f"{ph}_us": round(us, 2) for ph, us in parts.items()},
                        **{f"{ph}_regs": shape[(label, ph)][2] for ph in parts}))
    if args.csv and out:
        keys = list(dict.fromkeys(k for r in out for k in r))
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
