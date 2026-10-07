#!/usr/bin/env python3
# Copyright (c) 2026. Per-phase (dispatch vs combine) breakdown: nvls vs dynamic.
"""Time DISPATCH and COMBINE separately for the NVLS and dynamic collectives.

run.py times dispatch+combine together, which is the right end-to-end number but hides
where a regression lives. This script replays each phase on its own -- `--num-layers`
back-to-back dispatches in one CUDA graph, then the same for combines -- so the NVLS
path can be compared against the dynamic kernels' NVLS branch phase by phase.

The dynamic bencher is instantiated three times with pinned thresholds, so every column
is the same code path the real dispatcher would run:
  dynamic[AGv]  thresholds = huge  -> multicast dispatch + reduce-scatter-v
  dynamic[A2Av] thresholds = 0     -> unicast dispatch (+ inline index build) + push
and compared against `nvls`, the reference fixed strategy.

Usage:
  torchrun --nproc_per_node=4 bench_phase_breakdown.py \\
      --batch-sizes 64,256,512,1024,2048,8192 --out-dtype fp32 --num-layers 88
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from common import Config, all_rank_counts, init_distributed, make_inputs, time_region  # noqa: E402

PIN_AGV = 1 << 30
PIN_A2AV = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", default="64,256,512,1024,2048,8192",
                   help="comma-separated GLOBAL token counts")
    p.add_argument("--num-layers", type=int, default=88)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out-dtype", choices=["bf16", "fp32"], default="fp32",
                   help="expert-output buffer dtype (fp32 = Megatron's actual config)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None, help="CSV output path (rank 0)")
    return p.parse_args()


def main():
    args = parse_args()
    group, rank, world, local_rank = init_distributed()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    cfg = Config(ep_size=world, rank=rank, local_rank=local_rank, seed=args.seed,
                 per_rank_cap=-(-max(batch_sizes) // world))
    device = torch.device("cuda", local_rank)
    out_dtype = torch.float32 if args.out_dtype == "fp32" else torch.bfloat16

    from bench_a2av import A2AVPushBencher
    from bench_dynamic import DynamicBencher
    from bench_nvls import NVLSBencher

    cols = []
    cols.append(("nvls", NVLSBencher(cfg, group, out_dtype=out_dtype)))
    cols.append(("dynamic[AGv]", DynamicBencher(cfg, group, PIN_AGV, PIN_AGV, out_dtype)))
    cols.append(("dynamic[A2Av]", DynamicBencher(cfg, group, PIN_A2AV, PIN_A2AV, out_dtype)))
    # The standalone A2AV implementation, for a like-for-like check against the dynamic
    # kernel's A2AV branch. They run the same collectives, but differ structurally: this
    # one launches the send-index builder as its OWN kernel before dispatch, while the
    # dynamic kernel folds the builder into the dispatch kernel. Splitting by phase says
    # whether any gap between them lives in dispatch (the builder) or combine.
    # NOTE it is bf16-only, so run with --out-dtype bf16 for a controlled comparison.
    cols.append(("a2av_push", A2AVPushBencher(cfg, group)))

    if rank == 0:
        print(f"# per-phase breakdown  EP={world} experts={cfg.num_experts} "
              f"topk={cfg.topk} hidden={cfg.hidden} out_dtype={args.out_dtype} "
              f"layers={args.num_layers} reps={args.reps}", flush=True)

    dist.barrier(group)
    for _, b in cols:
        b.build()
    dist.barrier(group)

    nl = args.num_layers
    rows = []
    for B in batch_sizes:
        hidden, topk_idx, topk_weights = make_inputs(cfg, B, device)
        for name, b in cols:
            b.setup_batch(hidden, topk_idx, topk_weights)
            # Metadata is published once per step and is routing-independent; run it
            # outside the timed region so the phase numbers are pure data movement.
            b.metadata()
            torch.cuda.synchronize()

            def _dispatch_only():
                for _ in range(nl):
                    b.dispatch()

            def _combine_only():
                for _ in range(nl):
                    b.combine()

            # time_region -> (local_us, cross_rank_max_us); the max is the critical path.
            _, d_us = time_region(_dispatch_only, group, args.warmup, args.reps, True)
            _, c_us = time_region(_combine_only, group, args.warmup, args.reps, True)
            rows.append((B, name, d_us / nl, c_us / nl))
            if rank == 0:
                print(f"  B={B:<7d} {name:<14s} dispatch={d_us / nl:8.2f}us  "
                      f"combine={c_us / nl:8.2f}us  total={(d_us + c_us) / nl:8.2f}us",
                      flush=True)

    if rank == 0:
        base = {(B, ph): v for B, n, d, c in rows if n == "nvls"
                for ph, v in (("dispatch", d), ("combine", c))}
        print("\n| global B | tokens/rank | impl | dispatch us | combine us | total us "
              "| dispatch vs nvls | combine vs nvls |")
        print("|" + "---|" * 8)
        for B, n, d, c in rows:
            dv = f"{base[(B, 'dispatch')] / d:.3f}x" if (B, "dispatch") in base else "-"
            cv = f"{base[(B, 'combine')] / c:.3f}x" if (B, "combine") in base else "-"
            print(f"| {B} | {B // world} | {n} | {d:.2f} | {c:.2f} | {d + c:.2f} "
                  f"| {dv} | {cv} |")
        if args.out:
            with open(args.out, "w") as f:
                f.write("global_B,impl,phase,latency_us\n")
                for B, n, d, c in rows:
                    f.write(f"{B},{n},dispatch,{d:.4f}\n")
                    f.write(f"{B},{n},combine,{c:.4f}\n")
            print(f"\n# wrote {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
