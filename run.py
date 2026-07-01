#!/usr/bin/env python3
# Copyright (c) 2026. MoE dispatch/combine decode benchmark driver.
"""
Benchmark DeepEP-v2 (A2A, NCCL-Gin) vs Megatron-NVLS (AGV-V/RSV-V) MoE
dispatch/combine in a single-node decode setting.

Launch (single GB200 node, 4x B200):
    torchrun --nproc_per_node=4 bench/run.py --impl both \
        --batch-sizes 1,2,4,8,16,32,64,128 --deepep-num-sms 16 --reps 100

B is the GLOBAL decode-token count across all EP ranks (B=1 => one token total).
Times ONE full decode step = `--num-layers` MoE layers of paired dispatch->combine
(NVLS also runs its once-per-step metadata kernel once), captured and replayed as a
single CUDA graph, reported in microseconds as the max across ranks (critical path).
Per-layer latency = decode_step / num_layers.
"""

import argparse
import os
import sys

# NOTE: We intentionally do NOT set EP_DISABLE_GIN. DeepEP-v2 is the "NCCL-Gin"
# path; the intra-node NVLink (LSA / direct-peer) transport is obtained from the
# NCCL device communicator that Gin sets up, so disabling Gin can break the v2
# path. On a single NVLink node the RDMA/scaleout Gin contexts are simply
# dormant (allow_hybrid_mode=False -> num_scaleout_ranks==1). Set
# EP_DISABLE_GIN=1 in the environment only as a deliberate experiment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from common import (  # noqa: E402
    Config, init_distributed, make_inputs, all_rank_counts, time_region,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["deepep", "nvls", "both"], default="both")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128",
                   help="comma-separated GLOBAL token counts")
    p.add_argument("--deepep-num-sms", default="16",
                   help="comma-separated SM counts to sweep for DeepEP (rounded to even, "
                        "clamped to the device SM count)")
    p.add_argument("--num-layers", type=int, default=88,
                   help="MoE layers replayed per decode step (Nemotron-Super = 88)")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timing", choices=["graph", "eager"], default="graph")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None, help="CSV output path (rank 0)")
    return p.parse_args()


def timed(fn, group, args):
    """Time `fn`, falling back to eager if graph capture fails."""
    use_graph = args.timing == "graph"
    try:
        return time_region(fn, group, args.warmup, args.reps, use_graph), use_graph
    except Exception as e:  # pragma: no cover - capture is environment-dependent
        if use_graph:
            if dist.get_rank() == 0:
                print(f"  [warn] graph capture failed ({type(e).__name__}: {e}); "
                      f"falling back to eager", flush=True)
            dist.barrier(group)
            return time_region(fn, group, args.warmup, args.reps, False), False
        raise


def main():
    args = parse_args()

    # Import deep_ep BEFORE init_process_group. Importing it AFTER torch has
    # initialized its NCCL leaves DeepEP linked against a different NCCL than the
    # one backing the torch process group, and reading that comm handle in
    # _C.calculate_elastic_buffer_size segfaults during ElasticBuffer construction.
    if args.impl in ("deepep", "both"):
        import deep_ep  # noqa: F401

    group, rank, world, local_rank = init_distributed()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    cfg = Config(
        ep_size=world, rank=rank, local_rank=local_rank, seed=args.seed,
        per_rank_cap=-(-max(batch_sizes) // world),  # ceil(max_B / ep)
    )
    device = torch.device("cuda", local_rank)

    # DeepEP num_sms sweep: even values, clamped to the device SM count; the device
    # max is always included so the sweep reaches "all SMs". (NVLS has no SM knob.)
    dev_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    _even = lambda n: max(2, min(int(n), dev_sms) - (min(int(n), dev_sms) % 2))
    deepep_sms = sorted({_even(s) for s in args.deepep_num_sms.split(",")})

    if rank == 0:
        print(f"# MoE dispatch/combine decode benchmark", flush=True)
        print(f"# EP={world} experts={cfg.num_experts} topk={cfg.topk} hidden={cfg.hidden} "
              f"per_rank_cap={cfg.per_rank_cap} timing={args.timing} reps={args.reps}",
              flush=True)
        print(f"# impl={args.impl} num_layers={args.num_layers} device_sms={dev_sms} "
              f"deepep_num_sms_sweep={deepep_sms}", flush=True)

    # Build benchers.
    benchers = []
    if args.impl in ("deepep", "both"):
        from bench_deepep import DeepEPBencher
        benchers.append(DeepEPBencher(cfg, group, deepep_sms[0]))
    if args.impl in ("nvls", "both"):
        from bench_nvls import NVLSBencher
        benchers.append(NVLSBencher(cfg, group))
    # Force NCCL communicator creation BEFORE building benchers. torch initializes
    # NCCL lazily (comm created on first collective); DeepEP's ElasticBuffer ctor
    # reads the comm handle in _C.calculate_elastic_buffer_size, which segfaults if
    # the comm has not been created yet. A barrier here materializes it.
    dist.barrier(group)
    for b in benchers:
        b.build()
    dist.barrier(group)

    rows = []  # (impl, B, num_sms, phase, counts, latency_us, timing_mode)
    nl = args.num_layers
    for b in benchers:
        # DeepEP sweeps num_sms; NVLS has no SM knob (num_sms = -1 = N/A).
        sms_list = deepep_sms if b.name == "deepep" else [-1]
        for sms in sms_list:
            if b.name == "deepep":
                b.set_num_sms(sms)
            for B in batch_sizes:
                hidden, topk_idx, topk_weights = make_inputs(cfg, B, device)
                b.setup_batch(hidden, topk_idx, topk_weights)
                counts = all_rank_counts(B, world)

                # Time ONE full decode step = `nl` MoE layers of paired dispatch->
                # combine (NVLS: + one metadata), captured & replayed as a single
                # CUDA graph. Recording all nl layers in the graph is what models the
                # layer dimension; the metadata is included once (per step), not per layer.
                (_, step_us), mode = timed(lambda: b.decode_step(nl), group, args)
                rows.append((b.name, B, sms, "decode_step", counts, step_us, mode))

                if rank == 0:
                    smstr = f" sms={sms:3d}" if b.name == "deepep" else " sms= na"
                    print(f"  {b.name:7s} B={B:4d}{smstr} counts={counts} "
                          f"decode_step({nl}L)={step_us:9.2f}us "
                          f"(per-layer={step_us / nl:6.2f}us)  "
                          f"[{'graph' if mode else 'eager'}]", flush=True)

    if rank == 0:
        print("\n# impl,global_B,num_sms,phase,latency_us,timing", flush=True)
        for name, B, sms, phase, counts, us, mode in rows:
            print(f"{name},{B},{sms},{phase},{us:.3f},{'graph' if mode else 'eager'}", flush=True)
        if args.out:
            with open(args.out, "w") as f:
                f.write("impl,global_B,num_sms,phase,per_rank_counts,latency_us,timing\n")
                for name, B, sms, phase, counts, us, mode in rows:
                    f.write(f"{name},{B},{sms},{phase},\"{counts}\",{us:.4f},"
                            f"{'graph' if mode else 'eager'}\n")
            print(f"# wrote {args.out}", flush=True)

    for b in benchers:
        if hasattr(b, "destroy"):
            b.destroy()
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
