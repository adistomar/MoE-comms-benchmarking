#!/usr/bin/env python3
# Copyright (c) 2026. MoE dispatch/combine decode benchmark driver.
"""
Benchmark three MoE dispatch/combine schemes in a single-node decode setting:
  - DeepEP-v2   (A2A over NCCL-Gin),
  - Megatron-NVLS (AllGather-V/ReduceScatter-V via NVLink multicast),
  - Megatron-NCCL (dense AllGather/ReduceScatter, CUDA-graph path, padded to equal counts).

Launch (single GB200 node, 4x B200):
    torchrun --nproc_per_node=4 bench/run.py --impl all \
        --batch-sizes 1,2,4,8,16,32,64,128 --deepep-num-sms 148 --reps 100

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
    p.add_argument("--impl", default="all",
                   help="which dispatcher(s): a single name, a COMMA-SEPARATED list, or a "
                        "shortcut. Names: deepep | nvls | nccl | a2av (A2AV dispatch + A2AV pull "
                        "combine) | a2av_rs (A2AV dispatch + reduce-scatter-v combine). Shortcuts: "
                        "both (deepep+nvls) | all (deepep+nvls+nccl+a2av+a2av_rs). "
                        "Examples: 'a2av', 'nvls,a2av,a2av_rs', 'nvls,deepep,a2av,a2av_rs'")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128",
                   help="comma-separated GLOBAL token counts")
    p.add_argument("--deepep-num-sms", default="148",
                   help="comma-separated SM counts to sweep for DeepEP (rounded to even, "
                        "clamped to the device SM count; default 148 = B200 SM count, "
                        "matching NVLS's fixed block cap)")
    p.add_argument("--num-layers", type=int, default=88,
                   help="MoE layers replayed per decode step (Nemotron-Super = 88)")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timing", choices=["graph", "eager"], default="graph")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None, help="CSV output path (rank 0)")
    p.add_argument("--validate", action="store_true",
                   help="run known-value correctness checks for each impl and exit "
                        "(no timing): confirms the isolated dispatch/combine actually "
                        "move/reduce data correctly before trusting the benchmark")
    return p.parse_args()


_KNOWN_IMPLS = ["deepep", "nvls", "nccl", "a2av", "a2av_rs"]


def parse_impls(spec):
    """Resolve --impl into an ordered list of impl names.

    Accepts a shortcut ('all' -> all five; 'both' -> deepep+nvls) or a comma-separated
    list of individual names (e.g. 'nvls,a2av,a2av_rs'). Order is preserved so all ranks
    build/iterate the benchers in lockstep.
    """
    if spec == "all":
        return list(_KNOWN_IMPLS)
    if spec == "both":
        return ["deepep", "nvls"]
    names = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [n for n in names if n not in _KNOWN_IMPLS]
    if bad:
        raise SystemExit(f"unknown impl(s) {bad}; choose from {_KNOWN_IMPLS} "
                         f"(comma-separated) or the shortcuts 'all'/'both'")
    if not names:
        raise SystemExit("no impls selected (--impl was empty)")
    return names


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
    impls = parse_impls(args.impl)

    # Import deep_ep BEFORE init_process_group. Importing it AFTER torch has
    # initialized its NCCL leaves DeepEP linked against a different NCCL than the
    # one backing the torch process group, and reading that comm handle in
    # _C.calculate_elastic_buffer_size segfaults during ElasticBuffer construction.
    if "deepep" in impls:
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

    # Build benchers, in the order the user listed them (all ranks agree -> lockstep).
    benchers = []
    for name in impls:
        if name == "deepep":
            from bench_deepep import DeepEPBencher
            benchers.append(DeepEPBencher(cfg, group, deepep_sms[0]))
        elif name == "nvls":
            from bench_nvls import NVLSBencher
            benchers.append(NVLSBencher(cfg, group))
        elif name == "nccl":
            from bench_nccl import NCCLBencher
            benchers.append(NCCLBencher(cfg, group))
        elif name == "a2av":
            from bench_a2av import A2AVBencher
            benchers.append(A2AVBencher(cfg, group))
        elif name == "a2av_rs":
            from bench_a2av import A2AVRSBencher
            benchers.append(A2AVRSBencher(cfg, group))
    # Force NCCL communicator creation BEFORE building benchers. torch initializes
    # NCCL lazily (comm created on first collective); DeepEP's ElasticBuffer ctor
    # reads the comm handle in _C.calculate_elastic_buffer_size, which segfaults if
    # the comm has not been created yet. A barrier here materializes it.
    dist.barrier(group)
    for b in benchers:
        b.build()
    dist.barrier(group)

    # --validate: run known-value correctness checks per impl and exit. Confirms the
    # isolated collectives actually move/reduce data correctly (gather row g == g;
    # RSV sums across peers; DeepEP dispatch->combine round-trips) before we trust any
    # latency number. Verdict is reduced across ranks (a check may fail on one rank only).
    if args.validate:
        local_ok = True
        for b in benchers:
            for name, ok, detail in b.validate():
                local_ok = local_ok and ok
                if rank == 0 or not ok:
                    print(f"  [rank{rank}] [{'PASS' if ok else 'FAIL'}] {name}  {detail}",
                          flush=True)

        # Cross-implementation equivalence (>=2 impls built): feed the SAME random
        # tokens + routing to every impl, run a full functional dispatch -> identity
        # expert -> combine, and require all outputs to match the analytic m*x (m =
        # #distinct dest ranks) AND each other. NCCL reduces in fp32 (exact m*x); NVLS and
        # DeepEP combine in bf16 (round). The 2%/5% tolerances below absorb bf16 rounding.
        # This shows the impls are mutually equivalent and that dividing by m recovers x.
        rt = [b for b in benchers if hasattr(b, "functional_roundtrip")]
        if len(rt) >= 2:
            E, K, H, epr = cfg.num_experts, cfg.topk, cfg.hidden, cfg.num_experts // world
            for B in (world * 2, 1):
                n = all_rank_counts(B, world)[rank]
                gen = torch.Generator(device=device).manual_seed(555 + B + rank * 977)
                x = (torch.randn(n, H, generator=gen, device=device).to(torch.bfloat16)
                     if n > 0 else torch.empty(0, H, device=device, dtype=torch.bfloat16))
                idx = (torch.stack([torch.randperm(E, generator=gen, device=device)[:K]
                                    for _ in range(n)])
                       if n > 0 else torch.empty(0, K, device=device, dtype=torch.int64))
                # Call every impl's round-trip in a fixed order (all ranks agree -> the
                # underlying collectives stay in lockstep).
                outs = {b.name: b.functional_roundtrip(x, idx).to(torch.float32) for b in rt}
                torch.cuda.synchronize()
                if n > 0:
                    m = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                     device=device, dtype=torch.float32).view(n, 1)
                    ref = m * x.to(torch.float32)
                    scale = float(ref.abs().max()) + 1e-6
                    errs = {name: float((o - ref).abs().max()) for name, o in outs.items()}
                    names = list(outs)
                    cross = max((float((outs[a] - outs[b_]).abs().max())
                                 for i, a in enumerate(names) for b_ in names[i + 1:]),
                                default=0.0)
                    ok = (cross / scale < 0.02) and all(e / scale < 0.05 for e in errs.values())
                    errstr = " ".join(f"{k}={v:.4f}" for k, v in errs.items())
                    detail = (f"n={n} max|pairwise|={cross:.4f} ({100 * cross / scale:.2f}% of "
                              f"{scale:.2f}); vs m*x: {errstr}")
                else:
                    ok, detail = True, "n=0 (idle rank)"
                local_ok = local_ok and ok
                if rank == 0 or not ok:
                    print(f"  [rank{rank}] [{'PASS' if ok else 'FAIL'}] "
                          f"cross-impl [{'='.join(outs)}] B={B:<4d}  {detail}", flush=True)

        verdict = torch.tensor([1.0 if local_ok else 0.0], device=device)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
        if rank == 0:
            print(f"# VALIDATION {'PASSED' if verdict.item() > 0.5 else 'FAILED'} "
                  f"(all {world} ranks)", flush=True)
        dist.barrier(group)
        for b in benchers:
            if hasattr(b, "destroy"):
                b.destroy()
        dist.destroy_process_group()
        sys.exit(0 if verdict.item() > 0.5 else 1)

    rows = []  # (impl, B, num_sms, phase, counts, latency_us, timing_mode)
    nl = args.num_layers
    for b in benchers:
        # DeepEP sweeps num_sms. NVLS runs once at its fixed block cap (b.num_sms=148,
        # its NVLS analog of num_sms). NCCL picks its own grid (b.num_sms=-1 = N/A).
        sms_list = deepep_sms if b.name == "deepep" else [b.num_sms]
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
                    smstr = f" sms={sms:3d}" if sms >= 0 else " sms= na"
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
