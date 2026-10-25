#!/usr/bin/env python3
# Copyright (c) 2026. MoE dispatch/combine decode benchmark driver.
"""
Benchmark MoE dispatch/combine schemes in a decode setting within one NVLink domain:
  - DeepEP        (EPBuffer, DeepEP >= 2.5; NVLink all-to-all, set up like Megatron's
                   DeepEPDispatcher),
  - Megatron-NVLS (AllGather-V/ReduceScatter-V via NVLink multicast),
  - Megatron-NCCL (dense AllGather/ReduceScatter, CUDA-graph path, padded to equal counts),
  - A2A-V         (unicast dispatch / pull combine over symmetric memory).

Launch (one node, 4 GPUs):
    torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,a2a \
        --batch-sizes 1,2,4,8,16,32,64,128 --reps 100

--model (models.py: super | ultra) sets the MoE shape (experts, top-k, dispatched row width
= moe_latent_size, expert FFN width) and the default --num-layers (the model's MoE block count).
--expert vllm runs Megatron's vllm_fused_moe between dispatch and combine, called the way
Megatron's MoE layer calls it for each dispatcher (needs --megatron-path); the default
identity expert times communication only.
B is the GLOBAL decode-token count across all EP ranks (B=1 => one token total).
Times ONE full decode step = `--num-layers` MoE layers of paired dispatch->expert->combine
(NVLS and A2A-V also run their once-per-step metadata kernel once), captured and replayed as a
single CUDA graph, reported in microseconds as the max across ranks (critical path).
Per-layer latency = decode_step / num_layers.
"""

import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from common import (  # noqa: E402
    Config, init_distributed, make_inputs, all_rank_counts, time_region,
)
from models import DEFAULT_MODEL, MODELS, get_model  # noqa: E402


IMPLS = ("deepep", "nvls", "nccl", "a2a")
IMPL_ALIASES = {"both": ("deepep", "nvls"), "all": ("deepep", "nvls", "nccl")}


def parse_impls(spec: str) -> "list[str]":
    """Comma-separated impl names and/or aliases -> ordered, de-duplicated impl list."""
    impls = []
    for name in spec.split(","):
        name = name.strip()
        for impl in IMPL_ALIASES.get(name, (name,)):
            if impl not in IMPLS:
                raise SystemExit(f"unknown --impl '{impl}' (choose from {IMPLS} or "
                                 f"{tuple(IMPL_ALIASES)})")
            if impl not in impls:
                impls.append(impl)
    return impls


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                   help="model whose MoE shape to benchmark (models.py): sets experts, top-k, "
                        "dispatched row width and the default --num-layers")
    p.add_argument("--impl", default="all",
                   help="comma-separated dispatcher(s): deepep | nvls | nccl | a2a, or the "
                        "aliases both (deepep,nvls) and all (deepep,nvls,nccl); "
                        "e.g. --impl deepep,a2a")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128",
                   help="comma-separated GLOBAL token counts")
    p.add_argument("--deepep-num-sms", default="auto",
                   help="comma-separated SM counts to sweep for DeepEP (rounded to even, "
                        "clamped to the device SM count). 'auto' (default) = DeepEP's "
                        "get_theoretical_num_sms, agreed across ranks -- Megatron's default")
    p.add_argument("--per-rank-cap", type=int, default=None,
                   help="per-rank token capacity of every buffer (default: "
                        "ceil(max batch / EP)); Megatron's e2e uses round_up(max_tokens)/TP")
    p.add_argument("--expert", choices=["identity", "vllm"], default="identity",
                   help="expert step between dispatch and combine: identity (comm only) or "
                        "vllm (Megatron's vllm_fused_moe with the model's expert shapes, called "
                        "the way Megatron calls it for each dispatcher)")
    p.add_argument("--megatron-path", default=os.environ.get("MEGATRON_PATH"),
                   help="Megatron-LM checkout to import vllm_fused_moe from (--expert vllm)")
    p.add_argument("--num-layers", type=int, default=None,
                   help="MoE layers replayed per decode step (default: the model's MoE block "
                        "count -- Super 40, Ultra 48)")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timing", choices=["graph", "eager"], default="graph")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None,
                   help="CSV output file (rank 0). A bare file name is placed under "
                        "--results-dir; a path with a directory is used as given.")
    p.add_argument("--results-dir", default=os.path.join(REPO_DIR, "results"),
                   help="directory for CSV outputs (created if missing)")
    p.add_argument("--validate", action="store_true",
                   help="run known-value correctness checks for each impl and exit "
                        "(no timing): confirms the isolated dispatch/combine actually "
                        "move/reduce data correctly before trusting the benchmark; with "
                        "--expert vllm, also checks dispatch->experts->combine against a "
                        "torch reference")
    return p.parse_args()


def run_validation(benchers, cfg, experts, device) -> bool:
    """Correctness checks on this rank -> True if all passed here. Every rank runs the same
    sequence (the collectives stay in lockstep); rank 0 prints, other ranks print failures.

    1. Each impl's own known-value checks (b.validate(), identity expert).
    2. Cross-implementation equivalence (>=2 impls): the SAME random tokens + routing through
       every impl's identity-expert round-trip must give the analytic m*x (m = #distinct
       destination ranks) and agree with each other.
    3. With --expert vllm: the timed layer (dispatch -> vllm_fused_moe -> combine, exactly as
       decode_step runs it) on the benchmark's inputs must match a plain-torch MoE reference
       computed from every rank's expert weights, and the impls must agree with each other.
    """
    rank, world = cfg.rank, cfg.ep_size
    local_ok = True

    def report(name, ok, detail):
        nonlocal local_ok
        local_ok = local_ok and ok
        if rank == 0 or not ok:
            print(f"  [rank{rank}] [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

    def compare(outs, ref, n, ref_tol, cross_tol):
        """Max error of each output vs ref and pairwise, relative to max|ref|."""
        scale = float(ref.abs().max()) + 1e-6
        errs = {name: float((o - ref).abs().max()) for name, o in outs.items()}
        names = list(outs)
        cross = max((float((outs[a] - outs[b_]).abs().max())
                     for i, a in enumerate(names) for b_ in names[i + 1:]), default=0.0)
        ok = (cross / scale < cross_tol) and all(e / scale < ref_tol for e in errs.values())
        errstr = " ".join(f"{k}={v:.4f}" for k, v in errs.items())
        return ok, (f"n={n} max|pairwise|={cross:.4f} ({100 * cross / scale:.2f}% of "
                    f"{scale:.2f}); vs ref: {errstr}")

    for b in benchers:
        b.expert = None
        for name, ok, detail in b.validate():
            report(name, ok, detail)

    # Cross-implementation equivalence (identity expert). NCCL reduces in fp32 (exact m*x);
    # NVLS, DeepEP and A2A-V combine in bf16 (round). The 2%/5% tolerances absorb that.
    rt = [b for b in benchers if hasattr(b, "functional_roundtrip")]
    if len(rt) >= 2:
        E, K, H, epr = cfg.num_experts, cfg.topk, cfg.hidden, cfg.num_local_experts
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
                ok, detail = compare(outs, m * x.to(torch.float32), n, 0.05, 0.02)
            else:
                ok, detail = True, "n=0 (idle rank)"
            report(f"cross-impl [{'='.join(outs)}] B={B:<4d}", ok, detail)

    # The expert path: one timed layer per impl vs the torch reference. Combine rounds to bf16
    # (NCCL: fp32) and vllm_fused_moe keeps bf16 intermediates, hence the 3%/2% tolerances.
    if experts is not None:
        for B in (world * 2, 1):
            n = all_rank_counts(B, world)[rank]
            hidden, topk_idx, topk_weights = make_inputs(cfg, B, device)
            outs = {}
            for b in benchers:
                b.expert = experts
                b.setup_batch(hidden, topk_idx, topk_weights)
                outs[b.name] = b.decode_step(1)[:n].to(torch.float32)
                b.expert = None
            torch.cuda.synchronize()
            if n > 0:
                ref = experts.reference(hidden, topk_idx, topk_weights)
                ok, detail = compare(outs, ref, n, 0.03, 0.02)
            else:
                ok, detail = True, "n=0 (idle rank)"
            report(f"vllm experts [{'='.join(outs)}] B={B:<4d}", ok, detail)
    return local_ok


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

    # Import deep_ep BEFORE init_process_group: its import-time NCCL check scans the loaded
    # libraries for libnccl and fails as a duplicate once a communicator has loaded HPC-X's
    # NCCL net plugin.
    if "deepep" in impls:
        import deep_ep  # noqa: F401
    if args.expert == "vllm":
        # Import Megatron before NCCL starts too: megatron.core imports deep_ep (via
        # transformer/moe/fused_a2a.py), which runs the same check.
        if args.megatron_path:
            sys.path.insert(0, args.megatron_path)
        import megatron.core.inference.moe.vllm_fused_moe  # noqa: F401

    group, rank, world, local_rank = init_distributed()

    spec = get_model(args.model)
    if args.num_layers is None:
        args.num_layers = spec.num_moe_layers
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    cfg = Config(
        model=args.model,
        ep_size=world, rank=rank, local_rank=local_rank, seed=args.seed,
        per_rank_cap=args.per_rank_cap or -(-max(batch_sizes) // world),  # ceil(max_B / ep)
    )
    assert cfg.per_rank_cap * world >= max(batch_sizes), "--per-rank-cap too small"
    device = torch.device("cuda", local_rank)

    # DeepEP num_sms sweep: even values, clamped to the device SM count; 'auto' (the default)
    # is DeepEP's theoretical SM count. (NVLS / A2A-V use a CTA cap instead, see
    # default_num_blocks.)
    dev_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    _even = lambda n: max(2, min(int(n), dev_sms) - (min(int(n), dev_sms) % 2))
    # 0 = DeepEP's theoretical SM count, resolved after the buffer is built.
    deepep_sms = sorted({0 if s == "auto" else _even(s) for s in args.deepep_num_sms.split(",")})

    if rank == 0:
        print(f"# MoE dispatch/combine decode benchmark", flush=True)
        print(f"# model={spec.name} ({args.model}): {spec.num_blocks} blocks, "
              f"{spec.num_moe_layers} MoE, "
              f"hidden_size={spec.hidden_size} moe_latent_size={spec.moe_latent_size} "
              f"moe_intermediate_size={spec.moe_intermediate_size}",
              flush=True)
        print(f"# EP={world} experts={cfg.num_experts} topk={cfg.topk} hidden={cfg.hidden} "
              f"per_rank_cap={cfg.per_rank_cap} timing={args.timing} reps={args.reps}",
              flush=True)
        print(f"# impl={','.join(impls)} num_layers={args.num_layers} device_sms={dev_sms} "
              f"deepep_num_sms_sweep={['auto' if s == 0 else s for s in deepep_sms]}", flush=True)

    # Build benchers.
    benchers = []
    if "deepep" in impls:
        from bench_deepep import DeepEPBencher
        benchers.append(DeepEPBencher(cfg, group, deepep_sms[0]))
    if "nvls" in impls:
        from bench_nvls import NVLSBencher
        benchers.append(NVLSBencher(cfg, group))
    if "nccl" in impls:
        from bench_nccl import NCCLBencher
        benchers.append(NCCLBencher(cfg, group))
    if "a2a" in impls:
        from bench_a2a import A2ABencher
        benchers.append(A2ABencher(cfg, group))
    # Force NCCL communicator creation BEFORE building benchers. torch initializes NCCL
    # lazily (comm created on first collective), and DeepEP's EPBuffer reuses the group's
    # communicator, so a barrier here materializes it (as Megatron's DeepEPDispatcher does).
    dist.barrier(group)
    for b in benchers:
        b.build()
    dist.barrier(group)
    experts = None  # attached to the benchers after --validate
    if args.expert == "vllm":
        from common import VllmExperts
        experts = VllmExperts(cfg, max_rows=cfg.global_cap, device=device)
    if rank == 0:
        print(f"# expert={args.expert} per_rank_cap={cfg.per_rank_cap}", flush=True)
    if rank == 0:
        for b in benchers:
            if b.name in ("nvls", "a2a"):
                print(f"# {b.name}: {b.num_blocks} CTAs x EP={world} -> "
                      f"{b.num_blocks * world * 4} B of barrier slots; "
                      f"signal pad = {b.signal_pad_bytes} B", flush=True)

    # --validate: run known-value correctness checks per impl and exit. Confirms the
    # isolated collectives actually move/reduce data correctly before we trust any latency
    # number. Verdict is reduced across ranks (a check may fail on one rank only).
    if args.validate:
        local_ok = run_validation(benchers, cfg, experts, device)
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

    for b in benchers:
        b.expert = experts

    rows = []  # (impl, B, num_sms, sms_setting, phase, counts, latency_us, timing_mode)
    nl = args.num_layers
    gpu = torch.cuda.get_device_name(local_rank).removeprefix("NVIDIA ").replace(",", " ")
    for b in benchers:
        # DeepEP sweeps num_sms. NVLS and A2A-V run once at their block cap (b.num_sms, see
        # default_num_blocks). NCCL picks its own grid (b.num_sms=-1 = N/A).
        sms_list = deepep_sms if b.name == "deepep" else [b.num_sms]
        for sms in sms_list:
            # How DeepEP's SM count was chosen: 'auto' (theoretical) or 'fixed' (given).
            sms_setting = ("auto" if sms == 0 else "fixed") if b.name == "deepep" else ""
            if b.name == "deepep":
                b.set_num_sms(sms)
                sms = b.num_sms  # resolves 'auto' (0)
            for B in batch_sizes:
                hidden, topk_idx, topk_weights = make_inputs(cfg, B, device)
                b.setup_batch(hidden, topk_idx, topk_weights)
                counts = all_rank_counts(B, world)

                # Time ONE full decode step = `nl` MoE layers of paired dispatch->
                # combine (NVLS: + one metadata), captured & replayed as a single
                # CUDA graph. Recording all nl layers in the graph is what models the
                # layer dimension; the metadata is included once (per step), not per layer.
                (_, step_us), mode = timed(lambda: b.decode_step(nl), group, args)
                rows.append((b.name, B, sms, sms_setting, "decode_step", counts, step_us, mode))

                if rank == 0:
                    smstr = f" sms={sms:3d}" if sms >= 0 else " sms= na"
                    smstr += f" ({sms_setting})" if sms_setting else ""
                    print(f"  {b.name:7s} B={B:4d}{smstr} counts={counts} "
                          f"decode_step({nl}L)={step_us:9.2f}us "
                          f"(per-layer={step_us / nl:6.2f}us)  "
                          f"[{'graph' if mode else 'eager'}]", flush=True)

    if rank == 0:
        print("\n# impl,global_B,num_sms,phase,latency_us,timing,model,num_layers", flush=True)
        for name, B, sms, _, phase, counts, us, mode in rows:
            print(f"{name},{B},{sms},{phase},{us:.3f},{'graph' if mode else 'eager'},"
                  f"{args.model},{nl}", flush=True)
        if args.out:
            out_path = (args.out if os.path.dirname(args.out)
                        else os.path.join(args.results_dir, args.out))
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                # model / num_layers / expert / gpu let plot_results.py label the plot
                # without flags.
                f.write("impl,global_B,num_sms,phase,per_rank_counts,latency_us,timing,"
                        "model,num_layers,expert,per_rank_cap,sms_setting,gpu\n")
                for name, B, sms, sms_setting, phase, counts, us, mode in rows:
                    f.write(f"{name},{B},{sms},{phase},\"{counts}\",{us:.4f},"
                            f"{'graph' if mode else 'eager'},{args.model},{nl},"
                            f"{args.expert},{cfg.per_rank_cap},{sms_setting},{gpu}\n")
            print(f"# wrote {out_path}", flush=True)

    for b in benchers:
        if hasattr(b, "destroy"):
            b.destroy()
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
