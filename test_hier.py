#!/usr/bin/env python3
# Copyright (c) 2026. Hierarchical-dispatch milestone tests (K0/K1) with torch oracles.
"""
Standalone torchrun driver validating the hierarchical-dispatch building blocks
against pure-torch oracles, on a small grid (e.g. 4 ranks, g=2 => 2x2).

  --milestone k1   Directed-P2P outer all-to-all SCATTER (nvls/torch_symm_triton/
                   directed_p2p.py). Each rank sends its tokens to the column-peer
                   ranks owning the groups its tokens route to, at contiguous rows.
                   Oracle: gather every column source's (hidden, routing) and replay
                   outer_send_plan to reconstruct the expected destination buffer.

Run:
    torchrun --nproc_per_node=4 test_hier.py --milestone k1 --g 2
Exit 0 = PASS on all ranks.
"""

import argparse
import os
import sys

# Per-rank node-local Triton cache dir — avoids the shared-/lustre compile race at 64+ ranks
# ("OSError: Stale file handle"). Must precede any triton import (bench_hier -> triton).
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/triton_cache_{os.environ.get('SLURM_JOB_ID', '0')}_{os.environ.get('RANK', '0')}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from common import Config, init_distributed  # noqa: E402
from hier_common import (  # noqa: E402
    HierGrid, HierPlacement, make_groups, dest_group_mask, outer_send_plan,
)
from nvls.symmetric_memory import SymmetricMemoryManager  # noqa: E402
from nvls.torch_symm_triton.directed_p2p import directed_a2a_scatter  # noqa: E402


def _groups_per_token(routing, placement):
    """[n,K] int64 -> list (len n) of sorted distinct destination group ids."""
    mask = dest_group_mask(routing, placement)          # [n, G] bool
    return [sorted(torch.nonzero(mask[t], as_tuple=False).flatten().tolist())
            for t in range(mask.shape[0])]


def test_k0_inner(cfg, grid, row_group, col_group, args):
    """K0: row-scoped inner AGv/RSv. Reuse NVLSBencher over the ROW group (world=g).
    Gate: AGv gather bit-exact and RSv reduces over the g ROW ranks (output == g*x),
    NOT P*x -> confirms the multicast pointer + metadata are scoped to the row, and
    that a rank can hold both a row and a column symmetric-memory group."""
    from bench_nvls import NVLSBencher
    g = grid.g
    cfg_row = Config(ep_size=g, rank=grid.pos, local_rank=cfg.local_rank,
                     seed=args.seed, per_rank_cap=max(2, args.num_tokens))
    b = NVLSBencher(cfg_row, row_group)
    b.build()
    res = b.validate()                         # "NVLS AGV-V gather ..." / "NVLS RSV-V reduce ..."
    return [(f"K0 inner (row g={g}): {name}", ok, detail) for name, ok, detail in res]


def test_k3_overlap(cfg, grid, args):
    """K3b go/no-go: measure outer-P2P vs inner-multicast overlap at a few B and SM splits.
    overlap ~1 => the two hops overlap (pipelining can beat sequential); ~0 => they serialize
    on the shared fabric/SMs (pipelining won't help)."""
    from bench_hier import HierBencher
    from common import all_rank_counts
    dev = torch.device("cuda", cfg.local_rank)
    b = HierBencher(cfg, dist.group.WORLD, args.g)
    b.build()
    E, K, H = cfg.num_experts, cfg.topk, cfg.hidden
    Bs = [int(x) for x in args.batch_sizes.split(",")]
    for B in Bs:
        n = all_rank_counts(B, cfg.ep_size)[cfg.rank]
        gen = torch.Generator(device=dev).manual_seed(1234 + B + cfg.rank)
        hidden = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                  if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
        idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
               if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
        b.setup_batch(hidden, idx, None)
        for sm in (74, 48):
            r = b.overlap_probe(reps=30, sm_each=sm)
            if cfg.rank == 0:
                print(f"  [overlap] B={B:5d} sm_each={sm:3d}: outer={r['outer_us']:6.1f} "
                      f"inner={r['inner_us']:6.1f} both={r['both_us']:6.1f} "
                      f"(sum={r['sum_us']:6.1f} max={r['max_us']:6.1f})  overlap={r['overlap']:+.2f}",
                      flush=True)
        # Best-case pipelined per-layer (all outer || all inner) vs sequential (full SMs).
        pb = b.pipeline_bound(num_layers=20, reps=3, sm_each=74)
        if cfg.rank == 0:
            print(f"  [pipe]    B={B:5d}: seq_full_sms={pb['seq_full_us']:6.1f}  "
                  f"pipelined_bound(sm74)={pb['pipe_bound_us']:6.1f} us/layer", flush=True)
    return [("K3b overlap + pipeline-bound probe", True, "see [overlap]/[pipe] lines")]


def test_decompose(cfg, grid, args):
    """Fused-kernel GO/NO-GO: split the sequential per-layer latency into its 4 phases,
    EACH graph-captured under the same conditions as run.py's full-step number. Each comm
    phase carries exactly one symm-mem barrier + one grid launch, so at tiny B (payload
    ~free) a phase's us is ~pure launch+barrier overhead = the part a FUSED single kernel
    removes. `full` should ~= `sum` (cross-check). `[ceil]` reports the outer||inner overlap
    so we can bound what a pipelined single kernel could achieve vs the sequential `full`."""
    from bench_hier import HierBencher
    from common import all_rank_counts
    dev = torch.device("cuda", cfg.local_rank)
    b = HierBencher(cfg, dist.group.WORLD, args.g)
    b.build()
    E, K, H = cfg.num_experts, cfg.topk, cfg.hidden
    Bs = [int(x) for x in args.batch_sizes.split(",")]
    for B in Bs:
        n = all_rank_counts(B, cfg.ep_size)[cfg.rank]
        gen = torch.Generator(device=dev).manual_seed(1234 + B + cfg.rank)
        hidden = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                  if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
        idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
               if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
        b.setup_batch(hidden, idx, None)
        d = b.phase_decompose(num_layers=args.num_layers, reps=args.reps, warmup=6)
        ov = b.overlap_probe(reps=30, sm_each=74)
        if cfg.rank == 0:
            print(f"  [decomp] B={B:5d} topk={cfg.topk}: "
                  f"scatter={d['scatter']:6.2f} inner_agv={d['inner_agv']:6.2f} "
                  f"mask_rsv={d['mask_rsv']:6.2f} gather_reduce={d['gather_reduce']:6.2f} | "
                  f"sum={d['sum_phases']:6.2f} full={d['full_step']:6.2f} us/layer", flush=True)
            print(f"  [ceil]   B={B:5d}: outer={ov['outer_us']:6.2f} inner={ov['inner_us']:6.2f} "
                  f"both={ov['both_us']:6.2f} overlap={ov['overlap']:+.2f} "
                  f"dispatch_max(o,i)={ov['max_us']:6.2f} us/layer", flush=True)
    return [("phase decomposition", True, "see [decomp]/[ceil] lines")]


def test_k3_fused(cfg, grid, args):
    """K3 increment 1: the FUSED barrier-free dispatch (_dispatch_fused) must (a) produce
    bit-identical gathered ah/ar to the staged _dispatch, and (b) still round-trip to m*x.
    Runs both paths on the same input and compares. Small B first (bugs localize)."""
    from bench_hier import HierBencher
    from common import all_rank_counts
    dev = torch.device("cuda", cfg.local_rank)
    b = HierBencher(cfg, dist.group.WORLD, args.g)
    b.build()
    E, K, H, epr = cfg.num_experts, cfg.topk, cfg.hidden, b.epr
    out = []
    for B in [int(x) for x in args.batch_sizes.split(",")]:
        n = all_rank_counts(B, cfg.ep_size)[cfg.rank]
        gen = torch.Generator(device=dev).manual_seed(4242 + B + cfg.rank)
        x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
             if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
        idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
               if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
        b.setup_batch(x, idx, None); b._dispatch()
        ah_s, ar_s = b.ah[:b._valid].clone(), b.ar[:b._valid].clone()
        b._combine(); out_s = b.out[:n].clone().float()
        b.setup_batch(x, idx, None)
        b._dispatch_fused(scatter_ctas=args.scatter_ctas, no_flags=args.no_flags,
                          skip_wait=args.skip_wait, wait_iters=args.wait_iters)
        ah_f, ar_f = b.ah[:b._valid].clone(), b.ar[:b._valid].clone()
        b._combine(); out_f = b.out[:n].clone().float()
        torch.cuda.synchronize()
        R = b._R
        seen = int((b.dbg[:R] == 1).sum()) if R > 0 else 0
        print(f"  [rank{cfg.rank}] B={B} R={R} flags_seen={seen}/{R} "
              f"flag_raw_set={int((b.dflag[:R] == 1).sum()) if R>0 else 0}", flush=True)
        ah_ok = bool(torch.equal(ah_f, ah_s)); ar_ok = bool(torch.equal(ar_f, ar_s))
        if n > 0:
            mm = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                              device=dev, dtype=torch.float32).view(n, 1)
            ref = mm * x.float()
            mx_ok = bool(torch.allclose(out_f, ref, rtol=0.03, atol=0.05))
            detail = (f"n={n} ah_eq={ah_ok} ar_eq={ar_ok} mx_ok={mx_ok} "
                      f"max|fused-mx|={float((out_f - ref).abs().max()):.4f} "
                      f"max|fused-staged|={float((out_f - out_s).abs().max()):.4f}")
            ok = ah_ok and ar_ok and mx_ok
        else:
            ok, detail = ah_ok and ar_ok, f"n=0 (idle) ah_eq={ah_ok} ar_eq={ar_ok}"
        out.append((f"K3 fused dispatch B={B:<5d}", ok, detail))
    return out


def test_k2_roundtrip(cfg, grid, args):
    """K2: full hierarchical dispatch->combine == m*x, end-to-end. HierBencher makes
    its own row/column subgroups. This is the R6/R7/R8 gate (bf16 drift, return-path
    identity, 0-token ranks)."""
    from bench_hier import HierBencher
    import torch.distributed as dist
    b = HierBencher(cfg, dist.group.WORLD, args.g)
    b.build()
    return b.validate()


def test_k1_scatter(cfg, grid, row_group, col_group, args):
    dev = torch.device("cuda", cfg.local_rank)
    G, g, my_group = grid.G, grid.g, grid.group_id
    H, K, E = cfg.hidden, cfg.topk, cfg.num_experts
    placement = HierPlacement(E, cfg.ep_size, g)
    n = args.num_tokens
    cap = G * n  # worst-case tokens landing on any destination in this column

    # Local inputs. hidden[t] carries a recognizable value (my_group*1000 + t) so the
    # oracle can verify placement; routing = distinct random experts (drives dest groups).
    gen = torch.Generator(device=dev).manual_seed(4242 + my_group * 97 + grid.pos * 7)
    hidden = torch.empty(n, H, dtype=torch.bfloat16, device=dev)
    for t in range(n):
        hidden[t] = float(my_group * 1000 + t)
    routing = torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)]).to(torch.int64)
    srctok = torch.arange(n, dtype=torch.int32, device=dev)

    # Column count matrix M[G,G]: exchange each source's per-destination token counts.
    my_M_row = dest_group_mask(routing, placement).sum(dim=0).to(torch.int64)   # [G]
    M_flat = torch.empty(G * G, dtype=torch.int64, device=dev)
    dist.all_gather_into_tensor(M_flat, my_M_row, group=col_group)
    M = M_flat.view(G, G)                                   # M[source, dest]

    # This rank's send plan (K1b contiguous landing).
    tdg = _groups_per_token(routing, placement)
    sends, recv_counts = outer_send_plan(tdg, M.tolist(), my_group)
    send_dest = torch.tensor([d for d, _, _ in sends], dtype=torch.int32, device=dev)
    send_tok = torch.tensor([t for _, t, _ in sends], dtype=torch.int32, device=dev)
    send_row = torch.tensor([r for _, _, r in sends], dtype=torch.int32, device=dev)

    # Column receive buffer (packed hidden/routing/srctok), sized to the worst case.
    from common import size_mb
    specs = [(G * cap * H, torch.bfloat16), (G * cap * K, torch.int64), (G * cap, torch.int32)]
    total_mb = sum(size_mb([nu], dt) for nu, dt in specs) + 4
    buf = SymmetricMemoryManager.get_buffer(
        "hier_col_k1", process_group=col_group, size_mb=total_mb).maybe_get_tensors(specs)
    if buf["handle"] is None:
        return [("K1 scatter", False, "column symm-mem alloc failed")]
    hdl = buf["handle"]
    (raw_h, h_off), (raw_r, r_off), (raw_s, s_off) = buf["tensors"]
    recv_h = raw_h.view(torch.bfloat16).view(G * cap, H)
    recv_r = raw_r.view(torch.int64).view(G * cap, K)
    recv_s = raw_s.view(torch.int32).view(G * cap)
    recv_h.zero_(); recv_r.zero_(); recv_s.fill_(-1)
    torch.cuda.synchronize()
    dist.barrier(col_group)

    directed_a2a_scatter(
        hdl.buffer_ptrs_dev, hdl.signal_pad_ptrs_dev,
        hidden.contiguous(), routing.contiguous(), srctok.contiguous(),
        send_dest, send_tok, send_row,
        h_off, r_off, s_off,
        my_group=my_group, gcol=G, hidden_size=H, topk=K)
    torch.cuda.synchronize()
    dist.barrier(col_group)

    # Oracle: gather every column source's inputs, replay the plan, reconstruct expected.
    H_all = torch.empty(G * n, H, dtype=torch.bfloat16, device=dev)
    R_all = torch.empty(G * n, K, dtype=torch.int64, device=dev)
    dist.all_gather_into_tensor(H_all, hidden.contiguous(), group=col_group)
    dist.all_gather_into_tensor(R_all, routing.contiguous(), group=col_group)
    H_all, R_all = H_all.view(G, n, H), R_all.view(G, n, K)

    R_me = recv_counts[my_group]
    exp_h = torch.zeros(R_me, H, dtype=torch.bfloat16, device=dev)
    exp_r = torch.zeros(R_me, K, dtype=torch.int64, device=dev)
    exp_s = torch.full((R_me,), -1, dtype=torch.int32, device=dev)
    for gs in range(G):
        sends_gs, _ = outer_send_plan(_groups_per_token(R_all[gs], placement), M.tolist(), gs)
        for d, t, row in sends_gs:
            if d == my_group:
                exp_h[row] = H_all[gs, t]
                exp_r[row] = R_all[gs, t]
                exp_s[row] = t
    ok = (torch.equal(recv_h[:R_me], exp_h) and torch.equal(recv_r[:R_me], exp_r)
          and torch.equal(recv_s[:R_me], exp_s))
    bad_h = int((recv_h[:R_me] != exp_h).any(dim=1).sum())
    detail = f"G={G} g={g} n={n} R[my_group]={R_me} sends={len(sends)} mismatched_rows(hidden)={bad_h}"
    return [(f"K1 directed-a2a scatter (rank in group {my_group})", ok, detail)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--milestone",
                   choices=["k0", "k1", "k2", "k3overlap", "decompose", "k3fused"], default="k1")
    p.add_argument("--g", type=int, default=2, help="inner-group size (g); G = world//g")
    p.add_argument("--num-tokens", type=int, default=3, help="tokens per rank (k0/k1)")
    p.add_argument("--batch-sizes", default="128,1024,8192", help="k3overlap/decompose: global B values")
    p.add_argument("--topk", type=int, default=22, help="experts per token (decompose)")
    p.add_argument("--num-layers", type=int, default=88, help="layers/step (decompose)")
    p.add_argument("--reps", type=int, default=20, help="graph replays timed (decompose)")
    p.add_argument("--scatter-ctas", type=int, default=96, help="producer CTAs in fused dispatch")
    p.add_argument("--no-flags", action="store_true",
                   help="DIAGNOSTIC: skip fused-dispatch flag handshake (bisect hang location)")
    p.add_argument("--skip-wait", action="store_true",
                   help="DIAGNOSTIC: producer sends flags but consumer skips the wait")
    p.add_argument("--wait-iters", type=int, default=50000,
                   help="DIAGNOSTIC: consumer flag-poll iterations (1 = single poll, no spin)")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    group, rank, world, local_rank = init_distributed()
    cap = max(2, args.num_tokens)
    if args.milestone in ("k3overlap", "decompose", "k3fused"):
        cap = max(2, -(-max(int(x) for x in args.batch_sizes.split(",")) // world))
    cfg = Config(ep_size=world, rank=rank, local_rank=local_rank, seed=args.seed,
                 per_rank_cap=cap, topk=args.topk)
    grid = HierGrid(world, args.g, rank)

    if args.milestone in ("k0", "k1"):
        row_group, col_group = make_groups(grid)
        dist.barrier(group)
        results = (test_k1_scatter if args.milestone == "k1" else test_k0_inner)(
            cfg, grid, row_group, col_group, args)
    elif args.milestone == "k3overlap":
        dist.barrier(group)
        results = test_k3_overlap(cfg, grid, args)
    elif args.milestone == "decompose":
        dist.barrier(group)
        results = test_decompose(cfg, grid, args)
    elif args.milestone == "k3fused":
        dist.barrier(group)
        results = test_k3_fused(cfg, grid, args)
    else:  # k2 — HierBencher builds its own subgroups
        dist.barrier(group)
        results = test_k2_roundtrip(cfg, grid, args)

    local_ok = True
    for name, ok, detail in results:
        local_ok = local_ok and ok
        if rank == 0 or not ok:
            print(f"  [rank{rank}] [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    verdict = torch.tensor([1.0 if local_ok else 0.0], device=torch.device("cuda", local_rank))
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        print(f"# {args.milestone.upper()} {'PASSED' if verdict.item() > 0.5 else 'FAILED'} "
              f"(all {world} ranks)", flush=True)
    dist.barrier(group)
    dist.destroy_process_group()
    sys.exit(0 if verdict.item() > 0.5 else 1)


if __name__ == "__main__":
    main()
