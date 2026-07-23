#!/usr/bin/env python3
# Copyright (c) 2026. Feasibility probe: nested per-token multicast groups.
"""
Can a rank participate in the full NESTED set of NVLS multicast groups at once?

Idea (collaborator): instead of one flat multicast to all P ranks, pre-create nested
aligned multicast subteams -- size 4 {0-3},{4-7},...; size 8 {0-7},...; ... up to {0..P-1}
-- and per token pick the SMALLEST group spanning its destination ranks. The gating
feasibility question is whether the HARDWARE (NVSwitch multicast) + torch symm-mem let a
rank hold multiple overlapping multicast groups. We already do N=2/rank (row+col); this
stresses it to the full nested set (~5/rank, 31 total at P=64).

This probe: create every nested aligned group via dist.new_group, rendezvous a small symm
buffer over each (members only), and check EVERY rank gets a valid multicast_ptr for all
the groups it joined. Reports where it breaks if the hardware group-count ceiling is hit.

  torchrun --nproc_per_node=4  probe_nested_mcast.py            # EP=4 (mechanism)
  <multinode> torchrun ...     probe_nested_mcast.py --min-size 4   # EP=64 (count stress)
"""
import argparse
import os

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/triton_cache_{os.environ.get('SLURM_JOB_ID', '0')}_{os.environ.get('RANK', '0')}")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist

from nvls.symmetric_memory import SymmetricMemoryBuffer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-size", type=int, default=2, help="smallest nested group size")
    p.add_argument("--buf-mb", type=int, default=4, help="symm buffer size per group (MB)")
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.barrier()

    # Nested aligned group sizes: min_size, 2*min_size, ..., world.
    sizes = []
    s = args.min_size
    while s <= world:
        sizes.append(s)
        s *= 2

    # Build every nested aligned group (ALL ranks call new_group for each, in the same order).
    groups = []                      # (size, start, members, pg)
    for s in sizes:
        for start in range(0, world, s):
            members = list(range(start, start + s))
            pg = dist.new_group(members)
            groups.append((s, start, members, pg))
    total = len(groups)
    if rank == 0:
        print(f"# world={world} nested sizes={sizes} total_groups={total} "
              f"(each rank joins {len(sizes)})", flush=True)

    # Rendezvous each group this rank is a member of; check its multicast_ptr.
    joined = valid = 0
    detail = []
    first_fail = None
    for s, start, members, pg in groups:
        if rank not in members:
            continue
        joined += 1
        b = SymmetricMemoryBuffer(size_in_mb=args.buf_mb, process_group=pg)
        if b.symm_mem_hdl is None:
            detail.append(f"s{s}@{start}:RDZV_FAIL")
            if first_fail is None:
                first_fail = (s, start, b.init_failure_reason)
            continue
        mc = getattr(b.symm_mem_hdl, "multicast_ptr", None)
        ok = mc is not None and int(mc) != 0
        if ok:
            valid += 1
            detail.append(f"s{s}@{start}:OK")
        else:
            detail.append(f"s{s}@{start}:NO_MC")
            if first_fail is None:
                first_fail = (s, start, "multicast_ptr is null/0")

    dist.barrier()
    # Per-rank report (serialized so lines don't interleave).
    for r in range(world):
        if rank == r:
            msg = f"  [rank{rank:2d}] joined={joined} valid_mcast={valid}  " + " ".join(detail)
            if first_fail is not None:
                msg += f"  FIRST_FAIL=s{first_fail[0]}@{first_fail[1]}({first_fail[2]})"
            print(msg, flush=True)
        dist.barrier()

    allok = torch.tensor([1.0 if valid == joined and joined == len(sizes) else 0.0], device=dev)
    dist.all_reduce(allok, op=dist.ReduceOp.MIN)
    # Also gather the min valid count to see how far it got before any ceiling.
    minvalid = torch.tensor([float(valid)], device=dev)
    dist.all_reduce(minvalid, op=dist.ReduceOp.MIN)
    if rank == 0:
        ok = allok.item() > 0.5
        print(f"# NESTED-MCAST PROBE {'PASSED' if ok else 'FAILED'}: "
              f"{total} total groups, each rank should join {len(sizes)}; "
              f"min valid_mcast across ranks = {int(minvalid.item())}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if allok.item() > 0.5 else 1)


if __name__ == "__main__":
    main()
