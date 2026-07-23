#!/usr/bin/env python3
# Copyright (c) 2026. Feasibility probe: ONE symm buffer aliased under MULTIPLE nested multicast VAs.
"""
Gate for the one-buffer variable-multicast design: can a SINGLE physical symmetric-memory buffer be
rendezvous'd over several nested process groups (size-2 {0,1}.., size-4 {0-3}.., ... size-P {0..P-1})
so each yields a DISTINCT valid multicast_ptr that ALIASES the same allocation?

If yes: a token can be multicast to just its group's ranks while landing at its fixed global row in the
ONE shared buffer -- exactly the design (no per-group buffers, no per-layer offset/scan; only tok_size
per layer). If no (torch rejects re-rendezvous of one buffer): we'd need the driver multicast API.

Stage 1 (the gate): rendezvous one buffer over every nested group this rank joins; check each returns a
distinct, valid multicast_ptr and the SAME local buffer base (aliasing).
Stage 2 (reach): rank 0 multicast-stores a tag via the size-2 VA (offset 0) and via the size-P VA
(offset 1); every rank reads its LOCAL buffer. Expect size-2 tag on ranks {0,1} only, size-P tag on all.

Run: torchrun --nproc_per_node=4 probe_aliased_mcast.py     (EP=4; also try 8/16/64 via nodes)
"""
import os

import torch
import torch.distributed as dist

import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from common import init_distributed
from nvls.torch_symm_triton.multimem_asm import st_32

NBYTES = 4 * 1024 * 1024  # 4 MB scratch buffer


@triton.jit
def _mc_store_kernel(mc_ptr, VAL: tl.constexpr, OFF: tl.constexpr):
    """Multicast-store the constant VAL to uint32 slot OFF of the group's buffer (all members)."""
    mc_ptr = mc_ptr.to(tl.int64)
    tid = tl.arange(0, 1)
    p = mc_ptr.to(tl.pointer_type(tl.uint32)) + OFF + tid
    v = (tid * 0 + VAL).to(tl.uint32)
    st_32(p, v, mask=(tid < 1), multicast_op=True)


def nested_groups(P, rank):
    """All ranks create every aligned nested group in the same order; return the ones THIS rank joins."""
    sizes, s = [], 2
    while s <= P:
        sizes.append(s)
        s *= 2
    mine = []
    for s in sizes:
        for start in range(0, P, s):
            members = list(range(start, start + s))
            pg = dist.new_group(members)
            if rank in members:
                mine.append((s, start, pg, members))
    return mine


def main():
    group, rank, world, local_rank = init_distributed()
    P = world
    dev = torch.device("cuda", local_rank)
    mine = nested_groups(P, rank)

    # Enable the symm allocator for every group we'll rendezvous over, then allocate ONE buffer.
    for _s, _st, pg, _m in mine:
        symm_mem.enable_symm_mem_for_group(pg.group_name)
    buf = symm_mem.empty(NBYTES, dtype=torch.uint8, device=dev)

    # ---- Stage 1: rendezvous the SAME buffer over each nested group ------------------------------
    handles = {}
    fail = None
    for s, start, pg, members in mine:
        try:
            handles[s] = symm_mem.rendezvous(buf, pg)
        except Exception as e:  # noqa: BLE001
            fail = f"rendezvous FAILED size={s} start={start}: {type(e).__name__}: {e}"
            print(f"[rank{rank}] {fail}", flush=True)
            break

    ok_stage1 = fail is None and len(handles) == len(mine)
    if ok_stage1:
        mcs = {s: int(h.multicast_ptr) for s, h in handles.items()}
        local_bases = {s: int(h.buffer_ptrs[h.rank]) for s, h in handles.items()
                       if hasattr(h, "buffer_ptrs")}
        distinct_mc = len(set(mcs.values())) == len(mcs)
        valid_mc = all(v != 0 for v in mcs.values())
        aliased = (len(set(local_bases.values())) == 1) if local_bases else None
        print(f"[rank{rank}] sizes={list(mcs)} mc_ptrs_distinct={distinct_mc} mc_all_valid={valid_mc} "
              f"local_base_aliased={aliased} mc={{"
              + ", ".join(f'{s}:{hex(v)}' for s, v in mcs.items()) + "}", flush=True)
        ok_stage1 = distinct_mc and valid_mc and (aliased is not False)

    # Reduce the stage-1 verdict across all ranks.
    t = torch.tensor([1 if ok_stage1 else 0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if int(t.item()) == 0:
        if rank == 0:
            print("# STAGE 1 FAILED: one buffer cannot alias multiple multicast VAs on this torch. "
                  "=> need the driver multicast API (cuMulticastBindMem) for the one-buffer design.",
                  flush=True)
        return

    # ---- Stage 2: reach test -- size-2 VA hits {0,1} only; size-P VA hits all --------------------
    buf.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    view = buf.view(torch.int32)

    s_small = min(handles)      # smallest group size this rank is in (2)
    if rank == 0:               # single writer avoids multicast write collisions
        _mc_store_kernel[(1, 1, 1)](handles[s_small].multicast_ptr, VAL=42, OFF=0)   # size-2 VA
        _mc_store_kernel[(1, 1, 1)](handles[P].multicast_ptr,       VAL=84, OFF=1)   # size-P VA
    torch.cuda.synchronize()
    dist.barrier()

    got_small = int(view[0].item())   # 42 iff this rank received the size-2 multicast from rank 0
    got_full = int(view[1].item())    # 84 iff this rank received the size-P multicast from rank 0
    exp_small = 42 if rank in (0, 1) else 0      # rank 0's size-2 group is {0,1}
    exp_full = 84                                 # size-P reaches everyone
    ok2 = (got_small == exp_small) and (got_full == exp_full)
    print(f"[rank{rank}] reach: size-2 slot={got_small}(exp {exp_small}) "
          f"size-{P} slot={got_full}(exp {exp_full}) -> {'OK' if ok2 else 'MISMATCH'}", flush=True)

    t2 = torch.tensor([1 if ok2 else 0], device=dev)
    dist.all_reduce(t2, op=dist.ReduceOp.MIN)
    if rank == 0:
        verdict = int(t2.item()) == 1
        print(f"# ALIASED-MULTICAST PROBE {'PASSED' if verdict else 'FAILED'} (all {world} ranks): "
              f"one buffer, {len(mine)} nested multicast VAs, group-scoped reach "
              f"{'confirmed' if verdict else 'WRONG'}.", flush=True)


if __name__ == "__main__":
    main()
