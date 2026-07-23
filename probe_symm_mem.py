#!/usr/bin/env python3
# Copyright (c) 2026. K-1 probe: does the symmetric-memory handle expose per-peer
# DATA pointers usable for a directed (non-multicast) P2P store from a Triton kernel?
"""
Milestone K-1 of the hierarchical-dispatch plan. The whole "outer all-to-all"
hop depends on ONE unverified fact: that we can obtain, for each remote rank, a
device-side base pointer to that rank's symmetric buffer, and that a directed
`st_32(..., multicast_op=False)` to it lands in the peer's memory.

The vendored NVLS code only ever uses `multicast_ptr` (writes to ALL peers) and
`signal_pad_ptrs_dev` (per-peer, but only for the barrier). It never uses a
per-peer DATA pointer array. This probe:

  1. Introspects the `_SymmetricMemory` handle and prints which per-peer
     pointer accessors exist (`buffer_ptrs_dev`, `get_buffer`, `buffer_ptrs`, ...).
  2. Obtains a per-peer base-pointer array (trying, in priority order:
     buffer_ptrs_dev -> get_buffer(r).data_ptr() -> report fallback needed).
  3. Runs a directed-P2P round trip: every rank writes its id into slot [rank]
     of EVERY peer's buffer via a tiny Triton kernel (mirrors the per-remote-rank
     addressing in barrier.py's symm_mem_sync), barriers, then each rank checks
     its buffer == [0, 1, ..., world-1].

Run (inside the container):
    torchrun --nproc_per_node=4 probe_symm_mem.py

Exit 0 = directed-P2P store works and we have the pointer path for K1.
Exit 1 = failed; read the printed attribute table to pick the fallback.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from common import init_distributed  # noqa: E402
from nvls.symmetric_memory import SymmetricMemoryManager  # noqa: E402
from nvls.torch_symm_triton.barrier import symm_mem_sync  # noqa: E402
from nvls.torch_symm_triton.multimem_asm import st_32  # noqa: E402
from nvls.torch_symm_triton.utils import sync_threads  # noqa: E402


@triton.jit
def _directed_scatter_kernel(peer_ptrs, signal_pad_ptrs,
                             RANK: tl.constexpr, WORLD_SIZE: tl.constexpr):
    """Single-CTA. Rank RANK writes value RANK into slot [RANK] of every peer's
    buffer via a directed (non-multicast) store, then a full publish/acquire barrier.

    `peer_ptrs` is the device VA of an array of WORLD_SIZE uint64 base pointers
    (one per rank), exactly analogous to `signal_pad_ptrs`. This mirrors the
    per-remote-rank addressing in barrier.py:107 (`tl.load(signal_pad_ptrs + r)`),
    but targets the DATA buffer instead of the signal pad."""
    if tl.program_id(0) > 0:
        return
    # Triton-3.6 fix: widen the raw pointer int arg to i64 before reinterpreting
    # (a low VA gets specialized as i32; tt.int_to_ptr needs i64). See nvls/metadata.py:79.
    peer_ptrs = peer_ptrs.to(tl.int64).to(tl.pointer_type(tl.uint64))
    mask = tl.full([], 1, dtype=tl.int1)
    val = tl.full([], RANK, dtype=tl.uint32)
    for dst in tl.static_range(WORLD_SIZE):
        # base = peer `dst`'s buffer base VA (a uint64 value loaded from the array).
        base = tl.load(peer_ptrs + dst).to(tl.pointer_type(tl.uint32))
        st_32(base + RANK, val, mask, multicast_op=False)   # directed P2P store
    sync_threads()
    symm_mem_sync(signal_pad_ptrs, None, RANK, WORLD_SIZE,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)


def _introspect(hdl, rank):
    """Print the handle's attribute surface (rank 0), and return a dict of which
    per-peer pointer accessors are present."""
    candidates = ["multicast_ptr", "signal_pad_ptrs_dev", "buffer_ptrs_dev",
                  "buffer_ptrs", "get_buffer", "rank", "world_size"]
    present = {c: hasattr(hdl, c) for c in candidates}
    if rank == 0:
        print("# [K-1] _SymmetricMemory handle type:", type(hdl), flush=True)
        print("# [K-1] attribute presence:", flush=True)
        for c in candidates:
            v = getattr(hdl, c, None)
            shown = v if (c in ("rank", "world_size") or not present[c]) else "<present>"
            print(f"#   {c:22s} present={present[c]!s:5s} value={shown}", flush=True)
        pubattrs = [a for a in dir(hdl) if not a.startswith("__")]
        print(f"# [K-1] full dir(hdl): {pubattrs}", flush=True)
    return present


def _get_peer_ptr_array(hdl, world, device, dtype, shape):
    """Return (peer_ptrs_va, keepalive, source_desc). peer_ptrs_va is the device VA
    of a [world] uint64 array of per-rank buffer base pointers, or None if we can't
    build one (the fallback branch — flag to user)."""
    # (1) Primary: buffer_ptrs_dev is already a device array of per-peer base pointers
    #     (the exact analog of signal_pad_ptrs_dev). Pass its VA straight in.
    bpd = getattr(hdl, "buffer_ptrs_dev", None)
    if bpd is not None:
        try:
            return int(bpd), None, "buffer_ptrs_dev"
        except (TypeError, ValueError):
            pass  # not an int VA; fall through

    # (2) Fallback A: build the array ourselves from get_buffer(rank).data_ptr().
    if hasattr(hdl, "get_buffer"):
        for call in (
            lambda r: hdl.get_buffer(r, shape, dtype, 0),
            lambda r: hdl.get_buffer(r, shape, dtype),
            lambda r: hdl.get_buffer(r, list(shape), dtype),
        ):
            try:
                ptrs = [call(r).data_ptr() for r in range(world)]
                arr = torch.tensor(ptrs, dtype=torch.int64, device=device)
                return arr.data_ptr(), arr, "get_buffer().data_ptr()"
            except (TypeError, RuntimeError):
                continue

    # (3) No direct per-peer data pointer -> fallback B (cudaIpc / NCCL device API).
    return None, None, None


def main():
    group, rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)

    # One symmetric int32 buffer of `world` slots (carved at offset 0 of the symm
    # allocation, so buffer base == &recv[0]). Sentinel -1 marks unwritten slots.
    shape, dtype = [world], torch.int32
    buf = SymmetricMemoryManager.get_buffer(
        "probe", process_group=group, size_mb=1).maybe_get_tensor(shape, dtype=dtype)
    if buf["handle"] is None:
        if rank == 0:
            print("# [K-1] FAIL: symmetric memory init failed (need NVLink domain + "
                  "torch.distributed._symmetric_memory + multicast + triton).", flush=True)
        dist.destroy_process_group()
        sys.exit(1)
    recv, hdl = buf["tensor"], buf["handle"]

    present = _introspect(hdl, rank)
    peer_va, keepalive, src = _get_peer_ptr_array(hdl, world, device, dtype, shape)  # noqa: F841

    if peer_va is None:
        if rank == 0:
            print("# [K-1] FAIL: no per-peer DATA pointer available on the handle "
                  f"(buffer_ptrs_dev present={present['buffer_ptrs_dev']}, "
                  f"get_buffer present={present['get_buffer']}). "
                  "Fallback B (cudaIpc / NCCL device API) needed — scope change; consult plan.",
                  flush=True)
        dist.barrier(group)
        dist.destroy_process_group()
        sys.exit(1)
    if rank == 0:
        print(f"# [K-1] using per-peer pointer source: {src}", flush=True)

    # Init sentinel, fence across ranks so no peer writes before all inits land.
    recv.fill_(-1)
    torch.cuda.synchronize()
    dist.barrier(group)

    _directed_scatter_kernel[(1, 1, 1)](
        peer_va, hdl.signal_pad_ptrs_dev, RANK=rank, WORLD_SIZE=world, num_warps=1)
    torch.cuda.synchronize()
    dist.barrier(group)

    expected = torch.arange(world, device=device, dtype=dtype)
    ok = bool(torch.equal(recv, expected))
    got = recv.tolist()
    print(f"#   [rank{rank}] directed-P2P recv={got} expected={expected.tolist()} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    verdict = torch.tensor([1.0 if ok else 0.0], device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        good = verdict.item() > 0.5
        print(f"# [K-1] {'PASSED' if good else 'FAILED'} (all {world} ranks). "
              f"Directed-P2P store via '{src}' "
              f"{'works — K1 pointer path confirmed.' if good else 'did NOT round-trip.'}",
              flush=True)
    dist.barrier(group)
    dist.destroy_process_group()
    sys.exit(0 if verdict.item() > 0.5 else 1)


if __name__ == "__main__":
    main()
